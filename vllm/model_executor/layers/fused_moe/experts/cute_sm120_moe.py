# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from functools import cache

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    count_expert_num_tokens,
    deepgemm_unpermute_and_reduce,
    ep_scatter,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up



@cache
def _has_flashinfer_cute_fp8() -> bool:
    try:
        return importlib.util.find_spec("flashinfer.grouped_mm") is not None
    except ModuleNotFoundError:
        return False


def _fi_moe_gemm_fp8_nt_groupwise():
    from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise

    return moe_gemm_fp8_nt_groupwise


def _repack_a_scale_for_fi(
    a_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    m_indptr: torch.Tensor,
    m_padded: int,
) -> torch.Tensor:
    """Repack per-token scales (M_sum, k_blocks) into the FlashInfer
    zero-padding MN-major layout (k_blocks, m_padded): expert i's columns
    start at (m_indptr[i] + 3 * i) // 4 * 4, padding columns zero-filled.
    """
    M_sum, k_blocks = a_scale.shape
    device = a_scale.device
    out = torch.zeros((k_blocks, m_padded), dtype=torch.float32, device=device)
    starts = m_indptr[:-1].to(torch.int64)
    expert_arange = torch.arange(starts.numel(), device=device, dtype=torch.int64)
    aligned_starts = (starts + 3 * expert_arange) // 4 * 4
    row = torch.arange(M_sum, device=device, dtype=torch.int64)
    expert_of_row = expert_ids.to(torch.int64)
    col = aligned_starts[expert_of_row] + (row - starts[expert_of_row])
    out[:, col] = a_scale.t().to(torch.float32)
    return out


class CuteFp8Experts(mk.FusedMoEExpertsModular):
    """FlashInfer cute SM120 FP8 groupwise grouped-GEMM MoE experts.

    Drives flashinfer's ``moe_gemm_fp8_nt_groupwise`` (float32 (1, 128, 128)
    groupwise scales, zero-padding CSR contract) with exact token packing
    (``align_m=1``, no per-expert M padding). The intermediate act+quant
    stage runs on a 128-row padded, zero-filled view to satisfy the fused
    kernel's alignment; both GEMMs consume exact ``[:M_sum]`` slices.
    """

    # FP8 groupwise recipe block: (1, _BLOCK_K, _BLOCK_K).
    _BLOCK_K = 128
    # M alignment required by silu_mul_per_token_group_quant_fp8_colmajor;
    # unrelated to the kernel recipe, applies only to the quant scratch view.
    _QUANT_PAD_M = 128

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.block_shape == [self._BLOCK_K, self._BLOCK_K]
        assert quant_config.quant_dtype == torch.float8_e4m3fn
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

        assert not envs.VLLM_MOE_SKIP_PADDING, (
            "CuteFp8Experts exact packing assumes no -1 entries in topk_ids"
        )
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self.gemm1_alpha = (
            quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
        )
        self.gemm1_beta = (
            quant_config.gemm1_beta if quant_config.gemm1_beta is not None else 0.0
        )
        # FlashInfer wants (E, k_blocks, n_blocks); checkpoints store
        # (E, n_blocks, k_blocks). Transposed copies are cached lazily.
        self._w1_scale_fi: torch.Tensor | None = None
        self._w2_scale_fi: torch.Tensor | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
            and _has_flashinfer_cute_fp8()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_ep
            or moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim % CuteFp8Experts._BLOCK_K == 0

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The fused act+quant kernel requires M % 128 == 0; FlashInfer GEMMs
        # consume exact [:M_sum] slices of the padded buffers.
        M_sum_q = round_up(M * topk, self._QUANT_PAD_M)
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum_q, max(activation_out_dim, K))
        workspace2 = (M_sum_q, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _w_scales_fi(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w1_scale_fi is None:
            assert self.w1_scale is not None and self.w2_scale is not None
            assert self.w1_scale.dtype == torch.float32
            self._w1_scale_fi = self.w1_scale.transpose(-1, -2).contiguous()
            self._w2_scale_fi = self.w2_scale.transpose(-1, -2).contiguous()
        assert self._w2_scale_fi is not None
        return self._w1_scale_fi, self._w2_scale_fi

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        if activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE):
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=False,
                clamp_limit=self.gemm1_clamp_limit,
                group_size=self._BLOCK_K,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
            )

        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(
            act_out, self._BLOCK_K, column_major_scales=True, out_q=output
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert output.dtype == torch.bfloat16, "FlashInfer FP8 entry emits bf16"
        assert self.w1_scale is not None and self.w2_scale is not None
        # TP-only scope for now: all experts are local and no token is
        # dropped, so exact packing covers every (token, topk) pair.
        assert expert_map is None, "CuteFp8Experts supports TP-only (no expert_map)"

        moe_gemm = _fi_moe_gemm_fp8_nt_groupwise()

        a1q = hidden_states
        _, N, K = w1.size()
        local_num_experts = w1.size(0)
        assert w2.size(1) == K

        M_sum = topk_ids.numel()
        M_sum_q = round_up(M_sum, self._QUANT_PAD_M)
        m_padded = (M_sum + local_num_experts * 3) // 4 * 4
        device = a1q.device

        if expert_tokens_meta is not None:
            expert_num_tokens = expert_tokens_meta.expert_num_tokens
        else:
            expert_num_tokens = count_expert_num_tokens(
                topk_ids, local_num_experts, expert_map
            )

        m_indptr = torch.zeros(local_num_experts + 1, dtype=torch.int32, device=device)
        m_indptr[1:] = torch.cumsum(expert_num_tokens, dim=0)

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        sf_k = K // self._BLOCK_K
        a1q_scale_perm = torch.empty((M_sum, sf_k), dtype=torch.float32, device=device)
        expert_start_loc = torch.empty(
            local_num_experts, dtype=torch.int32, device=device
        )
        expert_ids = torch.full((M_sum,), -1, dtype=torch.int32, device=device)
        inv_perm = torch.empty(topk_ids.shape, dtype=torch.int32, device=device)

        ep_scatter(
            recv_x=a1q,
            recv_x_scale=a1q_scale,
            recv_topk=topk_ids,
            num_recv_tokens_per_expert=expert_num_tokens,
            expert_start_loc=expert_start_loc,
            expert_map=expert_map,
            output_tensor=a1q_perm,
            output_tensor_scale=a1q_scale_perm,
            m_indices=expert_ids,
            output_index=inv_perm,
            align_m=1,
            block_size=self._BLOCK_K,
            pack_ue8m0=False,
        )

        a1q_scale_fi = _repack_a_scale_for_fi(
            a1q_scale_perm, expert_ids, m_indptr, m_padded
        )
        w1_scale_fi, w2_scale_fi = self._w_scales_fi()

        mm1_out = _resize_cache(workspace2, (M_sum_q, N))
        if M_sum_q > M_sum:
            mm1_out[M_sum:].zero_()
        moe_gemm(a1q_perm, w1, a1q_scale_fi, w1_scale_fi, m_indptr, out=mm1_out[:M_sum])

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum_q, activation_out_dim)
        )
        a2q_full, a2q_scale_full = self._act_mul_quant(
            input=mm1_out, output=quant_out, activation=activation
        )
        a2q = a2q_full[:M_sum]
        a2q_scale = a2q_scale_full[:M_sum]

        assert a2q_scale.dim() == 2 and a2q_scale.size(0) == M_sum
        a2q_scale_fi = _repack_a_scale_for_fi(a2q_scale, expert_ids, m_indptr, m_padded)

        mm2_out = _resize_cache(workspace2, (M_sum, K))
        moe_gemm(a2q, w2, a2q_scale_fi, w2_scale_fi, m_indptr, out=mm2_out)

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )
