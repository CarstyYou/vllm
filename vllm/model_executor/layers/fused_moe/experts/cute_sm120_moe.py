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
from vllm.utils.deep_gemm import pack_ue8m0_to_int, per_block_cast_to_fp8
from vllm.utils.math_utils import round_up



@cache
def _has_flashinfer_cute_fp8() -> bool:
    try:
        return importlib.util.find_spec("flashinfer.grouped_mm") is not None
    except ModuleNotFoundError:
        return False


def _fi_moe_gemm_mxfp8_nt_groupwise():
    from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise

    return moe_gemm_mxfp8_nt_groupwise


def _pack_a_scale_ue8m0(sf_pow2: torch.Tensor) -> torch.Tensor:
    """Pack per-token power-of-two fp32 scales (M, S) into int32 (M, ceil(S/4)),
    4 UE8M0 exponent bytes per int32 (DeepGEMM pack_ue8m0_to_int semantics)."""
    M, S = sf_pow2.shape
    pad = (-S) % 4
    if pad:
        sf_pow2 = torch.cat(
            [
                sf_pow2,
                torch.zeros(M, pad, dtype=sf_pow2.dtype, device=sf_pow2.device),
            ],
            dim=1,
        )
    sf_c = sf_pow2.contiguous()
    return (sf_c.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)


def _repack_packed_a_scale_for_fi(
    packed: torch.Tensor,
    expert_ids: torch.Tensor,
    m_indptr: torch.Tensor,
    m_padded: int,
) -> torch.Tensor:
    """Scatter int32-packed per-token scales (M_sum, k_align) into the FlashInfer
    MXFP8 zero-padding storage (k_align, m_padded); returns the transposed view
    the entry consumes (logical (m_padded, k_align))."""
    M_sum, k_align = packed.shape
    device = packed.device
    out = torch.zeros((k_align, m_padded), dtype=torch.int32, device=device)
    starts = m_indptr[:-1].to(torch.int64)
    expert_arange = torch.arange(starts.numel(), device=device, dtype=torch.int64)
    aligned_starts = (starts + 3 * expert_arange) // 4 * 4
    row = torch.arange(M_sum, device=device, dtype=torch.int64)
    expert_of_row = expert_ids.to(torch.int64)
    col = aligned_starts[expert_of_row] + (row - starts[expert_of_row])
    out[:, col] = packed.t()
    return out.transpose(0, 1)


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


def requant_weight_for_cute_mxfp8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    gran_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint float-scale FP8 -> FlashInfer MXFP8 weight contract.

    gran_k=128 (3a): per-(128, 128)-block UE8M0 requant (DeepGEMM semantics),
    then N-broadcast to per-token. gran_k=32 (3b): per-row (1, 32) UE8M0
    requant (OCP). Returns the requantized fp8 weight (E, N, K) and the
    int32-packed per-token b_scale as a logical (E, N, k_align) view over
    MN-major (E, k_align, N) storage (the kernel's deduce_sfb_layout stride
    (1, n, n*k_align) contract), k_align = ceil(K/(4*gran_k)).
    Runs per expert to bound the fp32 dequant peak to one expert.
    """
    assert gran_k in (128, 32)
    E, N, K = weight.shape
    block = 128
    n_blocks, k_blocks = weight_scale.shape[-2:]
    assert (n_blocks, k_blocks) == (-(-N // block), -(-K // block)), (
        "weight_scale must be checkpoint (128,128)-block layout"
    )
    w_new = torch.empty_like(weight)
    k_align = (K + 4 * gran_k - 1) // (4 * gran_k)
    b_scale = torch.empty(E, k_align, N, dtype=torch.int32, device=weight.device)
    for e in range(E):
        s_exp = (
            weight_scale[e]
            .float()
            .repeat_interleave(block, dim=0)[:N]
            .repeat_interleave(block, dim=1)[:, :K]
        )
        w_dq = weight[e].float() * s_exp
        if gran_k == 128:
            w_q, sf = per_block_cast_to_fp8(w_dq, [block, block], use_ue8m0=True)
            sf_pt = sf.repeat_interleave(block, dim=0)[:N]
        else:
            w_q, sf_pt = per_block_cast_to_fp8(w_dq, [1, gran_k], use_ue8m0=True)
        w_new[e] = w_q
        pad = (-sf_pt.size(1)) % 4
        if pad:
            sf_pt = torch.cat(
                [sf_pt, torch.zeros(N, pad, dtype=sf_pt.dtype, device=sf_pt.device)],
                dim=1,
            )
        b_scale[e] = pack_ue8m0_to_int(sf_pt).t()
    return w_new, b_scale.transpose(1, 2)


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


class CuteMxfp8Experts(CuteFp8Experts):
    """FlashInfer cute SM120 MXFP8 groupwise MoE experts (UE8M0 scales).

    Same zero-padding contract as :class:`CuteFp8Experts` (exact token packing,
    scale-plane-only padding). Weights are requantized at load time by
    ``requant_weight_for_cute_mxfp8`` (quant_config then carries the fp8
    weights and int32-packed per-token b_scales in FlashInfer layout).
    Activations are quantized here from bf16 with (1, _GRAN_K) UE8M0 scales
    (``expects_unquantized_inputs``), avoiding a lossy re-quant of the
    prepare-stage float-scale output.
    """

    _GRAN_K: int | None = None

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        assert self._GRAN_K in (128, 32), "use CuteMxfp8Gran{128,32}Experts"
        super().__init__(moe_config=moe_config, quant_config=quant_config)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def _w_scales_fi(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.w1_scale.dtype == torch.int32, (
            "CuteMxfp8Experts expects load-time requantized int32 b_scales"
        )
        return self.w1_scale, self.w2_scale

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
        assert a1q_scale is None and a2_scale is None
        assert output.dtype == torch.bfloat16
        assert hidden_states.dtype == torch.bfloat16
        assert expert_map is None, "CuteMxfp8Experts supports TP-only"

        moe_gemm = _fi_moe_gemm_mxfp8_nt_groupwise()
        gran_k = self._GRAN_K

        _, N, K = w1.size()
        local_num_experts = w1.size(0)
        assert w2.size(1) == K

        M_sum = topk_ids.numel()
        M_sum_q = round_up(M_sum, self._QUANT_PAD_M)
        m_padded = (M_sum + local_num_experts * 3) // 4 * 4
        device = hidden_states.device

        a1q, a1s = per_token_group_quant_fp8(hidden_states, gran_k, use_ue8m0=True)
        # fp32 power-of-two scales -> UE8M0 exponent bytes for the packing
        # scatter path (mantissa==0 guaranteed by the exp2(ceil(log2)) quant).
        a1s_u8 = (a1s.contiguous().view(torch.int32) >> 23).to(torch.uint8)

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
        sf_k = K // gran_k
        packed_sf_k = (sf_k + 3) // 4
        a1s_perm = torch.empty_strided(
            (M_sum, packed_sf_k),
            (1, round_up(M_sum, 4)),
            dtype=torch.int32,
            device=device,
        )
        expert_start_loc = torch.empty(
            local_num_experts, dtype=torch.int32, device=device
        )
        expert_ids = torch.full((M_sum,), -1, dtype=torch.int32, device=device)
        inv_perm = torch.empty(topk_ids.shape, dtype=torch.int32, device=device)

        ep_scatter(
            recv_x=a1q,
            recv_x_scale=a1s_u8,
            recv_topk=topk_ids,
            num_recv_tokens_per_expert=expert_num_tokens,
            expert_start_loc=expert_start_loc,
            expert_map=expert_map,
            output_tensor=a1q_perm,
            output_tensor_scale=a1s_perm,
            m_indices=expert_ids,
            output_index=inv_perm,
            align_m=1,
            block_size=gran_k,
            pack_ue8m0=True,
        )

        a1s_fi = _repack_packed_a_scale_for_fi(
            a1s_perm, expert_ids, m_indptr, m_padded
        )
        w1_scale_fi, w2_scale_fi = self._w_scales_fi()

        mm1_out = _resize_cache(workspace2, (M_sum_q, N))
        if M_sum_q > M_sum:
            mm1_out[M_sum:].zero_()
        moe_gemm(
            a1q_perm,
            w1,
            a1s_fi,
            w1_scale_fi,
            m_indptr,
            scale_granularity_mnk=(1, 1, gran_k),
            out=mm1_out[:M_sum],
        )

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum_q, activation_out_dim)
        )
        a2q_full, a2s_full = self._act_mul_quant(
            input=mm1_out, output=quant_out, activation=activation
        )
        a2q = a2q_full[:M_sum]
        a2s = a2s_full[:M_sum]

        assert a2s.dim() == 2 and a2s.size(0) == M_sum
        a2s_fi = _repack_packed_a_scale_for_fi(
            _pack_a_scale_ue8m0(a2s), expert_ids, m_indptr, m_padded
        )

        mm2_out = _resize_cache(workspace2, (M_sum, K))
        moe_gemm(
            a2q,
            w2,
            a2s_fi,
            w2_scale_fi,
            m_indptr,
            scale_granularity_mnk=(1, 1, gran_k),
            out=mm2_out,
        )

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

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        if activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE):
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=True,
                clamp_limit=self.gemm1_clamp_limit,
                group_size=self._GRAN_K,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
            )

        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(
            act_out, self._GRAN_K, column_major_scales=True, use_ue8m0=True,
            out_q=output,
        )


class CuteMxfp8Gran128Experts(CuteMxfp8Experts):
    """3a: DeepGEMM-convention (1, 1, 128) UE8M0 granularity."""

    _GRAN_K = 128


class CuteMxfp8Gran32Experts(CuteMxfp8Experts):
    """3b: OCP MXFP8 (1, 1, 32) UE8M0 granularity."""

    _GRAN_K = 32
