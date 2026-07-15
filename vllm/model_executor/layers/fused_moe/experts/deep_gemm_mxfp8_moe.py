# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M_and_alignment,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton as fused_silu_mul_fp8_quant_packed,
)
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
    mk_alignment_scope,
    per_block_cast_to_fp8,
    transform_sf_into_required_layout,
)


def requant_weight_for_dg_mxfp8_32(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint float-scale FP8 -> DeepGEMM MXFP8 GranK=32 weight contract.

    Dequants each expert with the checkpoint (128, 128) float scales, requants
    per-row (1, 32) with UE8M0 scales, and packs the scales through DeepGEMM's
    ``transform_sf_into_required_layout`` (recipe (1, 1, 32)). Runs per expert
    to bound the fp32 dequant peak to one expert.
    """
    assert is_deep_gemm_e8m0_used(), (
        "deep_gemm_mxfp8_32 requires DeepGEMM UE8M0 (VLLM_USE_DEEP_GEMM_E8M0=1); "
        "transform_sf_into_required_layout rejects int-packed scales otherwise"
    )
    gran_k = 32
    block = 128
    E, N, K = weight.shape
    n_blocks, k_blocks = weight_scale.shape[-2:]
    assert (n_blocks, k_blocks) == (-(-N // block), -(-K // block)), (
        "weight_scale must be checkpoint (128,128)-block layout"
    )
    assert K % gran_k == 0
    w_new = torch.empty_like(weight)
    sf = torch.empty(E, N, K // gran_k, dtype=torch.float32, device=weight.device)
    for e in range(E):
        s_exp = (
            weight_scale[e]
            .float()
            .repeat_interleave(block, dim=0)[:N]
            .repeat_interleave(block, dim=1)[:, :K]
        )
        w_dq = weight[e].float() * s_exp
        w_new[e], sf[e] = per_block_cast_to_fp8(w_dq, [1, gran_k], use_ue8m0=True)
    w_scale = transform_sf_into_required_layout(
        sf, mn=N, k=K, recipe=(1, 1, gran_k), num_groups=E, is_sfa=False
    )
    return w_new, w_scale


class DeepGemmMxfp8Gran32Experts(DeepGemmExperts):
    """DeepGEMM m-grouped FP8 experts at MXFP8 (1, 32) UE8M0 granularity.

    Comparison column for cute_sm120_mxfp8_32: same OCP GranK=32 recipe, DG
    kernel implementation. Weights are requantized at load time by
    ``requant_weight_for_dg_mxfp8_32`` (int32-packed UE8M0 b_scales in DG
    layout). Activations are quantized here from bf16 with (1, 32) UE8M0
    scales (``expects_unquantized_inputs``); the int32 packing happens inside
    ``deepgemm_moe_permute``, then the standard DG contiguous grouped GEMM
    runs with recipe (1, 32).
    """

    _GRAN_K = 32

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Only the fused gated triton kernel supports group_size=32; the
        # unfused fallback's packed CUDA quant op is hard-locked to 128.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )
        return fused_silu_mul_fp8_quant_packed(
            input=input,
            output_q=output,
            group_size=self._GRAN_K,
            clamp_limit=self.gemm1_clamp_limit,
            alpha=self.gemm1_alpha,
            beta=self.gemm1_beta,
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
        assert a1q_scale is None and a2_scale is None
        assert hidden_states.dtype == torch.bfloat16
        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.w1_scale.dtype == torch.int32, (
            "DeepGemmMxfp8Gran32Experts expects load-time requantized b_scales"
        )

        _, N, K = w1.size()
        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts
        assert w2.size(1) == K

        a1q, a1s = per_token_group_quant_fp8(
            hidden_states, self._GRAN_K, use_ue8m0=True
        )
        # fp32 power-of-two scales -> UE8M0 exponent bytes: deepgemm_moe_permute
        # keys its int32-packing branch on uint8 scale dtype.
        a1q_scale = (a1s.contiguous().view(torch.int32) >> 23).to(torch.uint8)

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        a1q, a1q_scale, expert_ids, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            block_size=self._GRAN_K,
        )
        assert a1q.size(0) == M_sum

        gemm_kwargs = {
            "recipe_a": (1, self._GRAN_K),
            "recipe_b": (1, self._GRAN_K),
        }

        with mk_alignment_scope(align_used):
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_fp8_gemm_nt_contiguous(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                expert_ids,
                **gemm_kwargs,
            )

            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn),
                (M_sum, activation_out_dim),
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_fp8_gemm_nt_contiguous(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                mm2_out,
                expert_ids,
                **gemm_kwargs,
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
