# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer


class FlashInferSm12xMxfp4Experts(mk.FusedMoEExpertsModular):
    """CuteDSL MXFP8 x MXFP4 fused MoE experts for SM12x, chain mode."""

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "Swiglu",
        MoEActivation.SITU: "Situ",
    }

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.hidden_dim = moe_config.hidden_dim
        self.hidden_dim_unpadded = (
            moe_config.hidden_dim_unpadded or moe_config.hidden_dim
        )
        self.situ_beta = moe_config.activation_situ_beta
        self.situ_linear_beta = moe_config.activation_situ_linear_beta

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(120) and has_flashinfer()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False  # the FC1 epilogue is gated-only

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SITU)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # TP only: apply()'s routing assumes topk_ids span this rank's experts
        return (
            not moe_parallel_config.use_ep
            and not moe_parallel_config.is_sequence_parallel
        )

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()  # fc2_finalize applies topk weights

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
        # both kernels allocate their own outputs; no preallocated buffers
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, self.hidden_dim_unpadded)
        return (workspace1, workspace2, output)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True  # route + MXFP8 quant happen in apply(), not prepare()

    def _activation_kwargs(self, activation: MoEActivation) -> dict:
        from flashinfer.tllm_enums import ActivationType

        act = ActivationType[self._ACTIVATION_MAP[activation]]
        kwargs: dict = {"activation_type": act.value}
        if act == ActivationType.Situ:
            assert self.situ_beta is not None and self.situ_linear_beta is not None, (
                "SITU requires activation_situ_beta and activation_situ_linear_beta"
            )
            kwargs["situ_beta"] = float(self.situ_beta)
            kwargs["situ_linear_beta"] = float(self.situ_linear_beta)
        return kwargs

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        from flashinfer.fused_moe import cute_dsl_sm12x_fused_moe_mxfp8_mxfp4

        assert expert_map is None
        assert a1q_scale is None, "expects_unquantized_inputs=True"
        assert self.w1_scale is not None and self.w2_scale is not None

        num_experts = global_num_experts if global_num_experts != -1 else w1.size(0)
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        cute_dsl_sm12x_fused_moe_mxfp8_mxfp4(
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights,
            w1,
            self.w1_scale,
            w2,
            self.w2_scale,
            num_experts,
            moe_output=output,
            **self._activation_kwargs(activation),
        )
