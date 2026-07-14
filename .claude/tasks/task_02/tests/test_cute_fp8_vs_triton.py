# SPDX-License-Identifier: Apache-2.0
"""Single-layer parity: CuteFp8Experts vs Triton fused_experts (float-scale FP8).

Standalone script (no pytest conftest deps). Writes a CSV of calc_diff per
cell and exits non-zero if any cell fails the 1e-3 gate.
"""

import csv
import sys

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.cute_sm120_moe import CuteFp8Experts
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.deep_gemm import calc_diff, per_block_cast_to_fp8
from vllm.v1.worker.workspace import init_workspace_manager

BLOCK_SIZE = [128, 128]


def make_float_scale_fp8_weights(e: int, n: int, k: int):
    """(w1, w2) FP8 block-quantized with plain float32 scales (no UE8M0)."""
    w1 = torch.empty(e, 2 * n, k, device="cuda", dtype=torch.float8_e4m3fn)
    w2 = torch.empty(e, k, n, device="cuda", dtype=torch.float8_e4m3fn)
    kb, nb2, nb = -(-k // 128), -(-2 * n // 128), -(-n // 128)
    w1_s = torch.empty(e, nb2, kb, device="cuda", dtype=torch.float32)
    w2_s = torch.empty(e, kb, nb, device="cuda", dtype=torch.float32)
    for i in range(e):
        w1_bf16 = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) / 10
        w2_bf16 = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / 10
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1_bf16, BLOCK_SIZE, use_ue8m0=False)
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2_bf16, BLOCK_SIZE, use_ue8m0=False)
    return w1, w2, w1_s, w2_s


def run_single_case(m, n, k, topk, num_experts):
    tokens_bf16 = (
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        .clamp_min_(-1)
        .clamp_max_(1)
    )
    _, a1_scale = per_token_group_quant_fp8(tokens_bf16, BLOCK_SIZE[1])
    w1, w2, w1_s, w2_s = make_float_scale_fp8_weights(num_experts, n, k)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s, w2_scale=w2_s, a1_scale=a1_scale, block_shape=BLOCK_SIZE
    )
    moe_config = make_dummy_moe_config()

    cute_kernel = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=CuteFp8Experts(
            moe_config=moe_config, quant_config=quant_config
        ),
    )

    out_triton = fused_experts(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_config=quant_config,
    )
    out_cute = cute_kernel.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )
    return calc_diff(out_cute, out_triton)


def main():
    torch.manual_seed(42)
    init_workspace_manager(torch.device("cuda:0"))

    mnks = [
        (1, 768, 256),
        (7, 768, 512),
        (33, 1024, 1024),
        (128, 2048, 512),
        (1024, 2048, 1024),
        (4096, 4096, 1024),
    ]
    topks = [6, 8]
    num_experts_list = [32, 128]

    rows, failed = [], 0
    for m, n, k in mnks:
        for topk in topks:
            for e in num_experts_list:
                diff = run_single_case(m, n, k, topk, e)
                ok = diff < 1e-3
                failed += 0 if ok else 1
                rows.append(
                    {"M": m, "N": n, "K": k, "topk": topk, "E": e,
                     "calc_diff": f"{diff:.3e}", "pass": ok}
                )
                print(rows[-1], flush=True)

    out_csv = sys.argv[1] if len(sys.argv) > 1 else "cute_fp8_vs_triton.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows) - failed}/{len(rows)} passed -> {out_csv}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
