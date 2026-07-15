# SPDX-License-Identifier: Apache-2.0
"""Single-layer parity for DeepGemmMxfp8Gran32Experts.

Reference is an eager fp32 MoE pipeline mirroring every quantization boundary
(per-token 1x32 UE8M0 act quant, per-row 1x32 UE8M0 requantized weights, bf16
GEMM1 write-back). The reference recomputes the weight requant chain directly
(dequant -> per_block_cast_to_fp8([1, 32], ue8m0)) instead of unpacking the
DG-transformed scales; the packed layout itself is probe-validated
(probe_dg_grank32.py). Gate: calc_diff < 1e-3 per cell.
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
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_mxfp8_moe import (
    DeepGemmMxfp8Gran32Experts,
    requant_weight_for_dg_mxfp8_32,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts  # noqa: F401
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.deep_gemm import calc_diff, per_block_cast_to_fp8
from vllm.v1.worker.workspace import init_workspace_manager

BLOCK = 128
GRAN = 32


def make_checkpoint_weights(e, n, k):
    w1 = torch.empty(e, 2 * n, k, device="cuda", dtype=torch.float8_e4m3fn)
    w2 = torch.empty(e, k, n, device="cuda", dtype=torch.float8_e4m3fn)
    w1_s = torch.empty(
        e, -(-2 * n // BLOCK), -(-k // BLOCK), device="cuda", dtype=torch.float32
    )
    w2_s = torch.empty(
        e, -(-k // BLOCK), -(-n // BLOCK), device="cuda", dtype=torch.float32
    )
    for i in range(e):
        b1 = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) / 10
        b2 = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / 10
        w1[i], w1_s[i] = per_block_cast_to_fp8(b1, [BLOCK, BLOCK], use_ue8m0=False)
        w2[i], w2_s[i] = per_block_cast_to_fp8(b2, [BLOCK, BLOCK], use_ue8m0=False)
    return w1, w2, w1_s, w2_s


def reference_requant(weight, weight_scale):
    E, N, K = weight.shape
    wq = torch.empty_like(weight)
    sf = torch.empty(E, N, K // GRAN, dtype=torch.float32, device=weight.device)
    for e in range(E):
        s_exp = (
            weight_scale[e]
            .float()
            .repeat_interleave(BLOCK, dim=0)[:N]
            .repeat_interleave(BLOCK, dim=1)[:, :K]
        )
        w_dq = weight[e].float() * s_exp
        wq[e], sf[e] = per_block_cast_to_fp8(w_dq, [1, GRAN], use_ue8m0=True)
    return wq, sf


def dequant_rows(q, sf, k):
    s = sf.repeat_interleave(GRAN, dim=1)[:, :k]
    return q.float() * s


def eager_reference(tokens, w1q, w1_sf, w2q, w2_sf, topk_weights, topk_ids):
    m, k = tokens.shape
    e, n2, _ = w1q.shape
    n = n2 // 2
    a1q, a1s = per_token_group_quant_fp8(tokens, GRAN, use_ue8m0=True)
    a1_dq = dequant_rows(a1q, a1s, k)
    w1_dq = torch.stack([dequant_rows(w1q[i], w1_sf[i], k) for i in range(e)])
    w2_dq = torch.stack([dequant_rows(w2q[i], w2_sf[i], n) for i in range(e)])
    out = torch.zeros(m, k, device=tokens.device, dtype=torch.float32)
    topk = topk_ids.size(1)
    for j in range(topk):
        for i in range(e):
            rows = (topk_ids[:, j] == i).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            mm1 = (a1_dq[rows] @ w1_dq[i].t()).to(torch.bfloat16)
            gate, up = mm1[:, :n].float(), mm1[:, n:].float()
            act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            a2q, a2s = per_token_group_quant_fp8(act, GRAN, use_ue8m0=True)
            a2_dq = dequant_rows(a2q, a2s, n)
            mm2 = a2_dq @ w2_dq[i].t()
            out[rows] += topk_weights[rows, j].unsqueeze(1).float() * mm2
    return out.to(torch.bfloat16)


def run_single_case(m, n, k, topk, num_experts):
    tokens = (
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        .clamp_min_(-1)
        .clamp_max_(1)
    )
    w1, w2, w1_s, w2_s = make_checkpoint_weights(num_experts, n, k)
    w1q, b1_scale = requant_weight_for_dg_mxfp8_32(w1, w1_s)
    w2q, b2_scale = requant_weight_for_dg_mxfp8_32(w2, w2_s)
    w1q_ref, w1_sf = reference_requant(w1, w1_s)
    w2q_ref, w2_sf = reference_requant(w2, w2_s)
    assert torch.equal(w1q, w1q_ref) and torch.equal(w2q, w2q_ref)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=b1_scale, w2_scale=b2_scale, block_shape=[BLOCK, BLOCK]
    )
    moe_config = make_dummy_moe_config()
    kernel = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=DeepGemmMxfp8Gran32Experts(
            moe_config=moe_config, quant_config=quant_config
        ),
    )

    out_kernel = kernel.apply(
        hidden_states=tokens,
        w1=w1q,
        w2=w2q,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )

    out_ref = eager_reference(
        tokens, w1q, w1_sf, w2q, w2_sf, topk_weights, topk_ids
    )
    return calc_diff(out_kernel, out_ref)


def main():
    torch.manual_seed(42)
    init_workspace_manager(torch.device("cuda:0"))

    mnks = [
        (1, 768, 256),
        (7, 768, 512),
        (33, 1024, 1024),
        (128, 2048, 512),
        (1024, 2048, 1024),
    ]
    rows, failed = [], 0
    for m, n, k in mnks:
        for topk, e in ((6, 32), (8, 128)):
            diff = run_single_case(m, n, k, topk, e)
            ok = diff < 1e-3
            failed += 0 if ok else 1
            rows.append(
                {"M": m, "N": n, "K": k, "topk": topk,
                 "E": e, "calc_diff": f"{diff:.3e}", "pass": bool(ok)}
            )
            print(rows[-1], flush=True)

    out_csv = sys.argv[1] if len(sys.argv) > 1 else "dg_mxfp8_parity.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows) - failed}/{len(rows)} passed -> {out_csv}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
