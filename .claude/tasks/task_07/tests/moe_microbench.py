# SPDX-License-Identifier: Apache-2.0
"""Decode-shaped (M=1) MoE GEMM microbench for NCU isolation.

Drives ONE backend's experts.apply() at a single decode step (M=1 token, topk
experts) with real 35B MoE dims, so ncu can isolate the MoE GEMM kernel at
m_per_expert≈1 with a controlled shape (no cudagraph, no prefill confound —
resolves exp_07 plan G3/G4). Run one backend per process:

  python moe_microbench.py <cute|dg> <E> <topk> <hidden> <inter> [iters]

Under ncu, filter --kernel-name to the MoE GEMM kernel + --launch-count to
grab the steady (post-warmup) invocation. Set DG_PRINT_CONFIGS=1 for dg to
print the JIT-selected block_m/block_n/num_stages.
"""

import sys

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.cute_sm120_moe import (
    CuteMxfp8Gran32Experts,
    requant_weight_for_cute_mxfp8,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_mxfp8_moe import (
    DeepGemmMxfp8Gran32Experts,
    requant_weight_for_dg_mxfp8_32,
)
from vllm.utils.deep_gemm import per_block_cast_to_fp8
from vllm.v1.worker.workspace import init_workspace_manager

BLOCK = 128
GRAN = 32


def make_ckpt(e, two_n, k):
    w = torch.empty(e, two_n, k, device="cuda", dtype=torch.float8_e4m3fn)
    ws = torch.empty(
        e, -(-two_n // BLOCK), -(-k // BLOCK), device="cuda", dtype=torch.float32
    )
    for i in range(e):
        b = torch.randn(two_n, k, device="cuda", dtype=torch.bfloat16) / 10
        w[i], ws[i] = per_block_cast_to_fp8(b, [BLOCK, BLOCK], use_ue8m0=False)
    return w, ws


def main():
    backend = sys.argv[1]
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    hidden = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    inter = int(sys.argv[5]) if len(sys.argv) > 5 else 768
    iters = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    M = 1  # single decode token

    torch.manual_seed(0)
    init_workspace_manager(torch.device("cuda:0"))

    n, k = inter, hidden
    w1, w1s = make_ckpt(E, 2 * n, k)   # gate+up
    w2, w2s = make_ckpt(E, k, n)       # down
    # self-document the real per-GEMM MNK at decode (m_per_expert≈1):
    # GEMM1 (gate+up): N=2*inter, K=hidden ; GEMM2 (down): N=hidden, K=inter
    print(f"MoE_MNK decode m_per_expert≈{max(M*topk//E,1)} | "
          f"GEMM1 N={2*n} K={k} | GEMM2 N={k} K={n} | E={E} topk={topk}",
          flush=True)

    if backend == "cute":
        w1q, b1 = requant_weight_for_cute_mxfp8(w1, w1s, GRAN)
        w2q, b2 = requant_weight_for_cute_mxfp8(w2, w2s, GRAN)
        cls = CuteMxfp8Gran32Experts
    else:
        w1q, b1 = requant_weight_for_dg_mxfp8_32(w1, w1s)
        w2q, b2 = requant_weight_for_dg_mxfp8_32(w2, w2s)
        cls = DeepGemmMxfp8Gran32Experts

    tokens = torch.randn(M, k, device="cuda", dtype=torch.bfloat16).clamp_(-1, 1)
    logits = torch.randn(M, E, device="cuda", dtype=torch.float32)
    tw, tid = torch.topk(logits, k=topk, dim=-1)
    tw = torch.softmax(tw, dim=-1)

    qc = fp8_w8a8_moe_quant_config(
        w1_scale=b1, w2_scale=b2, block_shape=[BLOCK, BLOCK]
    )
    moe = make_dummy_moe_config()
    kernel = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe, quant_config=qc, allow_new_interface=True, use_monolithic=False
        ),
        fused_experts=cls(moe_config=moe, quant_config=qc),
    )

    def one():
        return kernel.apply(
            hidden_states=tokens, w1=w1q, w2=w2q, topk_weights=tw, topk_ids=tid,
            global_num_experts=E, activation=MoEActivation.SILU,
            apply_router_weight_on_input=False, expert_map=None,
        )

    for _ in range(3):  # warmup (JIT + cache)
        one()
    torch.cuda.synchronize()
    for _ in range(iters):  # steady-state, ncu grabs these
        one()
    torch.cuda.synchronize()
    print(f"MICROBENCH_DONE backend={backend} E={E} topk={topk} "
          f"hidden={hidden} inter={inter} M={M}", flush=True)


if __name__ == "__main__":
    main()
