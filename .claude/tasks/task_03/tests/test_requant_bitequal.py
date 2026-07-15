# SPDX-License-Identifier: Apache-2.0
"""Bit-equal gate for requant_weight_for_cute_mxfp8 vs FI upstream helpers.

Ground truth uses flashinfer.testing.utils (the same helpers the FI upstream
mxfp8 test packer uses). 3b compares cross-library (vLLM per_block[1,32] vs FI
per_token gran_k=32); 3a validates broadcast+pack against an independent eager
composition. Exits non-zero on any mismatch.
"""

import importlib.util
import sys

import torch

from vllm.model_executor.layers.fused_moe.experts.cute_sm120_moe import (
    requant_weight_for_cute_mxfp8,
)
from vllm.utils.deep_gemm import per_block_cast_to_fp8

_FI_TEST = (
    "/home/scratch.xiy_gpu/mega_inference/flashinfer/tests/grouped_mm/"
    "test_cute_sm120_mxfp8.py"
)
_spec = importlib.util.spec_from_file_location("fi_mxfp8_test", _FI_TEST)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fi_pack = _mod.pack_ue8m0_to_int
fi_per_token_cast = _mod.per_token_cast_to_fp8


def make_checkpoint(e, n, k):
    w = torch.empty(e, n, k, device="cuda", dtype=torch.float8_e4m3fn)
    ws = torch.empty(
        e, -(-n // 128), -(-k // 128), device="cuda", dtype=torch.float32
    )
    for i in range(e):
        bf = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8
        w[i], ws[i] = per_block_cast_to_fp8(bf, [128, 128], use_ue8m0=False)
    return w, ws


def dequant(w_e, ws_e, n, k):
    s = (
        ws_e.float()
        .repeat_interleave(128, dim=0)[:n]
        .repeat_interleave(128, dim=1)[:, :k]
    )
    return w_e.float() * s


def ref_pack(sf_pt, n):
    pad = (-sf_pt.size(1)) % 4
    if pad:
        sf_pt = torch.cat(
            [sf_pt, torch.zeros(n, pad, dtype=sf_pt.dtype, device=sf_pt.device)],
            dim=1,
        )
    return fi_pack(sf_pt)


def check(name, ok):
    print(f"{name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    torch.manual_seed(7)
    e, n, k = 4, 512, 768  # k_blocks=6 (non-multiple-of-4 exercises pad path)
    w, ws = make_checkpoint(e, n, k)
    all_ok = True

    for gran_k in (128, 32):
        w_new, b_scale = requant_weight_for_cute_mxfp8(w, ws, gran_k)
        for i in range(e):
            w_dq = dequant(w[i], ws[i], n, k)
            if gran_k == 128:
                w_ref, sf = per_block_cast_to_fp8(w_dq, [128, 128], use_ue8m0=True)
                sf_pt = sf.repeat_interleave(128, dim=0)[:n]
            else:
                w_ref, sf_pt = fi_per_token_cast(w_dq, use_ue8m0=True, gran_k=32)
            ok_w = torch.equal(
                w_new[i].view(torch.uint8), w_ref.view(torch.uint8)
            )
            ok_s = torch.equal(b_scale[i], ref_pack(sf_pt, n))
            all_ok &= check(f"gran_k={gran_k} expert={i} weight", ok_w)
            all_ok &= check(f"gran_k={gran_k} expert={i} scale", ok_s)

    print("BITEQUAL_ALL_PASS" if all_ok else "BITEQUAL_FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
