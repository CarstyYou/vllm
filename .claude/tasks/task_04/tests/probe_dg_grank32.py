# SPDX-License-Identifier: Apache-2.0
"""Probe: does the vendored DG (nv-dev) sm120 m-grouped FP8 kernel accept granK=32?

Tries recipe=(1, 32, 32) on a tiny m-grouped contiguous case and checks the
output against an eager dequant reference. Prints PROBE_OK / PROBE_FAIL with
the exception, plus calc_diff when it runs.
"""

import torch

import vllm.third_party.deep_gemm as dg
from vllm.utils.deep_gemm import calc_diff


def ceil_to_ue8m0(x):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def per_token_cast(x, gran_k):
    m, k = x.shape
    v = x.view(m, k // gran_k, gran_k)
    amax = v.abs().float().amax(dim=2).clamp(1e-4)
    sf = ceil_to_ue8m0(amax / 448.0)
    q = (v / sf.unsqueeze(2)).to(torch.float8_e4m3fn).view(m, k)
    return q, sf


def per_block_cast(x, gran_k):
    n, k = x.shape
    v = x.view(n // gran_k, gran_k, k // gran_k, gran_k)
    amax = v.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(amax / 448.0)
    q = (v / sf).to(torch.float8_e4m3fn).view(n, k)
    return q, sf.view(n // gran_k, k // gran_k)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    G, M, N, K = 2, 128, 128, 256
    gran = 32

    a = torch.randn(G * M, K, device=dev, dtype=torch.bfloat16) / 10
    b = torch.randn(G, N, K, device=dev, dtype=torch.bfloat16) / 10
    m_indices = torch.arange(G, device=dev, dtype=torch.int32).repeat_interleave(M)

    aq, a_sf = per_token_cast(a, gran)
    bq_l, b_sf_l = zip(*[per_block_cast(b[i], gran) for i in range(G)])
    bq = torch.stack(list(bq_l))
    b_sf = torch.stack(list(b_sf_l))

    a_sf_t = dg.transform_sf_into_required_layout(
        a_sf, mn=G * M, k=K, recipe=(1, gran, gran), num_groups=None, is_sfa=True
    )
    b_sf_t = dg.transform_sf_into_required_layout(
        b_sf, mn=N, k=K, recipe=(1, gran, gran), num_groups=G, is_sfa=False
    )
    print("sfa layout:", a_sf_t.shape, a_sf_t.dtype, tuple(a_sf_t.stride()))
    print("sfb layout:", b_sf_t.shape, b_sf_t.dtype, tuple(b_sf_t.stride()))

    d = torch.empty(G * M, N, device=dev, dtype=torch.bfloat16)
    last_err = None
    for recipe in ((1, 1, gran), None):
        try:
            dg.m_grouped_fp8_gemm_nt_contiguous(
                (aq, a_sf_t), (bq, b_sf_t), d, m_indices, recipe=recipe
            )
            torch.cuda.synchronize()
            print(f"kernel accepted recipe={recipe}")
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"recipe={recipe} raised: {type(e).__name__}: {e}")
    if last_err is not None:
        print("PROBE_FAIL all recipe variants raised")
        return 1

    a_dq = aq.float() * a_sf.repeat_interleave(gran, dim=1)[:, :K]
    ref = torch.empty_like(d, dtype=torch.float32)
    for g in range(G):
        b_dq = bq[g].float() * b_sf[g].repeat_interleave(gran, 0).repeat_interleave(
            gran, 1
        )
        ref[g * M : (g + 1) * M] = a_dq[g * M : (g + 1) * M] @ b_dq.t()
    diff = calc_diff(d, ref.to(torch.bfloat16))
    ok = diff < 1e-3
    print(f"calc_diff={diff:.3e} -> {'PROBE_OK' if ok else 'PROBE_FAIL numerics'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
