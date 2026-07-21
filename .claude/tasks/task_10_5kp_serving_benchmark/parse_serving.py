import glob
import json
import os
import re
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "benchmark.csv"
HW = "RTX_PRO_5000"
UUID = "GPU-ab3d387a"
FI = "db7abc0a"
DG = "a6b593d"
BKMAP = {"cute_sm120_mxfp8_32": "cute", "deep_gemm_mxfp8_32": "dg"}

COLS = ["hardware", "gpu_uuid", "model", "tp", "isl", "osl", "cc", "backend",
        "round", "num_prompts", "completed", "failed", "req_throughput",
        "output_throughput", "total_token_throughput", "mean_ttft_ms",
        "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
        "p99_tpot_ms", "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
        "mean_e2el_ms", "duration_s", "max_concurrent_requests",
        "status", "cute_vs_dg_tput", "cute_vs_dg_tput_pct",
        "flashinfer_commit", "deepgemm_commit"]


def g(d, k):
    v = d.get(k)
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


rows = []
for fn in sorted(glob.glob(os.path.join(RAW, "bench_*_cc*_*.json"))):
    base = os.path.basename(fn)
    m = re.search(r"_tp1_(\w+?)_cc(\d+)_(\w+)\.json$", base)
    if not m:
        continue
    bk_full, cc, rnd = m.group(1), int(m.group(2)), m.group(3)
    if rnd == "warmup":
        continue
    bk = BKMAP.get(bk_full, bk_full)
    try:
        d = json.load(open(fn))
    except Exception as ex:
        print(f"SKIP {base}: {ex}")
        continue
    rows.append({
        "cc": cc, "backend": bk, "round": rnd,
        "output_throughput": d.get("output_throughput"),
        "d": d,
    })

# speedup keyed by cc: cute_tput / dg_tput
tput = {}
for r in rows:
    tput[(r["cc"], r["backend"])] = r["output_throughput"]

with open(OUT, "w") as f:
    f.write(",".join(COLS) + "\n")
    for r in sorted(rows, key=lambda x: (x["cc"], x["backend"])):
        d = r["d"]
        cc, bk = r["cc"], r["backend"]
        ct, dt = tput.get((cc, "cute")), tput.get((cc, "dg"))
        if ct and dt and bk == "cute":
            sp = ct / dt
            sps, spps = f"{sp:.3f}", f"{(sp-1)*100:+.1f}"
        else:
            sps, spps = "", ""
        np_, comp = d.get("num_prompts"), d.get("completed")
        failed = d.get("failed")
        if failed is None and np_ is not None and comp is not None:
            failed = np_ - comp
        status = "OK" if (np_ is not None and comp == np_ and (failed in (0, None) or failed == 0)) else "INCOMPLETE"
        vals = [HW, UUID, "Qwen3.5-35B-A3B-FP8", "1", "8000", "1000", str(cc), bk,
                r["round"], g(d, "num_prompts"), g(d, "completed"),
                ("" if failed is None else str(failed)),
                g(d, "request_throughput"), g(d, "output_throughput"),
                g(d, "total_token_throughput"), g(d, "mean_ttft_ms"),
                g(d, "median_ttft_ms"), g(d, "p99_ttft_ms"), g(d, "mean_tpot_ms"),
                g(d, "median_tpot_ms"), g(d, "p99_tpot_ms"), g(d, "mean_itl_ms"),
                g(d, "median_itl_ms"), g(d, "p99_itl_ms"), g(d, "mean_e2el_ms"),
                g(d, "duration"), g(d, "max_concurrent_requests"), status,
                sps, spps, FI, DG]
        f.write(",".join(vals) + "\n")
        if status != "OK":
            print(f"GATE_FAIL cc={cc} {bk} {r['round']}: completed={comp} num_prompts={np_} failed={failed}")

print(f"rows={len(rows)} -> {OUT}")
