# SPDX-License-Identifier: Apache-2.0
"""Aggregate exp_05 bench json -> per-(model,backend,cc) median metrics CSV.

Reads results/bench_<mtag>_cc<cc>_{r1,r2,r3}.json, takes the median across the
3 formal rounds for each metric, and emits a tidy CSV. Verifies output len ==
1024 per cell (--ignore-eos apples-to-apple guard); flags any deviation.
"""

import csv
import glob
import json
import os
import re
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "results"
)

# metric key in vllm bench serve json -> output column
METRICS = {
    "output_throughput": "out_tok_s",
    "mean_ttft_ms": "ttft_mean_ms",
    "p99_ttft_ms": "ttft_p99_ms",
    "mean_tpot_ms": "tpot_mean_ms",
    "p99_tpot_ms": "tpot_p99_ms",
    "request_throughput": "req_s",
}
PAT = re.compile(r"bench_(?P<mtag>.+)_cc(?P<cc>\d+)_(?P<round>r[123])\.json$")


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main():
    cells = {}  # (mtag, cc) -> {metric: [vals]}, olen list
    for f in glob.glob(os.path.join(RESULTS, "bench_*_r[123].json")):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        key = (m["mtag"], int(m["cc"]))
        c = cells.setdefault(key, {k: [] for k in METRICS})
        c.setdefault("_olen", [])
        for mk in METRICS:
            if mk in d and d[mk] is not None:
                c[mk].append(d[mk])
        comp = max(d.get("completed", 1), 1)
        c["_olen"].append(d.get("total_output_tokens", 0) / comp)

    rows, warn = [], []
    for (mtag, cc), c in sorted(cells.items()):
        row = {"mtag": mtag, "cc": cc, "rounds": len(c["output_throughput"])}
        for mk, col in METRICS.items():
            row[col] = round(median(c[mk]), 2) if c[mk] else ""
        olen = median(c["_olen"]) if c["_olen"] else 0
        row["olen"] = round(olen)
        if abs(olen - 1024) > 2:
            warn.append(f"{mtag} cc{cc}: olen={olen:.0f} != 1024")
        rows.append(row)

    out = os.path.join(RESULTS, "exp05_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cells -> {out}")
    if warn:
        print("OLEN WARNINGS:")
        for x in warn:
            print(" ", x)
    else:
        print("olen OK (all == 1024)")


if __name__ == "__main__":
    main()
