"""task_10 parser, 3-round edition. Run inside fi-ci-cu130.

Usage: python parse_serving_3r.py <raw_dir> <out_csv>

Strict collection: fixed expectation backends x cc x rounds = 42 unique keys
with exact filenames (model/TP/backend/cc/round); unknown bench JSONs,
missing keys, duplicates and JSON load failures exit non-zero. Any raw round
that fails validation (INCOMPLETE) aborts before aggregation. The target CSV
is removed up front and only written on full success (tmp + atomic rename),
so a failed run can never leave a stale or partial CSV behind.

Output: one row per (cc, backend, round) for r1/r2/r3 plus one aggregate row
round=median per (cc, backend): every numeric metric is the per-metric median
across the 3 rounds; its status is derived from the 3 raw statuses (OK only
if all rounds OK). cute_vs_dg ratio is computed on median rows only
(median cute output_throughput / median dg output_throughput). Adds per-point
3-round spread columns and a per-cc crossover-stability verdict: per-round
leader is three-state win/tie/loss from CuTe's perspective (tie is never a
CuTe win); stable iff all 3 rounds agree.
"""
import glob
import json
import math
import os
import re
import statistics
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "benchmark.csv"
HW = "RTX_PRO_5000"
UUID = "GPU-ab3d387a"
FI = "db7abc0a"
DG = "a6b593d"
MODEL_TAG = "Qwen3.5-35B-A3B-FP8_tp1"
BKS = ["cute_sm120_mxfp8_32", "deep_gemm_mxfp8_32"]
BKMAP = {"cute_sm120_mxfp8_32": "cute", "deep_gemm_mxfp8_32": "dg"}
EXPECT_CC = [1, 4, 8, 16, 32, 64, 128]
ROUNDS = ["r1", "r2", "r3"]
WARMUP_RE = re.compile(
    r"^bench_%s_(cute_sm120_mxfp8_32|deep_gemm_mxfp8_32)"
    r"_cc(1|4|8|16|32|64|128)_(warmup|warmup2)\.json$" % re.escape(MODEL_TAG))

REQUIRED_FINITE = ["request_throughput", "output_throughput",
                   "total_token_throughput", "mean_ttft_ms", "median_ttft_ms",
                   "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
                   "p99_tpot_ms", "mean_itl_ms", "median_itl_ms",
                   "p99_itl_ms", "duration"]
POSITIVE = {"request_throughput", "output_throughput",
            "total_token_throughput"}
# strict-int schema: bool is excluded (failed=false / max_concurrency=true
# must not pass); all 7 counters must be present as real ints
INT_FIELDS = ["num_prompts", "completed", "failed", "max_concurrency",
              "total_input_tokens", "total_output_tokens",
              "max_concurrent_requests"]

METRICS = ["request_throughput", "output_throughput", "total_token_throughput",
           "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms",
           "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms",
           "p99_itl_ms", "mean_e2el_ms", "duration",
           "max_concurrent_requests"]

COLS = ["hardware", "gpu_uuid", "model", "tp", "isl", "osl", "cc", "backend",
        "round", "num_prompts", "completed", "failed", "req_throughput",
        "output_throughput", "total_token_throughput", "mean_ttft_ms",
        "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
        "p99_tpot_ms", "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
        "mean_e2el_ms", "duration_s", "max_concurrent_requests",
        "status", "tput_r_min", "tput_r_max", "tput_spread_pct",
        "cute_vs_dg_tput", "cute_vs_dg_tput_pct", "crossover_stable",
        "flashinfer_commit", "deepgemm_commit"]


def die(msg):
    print(f"FATAL {msg}")
    sys.exit(1)


def fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def expect_np(cc):
    return max(4 * cc, 16)


def is_finite_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def is_strict_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def row_status(cc, d):
    for k in INT_FIELDS:
        if not is_strict_int(d.get(k)):
            return "INCOMPLETE"
    np_ = d["num_prompts"]
    if np_ != expect_np(cc):
        return "INCOMPLETE"
    if d["completed"] != np_:
        return "INCOMPLETE"
    if d["failed"] != 0:
        return "INCOMPLETE"
    if d["max_concurrency"] != cc:
        return "INCOMPLETE"
    # mandatory token totals: random-input-len 8000 / output-len 1000 with
    # --ignore-eos are exact, so equality is required (missing field fails
    # the strict-int check above)
    if d["total_input_tokens"] != np_ * 8000:
        return "INCOMPLETE"
    if d["total_output_tokens"] != np_ * 1000:
        return "INCOMPLETE"
    # observed in-flight ceiling: r1 evidence shows it always lands in
    # (cc, np] (bench counts request overlap, e.g. cc=1 -> 2, cc=128 -> 132);
    # below cc means the point never reached its nominal concurrency
    if not (cc <= d["max_concurrent_requests"] <= np_):
        return "INCOMPLETE"
    for k in REQUIRED_FINITE:
        v = d.get(k)
        if not is_finite_num(v):
            return "INCOMPLETE"
        if k in POSITIVE and v <= 0:
            return "INCOMPLETE"
    return "OK"


# stale-output isolation: never let an old CSV survive a failed run
if os.path.exists(OUT):
    os.remove(OUT)
TMP = OUT + ".tmp"

# --- strict collection: exact filename -> unique key, 42/42 required ---
expected = {}
for bk in BKS:
    for cc in EXPECT_CC:
        for rnd in ROUNDS:
            expected[(bk, cc, rnd)] = f"bench_{MODEL_TAG}_{bk}_cc{cc}_{rnd}.json"
by_name = {v: k for k, v in expected.items()}
assert len(by_name) == 42, "expected filename collision"

errors = []
seen = {}
for f in sorted(glob.glob(os.path.join(RAW, "bench_*.json"))):
    b = os.path.basename(f)
    if WARMUP_RE.match(b):
        continue
    key = by_name.get(b)
    if key is None:
        errors.append(f"UNKNOWN_FILE {b}")
        continue
    if key in seen:
        errors.append(f"DUPLICATE_KEY {key}")
        continue
    seen[key] = f
for k in expected:
    if k not in seen:
        errors.append(f"MISSING {k[0]} cc={k[1]} {k[2]}")
if errors:
    for e in errors:
        print(e)
    die(f"collection failed: {len(errors)} error(s), refuse to aggregate")

data = {}  # (cc, short_bk, rnd) -> json dict
for (bk, cc, rnd), f in seen.items():
    try:
        data[(cc, BKMAP[bk], rnd)] = json.load(open(f))
    except Exception as ex:
        errors.append(f"JSON_LOAD_FAIL {os.path.basename(f)}: {ex}")
if errors:
    for e in errors:
        print(e)
    die("JSON load failed, refuse to aggregate")

# --- per-round validation gates aggregation ---
status = {}  # (cc, bk, rnd) -> OK/INCOMPLETE
for (cc, bk, rnd), d in data.items():
    st = row_status(cc, d)
    status[(cc, bk, rnd)] = st
    if st != "OK":
        errors.append(f"GATE_FAIL cc={cc} {bk} {rnd}: " + " ".join(
            f"{k}={d.get(k)!r}" for k in INT_FIELDS + ["output_throughput"]))
if errors:
    for e in errors:
        print(e)
    die("raw INCOMPLETE round(s) present, refuse to aggregate")

# --- aggregation (only reached with 42/42 all-OK) ---
tputs = {}   # (cc, bk) -> [r1, r2, r3] output_throughput
med = {}     # (cc, bk) -> median dict over METRICS
med_status = {}
for cc in EXPECT_CC:
    for bk in ("cute", "dg"):
        ds = [data[(cc, bk, r)] for r in ROUNDS]
        tputs[(cc, bk)] = [d["output_throughput"] for d in ds]
        md = {}
        for k in METRICS:
            vals = [d.get(k) for d in ds]
            if all(v is not None for v in vals):
                md[k] = statistics.median(vals)
        md["num_prompts"] = ds[0].get("num_prompts")
        md["completed"] = statistics.median([d.get("completed") for d in ds])
        med[(cc, bk)] = md
        sts = [status[(cc, bk, r)] for r in ROUNDS]
        med_status[(cc, bk)] = "OK" if all(s == "OK" for s in sts) \
            else "INCOMPLETE"

# crossover stability per cc: three-state win/tie/loss (CuTe perspective)
# per round; a tie is NOT a CuTe win. Stable iff all 3 rounds agree.
def leader_sign(c, d):
    if c > d:
        return "win"
    if c < d:
        return "loss"
    return "tie"


signs = {}
stable = {}
for cc in EXPECT_CC:
    ss = [leader_sign(tputs[(cc, "cute")][i], tputs[(cc, "dg")][i])
          for i in range(3)]
    signs[cc] = ss
    stable[cc] = "yes" if len(set(ss)) == 1 else "NO"

with open(TMP, "w") as f:
    f.write(",".join(COLS) + "\n")

    def emit(cc, bk, rnd, d, st, failed, extra):
        vals = [HW, UUID, "Qwen3.5-35B-A3B-FP8", "1", "8000", "1000", str(cc),
                bk, rnd, fmt(d.get("num_prompts")), fmt(d.get("completed")),
                fmt(failed), fmt(d.get("request_throughput")),
                fmt(d.get("output_throughput")),
                fmt(d.get("total_token_throughput")), fmt(d.get("mean_ttft_ms")),
                fmt(d.get("median_ttft_ms")), fmt(d.get("p99_ttft_ms")),
                fmt(d.get("mean_tpot_ms")), fmt(d.get("median_tpot_ms")),
                fmt(d.get("p99_tpot_ms")), fmt(d.get("mean_itl_ms")),
                fmt(d.get("median_itl_ms")), fmt(d.get("p99_itl_ms")),
                fmt(d.get("mean_e2el_ms")), fmt(d.get("duration")),
                fmt(d.get("max_concurrent_requests")), st] + extra + [FI, DG]
        f.write(",".join(vals) + "\n")

    for cc in EXPECT_CC:
        for bk in ("cute", "dg"):
            for rnd in ROUNDS:
                d = data[(cc, bk, rnd)]
                emit(cc, bk, rnd, d, status[(cc, bk, rnd)],
                     d["num_prompts"] - d["completed"],
                     ["", "", "", "", "", ""])

    for cc in EXPECT_CC:
        for bk in ("cute", "dg"):
            ts = tputs[(cc, bk)]
            m = med[(cc, bk)]
            spread = (max(ts) - min(ts)) / statistics.median(ts) * 100
            if bk == "cute":
                sp = statistics.median(tputs[(cc, "cute")]) / \
                    statistics.median(tputs[(cc, "dg")])
                sps, spps = f"{sp:.3f}", f"{(sp - 1) * 100:+.1f}"
            else:
                sps, spps = "", ""
            emit(cc, bk, "median", m, med_status[(cc, bk)], 0,
                 [f"{min(ts):.2f}", f"{max(ts):.2f}", f"{spread:.1f}",
                  sps, spps, stable[cc]])

os.replace(TMP, OUT)
print(f"rows={len(data)}+{len(EXPECT_CC) * 2} median -> {OUT}")
print("cc, cute_med, dg_med, ratio, diff%, med_leader, spread%(cute/dg), "
      "stable, per-round sign+ratio")
for cc in EXPECT_CC:
    cm = statistics.median(tputs[(cc, "cute")])
    dm = statistics.median(tputs[(cc, "dg")])
    csp = (max(tputs[(cc, "cute")]) - min(tputs[(cc, "cute")])) / cm * 100
    dsp = (max(tputs[(cc, "dg")]) - min(tputs[(cc, "dg")])) / dm * 100
    rr = [tputs[(cc, "cute")][i] / tputs[(cc, "dg")][i] for i in range(3)]
    med_leader = leader_sign(cm, dm)
    print(f"cc={cc}: {cm:.2f} {dm:.2f} {cm / dm:.3f} {(cm / dm - 1) * 100:+.1f}% "
          f"med_leader={med_leader} {csp:.1f}%/{dsp:.1f}% stable={stable[cc]} "
          f"rounds={[f'{s}:{r:.3f}' for s, r in zip(signs[cc], rr)]}")
