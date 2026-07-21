"""task_10 closure gate, 3-round edition: 42 formal JSONs (r1/r2/r3 x 14).

Strict: fixed expectation backends x cc x rounds = 42 unique keys; filenames
must exactly match the unique model/TP/backend/cc/round pattern; unknown
bench JSONs, missing keys, duplicates and JSON load failures all exit
non-zero.

Run inside fi-ci-cu130. Usage: python validate_closure_3r.py <out_dir>
"""
import glob
import json
import math
import os
import re
import sys

O = sys.argv[1]
MODEL_TAG = "Qwen3.5-35B-A3B-FP8_tp1"
EXPECT_CC = [1, 4, 8, 16, 32, 64, 128]
BKS = ["cute_sm120_mxfp8_32", "deep_gemm_mxfp8_32"]
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


def expect_np(cc):
    return max(4 * cc, 16)


def is_finite_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def is_strict_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


expected = {}  # (bk, cc, rnd) -> exact filename
for bk in BKS:
    for cc in EXPECT_CC:
        for rnd in ROUNDS:
            expected[(bk, cc, rnd)] = f"bench_{MODEL_TAG}_{bk}_cc{cc}_{rnd}.json"
by_name = {v: k for k, v in expected.items()}
assert len(by_name) == 42, "expected filename collision"

fails = []
seen = {}
for f in sorted(glob.glob(os.path.join(O, "bench_*.json"))):
    b = os.path.basename(f)
    if WARMUP_RE.match(b):
        continue
    key = by_name.get(b)
    if key is None:
        fails.append(f"UNKNOWN_FILE {b}")
        continue
    if key in seen:
        fails.append(f"DUPLICATE_KEY {key}")
        continue
    seen[key] = f

missing = [k for k in expected if k not in seen]

ok = 0
for (bk, cc, rnd), f in sorted(seen.items()):
    try:
        d = json.load(open(f))
    except Exception as ex:
        fails.append(f"JSON_LOAD_FAIL {os.path.basename(f)}: {ex}")
        continue
    np_ = d.get("num_prompts")
    comp = d.get("completed")
    exp = expect_np(cc)
    errs = []
    for k in INT_FIELDS:
        if not is_strict_int(d.get(k)):
            errs.append(f"{k}={d.get(k)!r} not a strict int")
    if np_ != exp:
        errs.append(f"num_prompts={np_}!={exp}")
    if comp != np_:
        errs.append(f"completed={comp}!=np={np_}")
    if d.get("failed") != 0:
        errs.append(f"failed={d.get('failed')}!=0")
    if (np_ or 0) - (comp or 0) != 0:
        errs.append(f"derived_failed={(np_ or 0) - (comp or 0)}")
    if d.get("max_concurrency") != cc:
        errs.append(f"max_concurrency={d.get('max_concurrency')}!=cc={cc}")
    # mandatory token totals (exact under --ignore-eos + fixed random lens)
    if d.get("total_input_tokens") != exp * 8000:
        errs.append(f"total_input_tokens={d.get('total_input_tokens')}"
                    f"!={exp * 8000}")
    if d.get("total_output_tokens") != exp * 1000:
        errs.append(f"total_output_tokens={d.get('total_output_tokens')}"
                    f"!={exp * 1000}")
    # observed in-flight ceiling: r1 evidence puts it in (cc, np] (bench
    # counts request overlap); below cc = never reached nominal concurrency
    mcr = d.get("max_concurrent_requests")
    if not (is_strict_int(mcr) and cc <= mcr <= exp):
        errs.append(f"max_concurrent_requests={mcr!r} not int in "
                    f"[cc={cc}, np={exp}]")
    for k in REQUIRED_FINITE:
        v = d.get(k)
        if not is_finite_num(v):
            errs.append(f"{k}={v} not finite number")
        elif k in POSITIVE and v <= 0:
            errs.append(f"{k}={v} not >0")
    if errs:
        fails.append(f"{bk} cc={cc} {rnd}: " + "; ".join(errs))
    else:
        ok += 1

print(f"formal_json_count={len(seen)} (expect 42)")
print(f"gate_pass={ok} gate_fail={len(fails)} missing={len(missing)}")
for bk, cc, rnd in missing:
    print(f"MISSING {bk} cc={cc} {rnd}")
for x in fails:
    print(f"FAIL {x}")
verdict = "PASS" if (len(seen) == 42 and ok == 42 and not fails
                     and not missing) else "FAIL"
print(f"CLOSURE_GATE={verdict}")
sys.exit(0 if verdict == "PASS" else 1)
