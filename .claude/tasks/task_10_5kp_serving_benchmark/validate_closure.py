import glob
import json
import os
import re
import sys

O = sys.argv[1]
EXPECT_CC = [1, 4, 8, 16, 32, 64, 128]
BKS = ["cute_sm120_mxfp8_32", "deep_gemm_mxfp8_32"]


def expect_np(cc):
    return max(4 * cc, 16)


files = sorted(glob.glob(os.path.join(O, "bench_*_cc*_r1.json")))
seen = {}
fails = []
ok = 0
for f in files:
    b = os.path.basename(f)
    m = re.search(r"tp1_(\w+?)_cc(\d+)_r1\.json$", b)
    if not m:
        fails.append(f"UNPARSEABLE {b}")
        continue
    bk, cc = m.group(1), int(m.group(2))
    key = (bk, cc)
    if key in seen:
        fails.append(f"DUPLICATE {bk} cc={cc}")
        continue
    seen[key] = f
    d = json.load(open(f))
    np_ = d.get("num_prompts")
    comp = d.get("completed")
    tot = d.get("total_output_tokens")
    ot = d.get("output_throughput")
    exp = expect_np(cc)
    errs = []
    if np_ != exp:
        errs.append(f"num_prompts={np_}!={exp}")
    if comp != np_:
        errs.append(f"completed={comp}!=np={np_}")
    failed = (np_ or 0) - (comp or 0)
    if failed != 0:
        errs.append(f"failed={failed}")
    if tot is not None and tot != np_ * 1000:
        errs.append(f"total_output_tokens={tot}!={np_ * 1000}")
    if not (ot and ot > 0):
        errs.append(f"output_throughput={ot}")
    if errs:
        fails.append(f"{bk} cc={cc}: " + "; ".join(errs))
    else:
        ok += 1

# completeness: exactly 14 unique (backend,cc)
missing = []
for bk in BKS:
    for cc in EXPECT_CC:
        if (bk, cc) not in seen:
            missing.append(f"{bk} cc={cc}")

print(f"formal_json_count={len(seen)} (expect 14)")
print(f"gate_pass={ok} gate_fail={len(fails)}")
for m in missing:
    print(f"MISSING {m}")
for x in fails:
    print(f"FAIL {x}")
verdict = "PASS" if (len(seen) == 14 and not fails and not missing) else "FAIL"
print(f"CLOSURE_GATE={verdict}")
