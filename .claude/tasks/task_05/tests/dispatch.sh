#!/bin/bash
# usage: dispatch.sh <alloc-id> <model_dir> <tp> <gdn:fi|triton> <col1> [col2 ...]
# Submits one bench_serve.sh task per column onto the given allocation (serial via ts).
# Each col = "<backend>:<tag_suffix>:<e8m0>" e.g. cute_sm120_fp8:cutefp8:  deep_gemm:dgfloat:0
T=/home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_05/tests
ALLOC=$1; MODEL=$2; TP=$3; GDN=$4; shift 4
for spec in "$@"; do
  BK=${spec%%:*}; rest=${spec#*:}; SUF=${rest%%:*}; E8=${rest#*:}
  ssh-gw task submit "$ALLOC" --label "b-${MODEL:0:6}-${SUF}" --timeout 6h -- \
    bash $T/bench_serve.sh "$MODEL" "$TP" "$BK" "$SUF" "$GDN" "$E8" 2>&1 | head -1
done
