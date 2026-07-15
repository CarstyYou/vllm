#!/bin/bash
# usage: h20_mc.sh <evalset|lane> <e8m0:0|1> <args...>
# H20 multi-GPU baseline wrapper: DG backend, float (E8M0=0) or UE8M0 (E8M0=1)
# scale mode, sm90 GDN prefill forced to triton (FI sm90 GDN broken in this env).
MODE=$1
E8=$2
shift 2
export VLLM_USE_DEEP_GEMM_E8M0=$E8
export EXTRA_SERVE_ARGS="--gdn-prefill-backend triton"
if [ "$E8" = "1" ]; then export TAG_SUFFIX="_h20ue8m0"; else export TAG_SUFFIX="_h20float"; fi
T=/home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/tests
if [ "$MODE" = "evalset" ]; then
  exec bash $T/run_evalset_mc_v2.sh "$@"
else
  exec bash $T/run_dsv4_lane_v2.sh "$@"
fi
