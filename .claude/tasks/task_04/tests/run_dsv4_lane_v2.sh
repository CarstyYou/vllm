#!/bin/bash
# usage: run_dsv4_lane.sh <lane:0|1> <backend> [backend...]
# Lane-pinned DSv4-Flash TP4 evalset runner: lane 0 -> GPUs 0-3 / port 18400,
# lane 1 -> GPUs 4-7 / port 19400. Backends run sequentially within the lane.
set -uo pipefail
LANE=$1
shift
if [ "$LANE" = "0" ]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  PORT=18400
else
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  PORT=19400
fi
T=/home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/tests
RC=0
for B in "$@"; do
  echo "=== lane $LANE start backend=$B ==="
  bash $T/run_evalset_mc_v2.sh DeepSeek-V4-Flash-Base 4 "$B" $PORT || RC=1
  echo "=== lane $LANE done backend=$B rc=$RC ==="
done
exit $RC
