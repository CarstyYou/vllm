#!/bin/bash
# usage: smoke_mc.sh <model_dir> <tp> <backend|auto> [port]
# 20-question GSM8K smoke; outputs to /tmp (engineering validation only, not results).
export OUT_DIR=/tmp/task04_smoke
mkdir -p $OUT_DIR
exec bash /home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/tests/run_gsm8k_mc.sh "$1" "$2" "$3" 20 "${4:-18400}"
