#!/bin/bash
# usage: verify_plain.sh <gsm8k|aime|mbpp> <model_dir> <tp> <backend> [args...]
export OUT_DIR=/home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/results/verify
mkdir -p $OUT_DIR
EVAL=$1
shift
exec bash /home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/tests/run_${EVAL}_mc.sh "$@"
