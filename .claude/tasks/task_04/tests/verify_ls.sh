#!/bin/bash
# usage: verify_ls.sh <aime|mbpp> <model_dir> <tp> <backend> [port]
export EXTRA_LM_ARGS="--log_samples"
export OUT_DIR=/home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/results/verify
mkdir -p $OUT_DIR
EVAL=$1
shift
exec bash /home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/tests/run_${EVAL}_mc.sh "$@"
