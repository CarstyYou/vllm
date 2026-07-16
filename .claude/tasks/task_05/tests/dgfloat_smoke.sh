#!/bin/bash
export VLLM_USE_DEEP_GEMM_E8M0=0
export OUT_DIR=/tmp/task05_smoke
mkdir -p $OUT_DIR
exec bash /home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/tests/run_gsm8k_v2.sh deep_gemm 20
