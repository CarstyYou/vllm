#!/bin/bash
# usage: h20_dg.sh <gsm8k|mmlu|aime|mbpp> [args...]
# H20 baseline: explicit DG MoE with float32 scales (E8M0 has no arch gate in
# is_deep_gemm_e8m0_used and would silently UE8M0-requant on sm90).
export MOE_BACKEND=deep_gemm
export VLLM_USE_DEEP_GEMM_E8M0=0
EVAL=$1
shift
exec bash /home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/tests/run_${EVAL}_h20.sh "$@"
