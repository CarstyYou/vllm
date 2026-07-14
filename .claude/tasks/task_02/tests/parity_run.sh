#!/bin/bash
set -uo pipefail
VLLM_ROOT=/home/scratch.xiy_gpu/mega_inference/vllm
export CUDA_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3
export PATH=$VLLM_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/home/scratch.xiy_gpu
export FLASHINFER_JIT_DEBUG=0
export PYTHONPATH=$VLLM_ROOT
cd $VLLM_ROOT
mkdir -p .claude/tasks/task_02/results
.venv/bin/python .claude/tasks/task_02/tests/test_cute_fp8_vs_triton.py \
  .claude/tasks/task_02/results/cute_fp8_vs_triton.csv
RC=$?
echo PARITY_RC=$RC
exit $RC
