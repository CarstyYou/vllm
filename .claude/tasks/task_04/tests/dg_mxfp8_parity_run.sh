#!/bin/bash
set -uo pipefail
VLLM_ROOT=/home/scratch.xiy_gpu/mega_inference/vllm
export CUDA_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3
export PATH=$VLLM_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$VLLM_ROOT
cd $VLLM_ROOT
mkdir -p .claude/tasks/task_04/results
.venv/bin/python .claude/tasks/task_04/tests/test_dg_mxfp8_parity.py \
  .claude/tasks/task_04/results/dg_mxfp8_parity.csv
RC=$?
echo DG_MXFP8_PARITY_RC=$RC
exit $RC
