#!/bin/bash
set -uo pipefail
VLLM_ROOT=/home/scratch.xiy_gpu/mega_inference/vllm
export CUDA_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/home/scratch.xiy_gpu
export XDG_CACHE_HOME=/home/scratch.xiy_gpu/.cache
export VLLM_CACHE_ROOT=/home/scratch.xiy_gpu/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/home/scratch.xiy_gpu/.cache/inductor
export TRITON_CACHE_DIR=/home/scratch.xiy_gpu/.cache/triton
export HF_HOME=/home/scratch.xiy_gpu/.cache/hf
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=$VLLM_ROOT
mkdir -p $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $HF_HOME
cd $VLLM_ROOT
source .venv/bin/activate
T=$VLLM_ROOT/.claude/tasks/task_07/tests
echo "===== CUTE ====="
python $T/moe_microbench.py cute 256 8 2048 512 5 2>&1 | grep -iE "MoE_MNK|MICROBENCH_DONE|Error|Traceback|assert" | head -8
echo "===== DG ====="
DG_PRINT_CONFIGS=1 python $T/moe_microbench.py dg 256 8 2048 512 5 2>&1 | grep -iE "MoE_MNK|MICROBENCH_DONE|block_m|block_n|num_stages|GemmConfig|Error|Traceback|assert" | head -12
echo "SMOKE_DONE"
