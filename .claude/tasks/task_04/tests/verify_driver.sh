#!/bin/bash
# Runs the 4 anomaly-verification runs sequentially (task_04 sub-task 8).
T=/home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/tests
V=/home/scratch.xiy_gpu/mega_inference/vllm/.claude/tasks/task_04/results/verify
mkdir -p $V
{
echo "== V4: 35B dg_mxfp8_32 gsm8k repro (TP1)"
bash $T/verify_plain.sh gsm8k Qwen3.5-35B-A3B-FP8 1 deep_gemm_mxfp8_32 1319
echo "V4_RC=$?"
echo "== V1: 397B mxfp8_32 aime repro + log_samples"
bash $T/verify_ls.sh aime Qwen3.5-397B-A17B-FP8 8 cute_sm120_mxfp8_32
echo "V1_RC=$?"
echo "== V2: 397B triton mbpp + log_samples"
bash $T/verify_ls.sh mbpp Qwen3.5-397B-A17B-FP8 8 triton
echo "V2_RC=$?"
echo "== V3: 397B triton gsm8k stability rerun"
bash $T/verify_plain.sh gsm8k Qwen3.5-397B-A17B-FP8 8 triton 1319
echo "V3_RC=$?"
} >> $V/driver.log 2>&1
touch $V/driver.done
