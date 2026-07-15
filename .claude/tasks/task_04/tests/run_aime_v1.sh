#!/bin/bash
# usage: run_aime_v2.sh <backend> [port]
# AIME-only serve config (Q-A lock): --max-model-len 36864 to fit max_gen_toks=32768.
set -euo pipefail
MOEB=$1
BACKEND=${1}_v1cfg
PORT=${2:-18200}
VLLM_ROOT=/home/scratch.xiy_gpu/mega_inference/vllm
MODEL=/home/scratch.trt_llm_data/llm-models/Qwen3.5-35B-A3B-FP8
OUT_DIR=${OUT_DIR:-$VLLM_ROOT/.claude/tasks/task_04/results}
mkdir -p "$OUT_DIR"

export CUDA_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/home/scratch.xiy_gpu
export PATH=$CUDA_HOME/bin:$PATH
source $VLLM_ROOT/.venv/bin/activate
cd $VLLM_ROOT

# drain guard: wait for prior task's server to release GPU memory
for i in $(seq 1 60); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$FREE" -gt 80000 ] && break
  sleep 5
done

SERVE_LOG=$OUT_DIR/serve_aime_${BACKEND}.log
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen35 \
  --max-model-len 36864 --moe-backend=$MOEB \
  --port $PORT --disable-uvicorn-access-log > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

for i in $(seq 1 240); do
  if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER_DIED"; tail -30 "$SERVE_LOG"; exit 1; fi
  sleep 5
done
curl -sf http://localhost:$PORT/health > /dev/null || { echo "SERVER_TIMEOUT"; exit 1; }
echo "server up (backend=$BACKEND)"
grep -E "Using .* Fp8 MoE backend|Auto-disabled|E8M0 enabled|PDL enabled|non-default args" "$SERVE_LOG" > $OUT_DIR/evidence_${BACKEND}_$(basename $0 .sh).txt || true

lm_eval --model local-completions \
  --model_args "base_url=http://localhost:$PORT/v1/completions,model=qwen35,tokenizer=$MODEL,num_concurrent=32,max_retries=3,max_length=36864,timeout=7200" \
  --tasks aime24,aime25 --seed 42 \
  --output_path $OUT_DIR/aime_${BACKEND} \
  2>&1 | tee $OUT_DIR/aime_${BACKEND}.log | tail -40
echo "AIME_DONE backend=$BACKEND"
