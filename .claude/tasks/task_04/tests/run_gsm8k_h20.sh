#!/bin/bash
# usage: run_gsm8k.sh <backend:triton|deep_gemm> <num_questions> [port]
set -euo pipefail
BACKEND=h20_${MOE_BACKEND:-native}
if [ "${MOE_BACKEND:-}" ]; then MOE_ARG="--moe-backend=$MOE_BACKEND"; else MOE_ARG=""; fi
NQ=${1:-1319}
PORT=${2:-18000}
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

SERVE_LOG=$OUT_DIR/serve_${BACKEND}.log
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen35 \
  --max-model-len 4096 --enable-prefix-caching --gdn-prefill-backend triton $MOE_ARG \
  --port $PORT --disable-uvicorn-access-log > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

for i in $(seq 1 240); do
  if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER_DIED"; tail -30 "$SERVE_LOG"; exit 1; fi
  sleep 5
done
curl -sf http://localhost:$PORT/health > /dev/null || { echo "SERVER_TIMEOUT"; tail -30 "$SERVE_LOG"; exit 1; }
echo "server up (backend=$BACKEND)"
grep -E "Using .* Fp8 MoE backend|Auto-disabled|E8M0 enabled|PDL enabled|non-default args" "$SERVE_LOG" > $OUT_DIR/evidence_${BACKEND}_$(basename $0 .sh).txt || true

grep -iE "moe.*backend|deepgemm|e8m0" "$SERVE_LOG" | head -10 || true

if [ "$NQ" -lt 1319 ]; then TAG="smoke_n${NQ}"; else TAG="n${NQ}"; fi
cd $VLLM_ROOT/tests/evals/gsm8k
python gsm8k_eval.py --host http://localhost --port $PORT \
  --num-questions $NQ --num-shots 5 \
  --save-results $OUT_DIR/gsm8k_${BACKEND}_${TAG}.json \
  2>&1 | tee $OUT_DIR/gsm8k_${BACKEND}_${TAG}.log
echo "GSM8K_DONE backend=$BACKEND n=$NQ"
