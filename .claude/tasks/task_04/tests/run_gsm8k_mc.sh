#!/bin/bash
# usage: run_gsm8k_mc.sh <model_dir> <tp> <backend|auto> <num_questions> [port]
set -euo pipefail
MODEL_DIR=$1
TP=$2
BACKEND=$3
NQ=$4
PORT=${5:-18000}
VLLM_ROOT=/home/scratch.xiy_gpu/mega_inference/vllm
MODEL=/home/scratch.trt_llm_data/llm-models/$MODEL_DIR
MTAG=${MODEL_DIR}_tp${TP}_${BACKEND}
OUT_DIR=${OUT_DIR:-$VLLM_ROOT/.claude/tasks/task_04/results}
mkdir -p "$OUT_DIR"

export CUDA_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/home/scratch.xiy_gpu
export PATH=$CUDA_HOME/bin:$PATH
source $VLLM_ROOT/.venv/bin/activate
cd $VLLM_ROOT
if [ "$BACKEND" = "auto" ]; then MOE_ARG=""; else MOE_ARG="--moe-backend=$BACKEND"; fi
case "$MODEL_DIR" in DeepSeek-V4*) KV_ARG="--kv-cache-dtype fp8_ds_mla";; *) KV_ARG="";; esac

SERVE_LOG=$OUT_DIR/serve_gsm8k_${MTAG}.log
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name evalmodel \
  --max-model-len 4096 --enable-prefix-caching \
  --tensor-parallel-size $TP $MOE_ARG $KV_ARG \
  --port $PORT --disable-uvicorn-access-log > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

for i in $(seq 1 360); do
  if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER_DIED"; tail -30 "$SERVE_LOG"; exit 1; fi
  sleep 5
done
curl -sf http://localhost:$PORT/health > /dev/null || { echo "SERVER_TIMEOUT"; tail -30 "$SERVE_LOG"; exit 1; }
echo "server up ($MTAG)"
grep -E "Using .* Fp8 MoE backend|Auto-disabled|E8M0 enabled|PDL enabled|non-default args" "$SERVE_LOG" > $OUT_DIR/evidence_${MTAG}_$(basename $0 .sh).txt || true

if [ "$NQ" -lt 1319 ]; then QTAG="smoke_n${NQ}"; else QTAG="n${NQ}"; fi
cd $VLLM_ROOT/tests/evals/gsm8k
python gsm8k_eval.py --host http://localhost --port $PORT \
  --num-questions $NQ --num-shots 5 \
  --save-results $OUT_DIR/gsm8k_${MTAG}_${QTAG}.json \
  2>&1 | tee $OUT_DIR/gsm8k_${MTAG}_${QTAG}.log
echo "GSM8K_DONE $MTAG n=$NQ"
