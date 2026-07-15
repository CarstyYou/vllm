#!/bin/bash
# usage: run_evalset_mc.sh <model_dir> <tp> <backend|auto> [port]
# One serve (ctx 4096) for GSM8K+MMLU+MBPP, a second serve (ctx 36864) for AIME —
# avoids reloading multi-hundred-GB weights once per eval.
set -uo pipefail
MODEL_DIR=$1
TP=$2
BACKEND=$3
PORT=${4:-18400}
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

start_serve() {  # $1=max_model_len $2=serve_log
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name evalmodel \
    --max-model-len $1 --enable-prefix-caching \
    --tensor-parallel-size $TP $MOE_ARG $KV_ARG \
    --port $PORT --disable-uvicorn-access-log > "$2" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 480); do
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then return 0; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER_DIED"; tail -30 "$2"; return 1; fi
    sleep 5
  done
  echo "SERVER_TIMEOUT"; tail -30 "$2"; return 1
}

stop_serve() {
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
  sleep 10
}
SERVER_PID=""
trap 'kill ${SERVER_PID:-} 2>/dev/null || true' EXIT

RC=0

SERVE_LOG=$OUT_DIR/serve_main_${MTAG}.log
if start_serve 4096 "$SERVE_LOG"; then
  echo "server up ($MTAG ctx=4096)"
  grep -E "Using .* Fp8 MoE backend|Auto-disabled|E8M0 enabled|PDL enabled|non-default args" "$SERVE_LOG" > $OUT_DIR/evidence_${MTAG}_evalset.txt || true

  (cd $VLLM_ROOT/tests/evals/gsm8k && python gsm8k_eval.py --host http://localhost --port $PORT \
    --num-questions 1319 --num-shots 5 \
    --save-results $OUT_DIR/gsm8k_${MTAG}_n1319.json \
    2>&1 | tee $OUT_DIR/gsm8k_${MTAG}_n1319.log) || RC=1
  echo "GSM8K_DONE $MTAG rc=$RC"

  lm_eval --model local-completions \
    --model_args "base_url=http://localhost:$PORT/v1/completions,model=evalmodel,tokenizer=$MODEL,num_concurrent=128,max_retries=3" \
    --tasks mmlu --num_fewshot 5 --seed 42 \
    --output_path $OUT_DIR/mmlu_${MTAG} \
    2>&1 | tee $OUT_DIR/mmlu_${MTAG}.log | tail -20 || RC=1
  echo "MMLU_DONE $MTAG rc=$RC"

  lm_eval --model local-completions \
    --model_args "base_url=http://localhost:$PORT/v1/completions,model=evalmodel,tokenizer=$MODEL,num_concurrent=128,max_retries=3" \
    --tasks mbpp --seed 42 --confirm_run_unsafe_code \
    --output_path $OUT_DIR/mbpp_${MTAG} \
    2>&1 | tee $OUT_DIR/mbpp_${MTAG}.log | tail -20 || RC=1
  echo "MBPP_DONE $MTAG rc=$RC"
  stop_serve
else
  RC=1
fi

SERVE_LOG=$OUT_DIR/serve_aime_${MTAG}.log
if start_serve 36864 "$SERVE_LOG"; then
  echo "server up ($MTAG ctx=36864)"
  lm_eval --model local-completions \
    --model_args "base_url=http://localhost:$PORT/v1/completions,model=evalmodel,tokenizer=$MODEL,num_concurrent=32,max_retries=3" \
    --tasks aime24,aime25 --seed 42 \
    --output_path $OUT_DIR/aime_${MTAG} \
    2>&1 | tee $OUT_DIR/aime_${MTAG}.log | tail -20 || RC=1
  echo "AIME_DONE $MTAG rc=$RC"
  stop_serve
else
  RC=1
fi

echo "EVALSET_DONE $MTAG RC=$RC"
exit $RC
