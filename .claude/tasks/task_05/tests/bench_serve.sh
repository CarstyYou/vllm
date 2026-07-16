#!/bin/bash
# usage: bench_serve.sh <model_dir> <tp> <backend|auto> <tag_suffix> [gdn:fi|triton] [e8m0:0|1]
# One serve instance; sweeps cc=1..128 (8 points), 1 warmup + 3 formal rounds each.
# ISL/OSL=8192/1024, random dataset seed=42, --ignore-eos, prefix caching OFF.
set -uo pipefail
MODEL_DIR=$1
TP=$2
BACKEND=$3
SUF=$4
GDN=${5:-fi}
E8=${6:-}
PORT=${PORT:-18500}
VLLM_ROOT=/home/scratch.xiy_gpu/mega_inference/vllm
MODEL=/home/scratch.trt_llm_data/llm-models/$MODEL_DIR
MTAG=${MODEL_DIR}_tp${TP}_${SUF}
OUT=${OUT_DIR:-$VLLM_ROOT/.claude/tasks/task_05/results}
mkdir -p $OUT

export CUDA_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/home/scratch.xiy_gpu
export PATH=$CUDA_HOME/bin:$PATH
source $VLLM_ROOT/.venv/bin/activate
cd $VLLM_ROOT
[ -n "$E8" ] && export VLLM_USE_DEEP_GEMM_E8M0=$E8
if [ "$BACKEND" = "auto" ]; then MOE_ARG=""; else MOE_ARG="--moe-backend=$BACKEND"; fi
case "$MODEL_DIR" in DeepSeek-V4*) KV_ARG="--kv-cache-dtype fp8_ds_mla";; *) KV_ARG="";; esac
if [ "$GDN" = "triton" ]; then GDN_ARG="--gdn-prefill-backend triton"; else GDN_ARG=""; fi

for i in $(seq 1 60); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$FREE" -gt 80000 ] && break
  sleep 5
done

SERVE_LOG=$OUT/serve_${MTAG}.log
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name benchmodel \
  --max-model-len 10240 --no-enable-prefix-caching \
  --tensor-parallel-size $TP $MOE_ARG $KV_ARG $GDN_ARG \
  --port $PORT --disable-uvicorn-access-log > "$SERVE_LOG" 2>&1 &
SERVER_PID=""
SERVER_PID=$!
trap 'kill ${SERVER_PID:-} 2>/dev/null || true' EXIT

for i in $(seq 1 480); do
  curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "SERVER_DIED"; tail -20 "$SERVE_LOG"; exit 1; }
  sleep 5
done
curl -sf http://localhost:$PORT/health >/dev/null || { echo "SERVER_TIMEOUT"; exit 1; }
echo "server up ($MTAG)"
grep -E "Using .* Fp8 MoE backend|E8M0|GDN prefill|KV cache size|non-default args" "$SERVE_LOG" > $OUT/evidence_${MTAG}.txt || true

RC=0
for CC in 1 2 4 8 16 32 64 128; do
  NP=$(( CC * 4 > 16 ? CC * 4 : 16 ))
  for ROUND in warmup r1 r2 r3; do
    [ "$ROUND" = "warmup" ] && RNP=$(( NP / 2 > 8 ? NP / 2 : 8 )) || RNP=$NP
    vllm bench serve --base-url http://localhost:$PORT --model "$MODEL" \
      --served-model-name benchmodel --dataset-name random \
      --random-input-len 8192 --random-output-len 1024 \
      --num-prompts $RNP --max-concurrency $CC --ignore-eos --seed 42 \
      --percentile-metrics ttft,tpot,itl \
      --save-result --result-dir $OUT --result-filename bench_${MTAG}_cc${CC}_${ROUND}.json \
      > $OUT/bench_${MTAG}_cc${CC}_${ROUND}.stdout 2>&1 || { echo "BENCH_FAIL cc=$CC $ROUND"; RC=1; }
  done
  echo "CC_DONE $MTAG cc=$CC"
done
echo "BENCH_DONE $MTAG RC=$RC"
exit $RC
