#!/bin/bash
# task_10 5KP serving harness — r2/r3 supplement rounds (runs INSIDE fi-ci-cu130).
# Identical config to bench_5kp.sh (r1); only output naming differs so r1
# artifacts are never overwritten. usage: bench_5kp_r23.sh <backend> "<cc list>" "<rounds>"
#   e.g. bench_5kp_r23.sh cute_sm120_mxfp8_32 "1 4" "warmup2 r2 r3"
set -uo pipefail
BACKEND=$1
CCLIST="$2"
ROUNDS="$3"
for R in $ROUNDS; do
  [ "$R" = r1 ] || [ "$R" = warmup ] && { echo "FATAL round '$R' reserved for r1 run — refuse to overwrite"; exit 2; }
done
SRC=/workspace/h20_baseline/vllm
SP=$SRC/.venv/lib/python3.12/site-packages
FI=/workspace/flashinfer
MODEL=/models/Qwen3.5-35B-A3B-FP8
OUT=/workspace/task_10_out
PORT=18600
PY=/opt/conda/envs/py312/bin/python
export PYTHONPATH=$SRC:$FI:$SP
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/workspace/fi_ws
export VLLM_CACHE_ROOT=/workspace/vllm_cache
export HF_HUB_OFFLINE=1
[ "$BACKEND" = deep_gemm_mxfp8_32 ] && export VLLM_USE_DEEP_GEMM_E8M0=1
RC=0
mkdir -p "$OUT" "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT"
cd /tmp
MTAG=Qwen3.5-35B-A3B-FP8_tp1_${BACKEND}
SLOG=$OUT/serve_${MTAG}_r23.log

# --- GPU binding guard + evidence (avoid accidental GPU0) ---
EXPECT_UUID=GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
case "${CUDA_VISIBLE_DEVICES:-}" in
  *ab3d387a*) : ;;
  *) echo "FATAL CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' != GPU2 ($EXPECT_UUID)"; exit 2 ;;
esac
BIND=$OUT/gpu_binding_${MTAG}_r23.txt
{ echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; echo "expect_gpu2_uuid=$EXPECT_UUID"; } > "$BIND"
$PY -c "import torch; print('visible_device_count', torch.cuda.device_count()); print('device0_name', torch.cuda.get_device_name(0)); print('device0_cap', torch.cuda.get_device_capability(0))" >> "$BIND" 2>&1
echo "--- model mount check ---" >> "$BIND"; ls "$MODEL/config.json" >> "$BIND" 2>&1
cat "$BIND"

$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name benchmodel \
  --max-model-len 10240 --no-enable-prefix-caching --tensor-parallel-size 1 \
  --max-num-seqs 128 \
  --moe-backend="$BACKEND" --gdn-prefill-backend=triton \
  --port $PORT --disable-uvicorn-access-log > "$SLOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null || true' EXIT

for i in $(seq 1 720); do
  curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER_DIED $MTAG"; tail -40 "$SLOG"; exit 1; }
  sleep 5
done
curl -sf http://localhost:$PORT/health >/dev/null || { echo "SERVER_TIMEOUT $MTAG"; tail -40 "$SLOG"; exit 1; }
echo "SERVER_UP $MTAG"
grep -iE "MoE backend|moe_backend|Fp8 MoE|CUTE_SM120_MXFP8|DEEP_GEMM_MXFP8|Using .* backend" "$SLOG" | tail -8 | tee "$OUT/evidence_${MTAG}_r23.txt"

for CC in $CCLIST; do
  NP=$(( CC*4 > 16 ? CC*4 : 16 ))
  for R in $ROUNDS; do
    case "$R" in warmup*) RNP=$(( NP/2 > 8 ? NP/2 : 8 ));; *) RNP=$NP;; esac
    $PY -m vllm.entrypoints.cli.main bench serve --base-url http://localhost:$PORT \
      --model "$MODEL" --served-model-name benchmodel \
      --dataset-name random --random-input-len 8000 --random-output-len 1000 \
      --num-prompts $RNP --max-concurrency $CC --ignore-eos --seed 42 \
      --percentile-metrics ttft,tpot,itl --save-result --result-dir "$OUT" \
      --result-filename bench_${MTAG}_cc${CC}_${R}.json > "$OUT/bench_${MTAG}_cc${CC}_${R}.stdout" 2>&1 \
      || { echo "BENCH_FAIL cc=$CC $R"; RC=1; }
  done
  echo "CC_DONE $MTAG cc=$CC"
done
kill $SPID 2>/dev/null || true
wait $SPID 2>/dev/null || true
echo "ALL_DONE $MTAG rc=$RC"
exit $RC
