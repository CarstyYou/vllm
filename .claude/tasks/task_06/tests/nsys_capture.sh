#!/bin/bash
# usage: nsys_capture.sh <backend> <cc> <tag>
# Captures a vLLM serve nsys trace for one bench load point. vLLM emits NVTX
# per forward step -> veloq inspect kernel:N nvtx_context.iter_index attributes
# each MoE GEMM kernel to a prefill/decode step. FI GDN for all columns (incl
# triton reference) to keep GDN path constant. prefix caching OFF, --ignore-eos.
set -uo pipefail
BACKEND=$1
CC=$2
TAG=$3
E8=${4:-}
VLLM_ROOT=/home/scratch.xiy_gpu/mega_inference/vllm
MODEL=/home/scratch.trt_llm_data/llm-models/Qwen3.5-35B-A3B-FP8
OUT=${OUT_DIR:-$VLLM_ROOT/.claude/tasks/task_06/results}
PORT=${PORT:-18600}
mkdir -p $OUT

export CUDA_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/home/scratch.xiy_gpu
export PATH=$CUDA_HOME/bin:$PATH
# home (~/.cache) is quota-full; redirect all caches to scratch/tmp
export VLLM_CACHE_ROOT=/home/scratch.xiy_gpu/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/home/scratch.xiy_gpu/.cache/inductor
export TRITON_CACHE_DIR=/home/scratch.xiy_gpu/.cache/triton
export HF_HOME=/home/scratch.xiy_gpu/.cache/hf
export XDG_CACHE_HOME=/home/scratch.xiy_gpu/.cache
export TMPDIR=/tmp
mkdir -p $VLLM_CACHE_ROOT $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $HF_HOME
source $VLLM_ROOT/.venv/bin/activate
cd $VLLM_ROOT
[ -n "$E8" ] && export VLLM_USE_DEEP_GEMM_E8M0=$E8
# cuda-13.3 bin/nsys links system libcrypto (missing OPENSSL_3.3.0); the bundled
# host/target-linux-x64/nsys + bundled libcrypto works.
NSYS_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3/host
NSYS="$NSYS_HOME/target-linux-x64/nsys"
export LD_LIBRARY_PATH="$NSYS_HOME/linux-desktop-glibc_2_11_3-x64:$NSYS_HOME/target-linux-x64:${LD_LIBRARY_PATH:-}"

for i in $(seq 1 60); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$FREE" -gt 80000 ] && break
  sleep 5
done

# nsys wraps the serve process; --capture-range=cudaProfilerApi would need code
# hooks, so instead capture the whole serve and filter to steady decode steps in
# veloq (NVTX step markers). Delay 0, no duration cap -> stop when serve killed.
SERVE_LOG=$OUT/serve_${TAG}.log
"$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=$OUT/trace_${TAG} --force-overwrite=true \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name benchmodel \
    --max-model-len 10240 --no-enable-prefix-caching \
    --moe-backend=$BACKEND --port $PORT --disable-uvicorn-access-log \
    > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID:-} 2>/dev/null || true' EXIT

for i in $(seq 1 480); do
  curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "SERVER_DIED"; tail -20 "$SERVE_LOG"; exit 1; }
  sleep 5
done
echo "server up ($TAG)"
grep -E "Using .* Fp8 MoE backend|GDN prefill" "$SERVE_LOG" > $OUT/evidence_${TAG}.txt || true

# warm the kernels (JIT/cudagraph) with a throwaway request, then the measured load
vllm bench serve --base-url http://localhost:$PORT --model "$MODEL" \
  --served-model-name benchmodel --dataset-name random \
  --random-input-len 8192 --random-output-len 64 \
  --num-prompts $CC --max-concurrency $CC --ignore-eos --seed 1 > /dev/null 2>&1
# measured window: fixed small decode load so trace stays small; NVTX steps mark decode
vllm bench serve --base-url http://localhost:$PORT --model "$MODEL" \
  --served-model-name benchmodel --dataset-name random \
  --random-input-len 8192 --random-output-len 256 \
  --num-prompts $CC --max-concurrency $CC --ignore-eos --seed 42 \
  > $OUT/bench_${TAG}.stdout 2>&1
echo "bench done ($TAG)"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 5
ls -la $OUT/trace_${TAG}.nsys-rep 2>/dev/null && echo "NSYS_REP_OK $TAG" || echo "NSYS_REP_MISSING $TAG"
