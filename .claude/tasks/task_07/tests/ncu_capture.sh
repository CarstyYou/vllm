#!/bin/bash
# usage: ncu_capture.sh <cute|dg> <E> <topk> <hidden> <inter>
# NCU-profiles the MoE GEMM kernel at decode M=1 via moe_microbench.py.
# --set full (occupancy/stalls/memory/pipe); kernel-name regex to isolate the
# MoE GEMM; launch-skip past the 3 warmup + first few steady iters, count a
# handful. dg: DG_PRINT_CONFIGS=1 prints JIT block_m/block_n/num_stages.
set -uo pipefail
BACKEND=$1
E=${2:-256}
TOPK=${3:-8}
HIDDEN=${4:-4096}
INTER=${5:-768}
VLLM_ROOT=/home/scratch.xiy_gpu/mega_inference/vllm
OUT=${OUT_DIR:-$VLLM_ROOT/.claude/tasks/task_07/results}
mkdir -p $OUT

export CUDA_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/home/scratch.xiy_gpu
export VLLM_CACHE_ROOT=/home/scratch.xiy_gpu/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/home/scratch.xiy_gpu/.cache/inductor
export TRITON_CACHE_DIR=/home/scratch.xiy_gpu/.cache/triton
export HF_HOME=/home/scratch.xiy_gpu/.cache/hf
export XDG_CACHE_HOME=/home/scratch.xiy_gpu/.cache
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=$VLLM_ROOT
mkdir -p $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $HF_HOME
source $VLLM_ROOT/.venv/bin/activate
[ "$BACKEND" = "dg" ] && export DG_PRINT_CONFIGS=1

# bundled ncu (system libcrypto lacks OPENSSL_3.3.0, same as nsys)
NHOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3
NCU="$NHOME/target/linux-desktop-glibc_2_11_3-x64/ncu"
export LD_LIBRARY_PATH="$NHOME/target/linux-desktop-glibc_2_11_3-x64:$NHOME/host/linux-desktop-glibc_2_11_3-x64:${LD_LIBRARY_PATH:-}"

if [ "$BACKEND" = "cute" ]; then
  KRE="regex:mxfp8_cute_sm120"
else
  KRE="regex:sm120_fp8_fp4_gemm"
fi

cd $VLLM_ROOT
echo "=== dg config print (if dg) captured to serve tag ==="
"$NCU" --set full --kernel-name "$KRE" --kernel-name-base demangled \
  --launch-skip 12 --launch-count 4 \
  --target-processes all \
  -f -o $OUT/ncu_${BACKEND}_decode \
  python .claude/tasks/task_07/tests/moe_microbench.py $BACKEND $E $TOPK $HIDDEN $INTER 40 \
  > $OUT/ncu_${BACKEND}_decode.log 2>&1
RC=$?
echo "NCU_RC=$RC"
ls -la $OUT/ncu_${BACKEND}_decode.ncu-rep 2>/dev/null && echo "NCU_REP_OK $BACKEND" || echo "NCU_REP_MISSING $BACKEND"
grep -iE "block_m|num_stages|GemmConfig|MICROBENCH_DONE" $OUT/ncu_${BACKEND}_decode.log 2>/dev/null | head -5
exit $RC
