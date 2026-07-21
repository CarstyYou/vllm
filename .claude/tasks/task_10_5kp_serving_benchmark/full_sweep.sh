#!/bin/bash
# task_10 full sweep: both backends x all cc, warmup1 + formal1.
UUID=GPU-ab3d387a-b17d-bd26-a5cf-7968a2129522
CC="1 4 8 16 32 64 128"
RC=0
for BK in cute_sm120_mxfp8_32 deep_gemm_mxfp8_32; do
  echo "=================== RUN $BK ==================="
  enroot start --rw \
    --mount /lustre/raplab/client/xiy/workspace:/workspace \
    --mount /home/xiy/workspace/models:/models \
    fi-ci-cu130 bash -c "export CUDA_VISIBLE_DEVICES=$UUID; bash /workspace/bench_5kp.sh $BK \"$CC\" \"warmup r1\""
  brc=$?
  [ "$brc" -ne 0 ] && { echo "BACKEND_FAIL $BK rc=$brc"; RC=1; }
  echo "=================== END $BK (rc=$brc) ==================="
done
echo "ALL_BACKENDS_DONE rc=$RC"
exit $RC
