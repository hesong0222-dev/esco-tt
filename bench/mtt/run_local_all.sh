#!/bin/bash
# Sequential local runs (one process at a time). ~1 h total on the M4 under moderate load.
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1
for t in d2_k16_slow_r1:12000 d2_k16_medium_r0:12000 d3_k10_slow_r1:15000 d3_k10_medium_r2:15000; do
  nice -n 10 .venv/bin/python bench/mtt/run_local.py ${t%%:*} ${t##*:} 1500
done
echo DONE
