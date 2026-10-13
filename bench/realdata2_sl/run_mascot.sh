#!/bin/bash
# MASCOT multistart (10 random starts, seed 1, as for masco) + fit from x0=0, both models, sisco and lisco trees.
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NPROC=4
O=bench/realdata2_sl
for m in sisco lisco; do
  T=data/aivorder_${m}_mcc.json
  for mod in sym asym; do
    [ -s $O/${m}_ms_mascot_$mod.json ] || nice -n 10 .venv/bin/python bench/realdata_multistart.py $T $mod $O/${m}_ms_mascot_$mod.json mascot 10 > $O/${m}_ms_mascot_$mod.log 2>&1
    [ -s $O/${m}_mascot_${mod}_x0.json ] || nice -n 10 .venv/bin/python bench/realdata_fit.py fit mascot $mod $T $O/${m}_mascot_${mod}_x0.json > /dev/null 2> $O/${m}_mascot_${mod}_x0.log
  done
done
echo DONE
