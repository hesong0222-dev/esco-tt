#!/bin/bash
# Extra MASCOT fits on sisco/lisco started from each MASCOT optimum of the masco tree (bench/realdata2/), so a basin
# found on masco is not missed only because no random start landed in it. Then extract distinct optima.
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NPROC=4
O=bench/realdata2_sl; R=bench/realdata2
m=$1
T=data/aivorder_${m}_mcc.json
for s in asym:best asym:mode_A asym:mode_C asym:ansroot sym:hub sym:regular; do
  mod=${s%%:*}; n=${s#*:}
  out=$O/${m}_mascot_${mod}_seed_masco_${n}.json
  [ -s $out ] || nice -n 10 .venv/bin/python bench/realdata_fit.py fit mascot $mod $T $out $R/aivorder_mascot_${mod}_${n}.json > /dev/null 2> ${out%.json}.log
done
.venv/bin/python $O/extract.py $m > $O/${m}_optima.txt && touch $O/${m}_seeds_done
