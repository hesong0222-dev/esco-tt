#!/bin/bash
# Single-process TT queue (one heavy process at a time). Resumable: each TT value is appended to
# <tree>_ttcache[_h2].jsonl when it finishes; rerunning this script skips finished work. Task order: plan.py.
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TT_EPS=1e-6 ITERS=8
O=bench/realdata2_sl
PY="nice -n 10 .venv/bin/python $O/ttq.py"
while true; do
  t=$(.venv/bin/python $O/plan.py); set -- $t
  case "$1" in
    "") break ;;
    wait) sleep 120; continue ;;
    ev) CACHE=$O/$2_ttcache.jsonl $PY eval data/aivorder_$2_mcc.json $O/$2_eval.json $O/$3 >> $O/$2_eval.log 2>&1 ;;
    ref) CACHE=$O/$2_ttcache.jsonl $PY refine data/aivorder_$2_mcc.json $O/$2_mascot_$3.json $O/$2_refine_$3.json >> $O/$2_refine_$3.log 2>&1 ;;
    pair) TT_SM=2 CACHE=$O/$2_ttcache_h2.jsonl $PY eval data/aivorder_$2_mcc.json $O/$2_pair_h2.json $O/$3 >> $O/$2_pair.log 2>&1 ;;
  esac
  rc=$?
  echo "$(date '+%F %T') rc=$rc $t" >> $O/queue.out
  [ $rc -ne 0 ] && echo "$t" >> $O/failed.txt
done
echo "$(date '+%F %T') queue finished" >> $O/queue.out
