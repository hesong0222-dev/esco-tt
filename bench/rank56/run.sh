#!/bin/zsh
# rank5/rank6 재실행: 작업마다 프로세스 1개, BLAS 스레드 1, 3시간(10800 s) 넘으면 중단.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NPROC=1 VERBOSE=1
jobs_=( "6,10,1,1,1e-8" "6,10,10,1,1e-8" "4,20,10,1,1e-8" "6,20,1,1,1e-6" "3,40,1,1,1e-8" "4,40,1,1,1e-8" )
for j in $jobs_; do
  tag=${j//,/_}
  ( start=$(date +%s)
    nice -n 10 .venv/bin/python bench/rank_study.py bench/rank56/$tag.csv $j > bench/rank56/$tag.log 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      if (( $(date +%s) - start > 10800 )); then kill $pid; echo "KILLED_TIMEOUT $(date)" >> bench/rank56/$tag.log; break; fi
      sleep 30
    done
    echo "END $(date) elapsed=$(( $(date +%s) - start ))" >> bench/rank56/$tag.log ) &
done
wait
echo ALL_DONE
