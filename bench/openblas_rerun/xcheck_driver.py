"""kaggle/xcheck.csv의 TT·MASCOT 값은 실제로는 Mac(.venv, Accelerate)에서 xargs로 돈 것 → OpenBLAS로 재계산.
작업마다 bench/kaggle_xcheck.py를 별도 프로세스로, 2시간 상한. 출력: bench/openblas_rerun/xcheck_openblas.csv"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, "..", "..")
OUT = os.path.join(H, "xcheck_openblas.csv")
JOBS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "kaggle", "xcheck_jobs.txt")
jobs = [l.split() for l in open(JOBS) if l.strip()]
# 오래 걸린 것 먼저 (옛 시간 기준 대략: 1e-7 > 1e-6 > MASCOT)
if len(sys.argv) <= 2: jobs.sort(key=lambda j: {"0": 2, "1e-6": 1, "1e-7": 0}[j[1]])
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1")
out = open(OUT, "a")
def run(j):
    t0 = time.time()
    p = subprocess.Popen(["nice", "-n", "10", os.path.join(ROOT, ".venv", "bin", "python"), os.path.join(ROOT, "bench", "kaggle_xcheck.py"), *j],
                         cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        o, _ = p.communicate(timeout=7200)
        line = o.strip().splitlines()[-1] if p.returncode == 0 else f"{j[0]},{j[1]},ERROR rc={p.returncode},{time.time()-t0:.1f}"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9); p.wait()
        line = f"{j[0]},{j[1]},TIMEOUT_2h,{time.time()-t0:.1f}"
    out.write(line + "\n"); out.flush(); print(line, flush=True)
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
    list(ex.map(run, jobs))
print("XCHECK_DONE", flush=True)
