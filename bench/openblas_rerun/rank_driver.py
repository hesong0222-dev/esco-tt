"""OpenBLAS 재계산: rank_study.csv 48행을 작업마다 별도 프로세스로, 2시간 넘으면 중단·표시."""
import csv, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
CAP = 7200
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 3
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "rank_study.csv"))))
rows.sort(key=lambda r: -float(r["runtime"]))  # 오래 걸리는 것 먼저
if len(sys.argv) > 2 and sys.argv[2] == "asc":  # 두 번째 드라이버: 짧은 것부터 (이미 시작된 작업은 건너뜀)
    rows.reverse()
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1", NPROC="1", VERBOSE="1")
def job(r):
    eps = float(r["eps"])
    spec = f"{r['d']},{r['k']},{r['m']},{r['seed']},{eps:g}" + (f",{r['rmax']}" if r["rmax"] else "")
    tag = spec.replace(",", "_")
    out = os.path.join(HERE, "rank", tag + ".csv")
    if os.path.exists(out):
        return tag, "exists"
    log = open(os.path.join(HERE, "rank", tag + ".log"), "w")
    t0 = time.time()
    p = subprocess.Popen(["nice", "-n", "10", os.path.join(ROOT, ".venv", "bin", "python"), os.path.join(ROOT, "bench", "rank_study.py"), out, spec],
                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        p.wait(timeout=CAP)
        st = f"done rc={p.returncode}"
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()
        st = "TIMEOUT_2h"
        if os.path.exists(out):
            os.remove(out)
    log.write(f"\nEND {st} elapsed={time.time()-t0:.0f}\n"); log.close()
    print(tag, st, f"{time.time()-t0:.0f}s", flush=True)
    return tag, st
with ThreadPoolExecutor(NW) as ex:
    for res in ex.map(job, rows):
        pass
print("ALL_DONE", flush=True)
