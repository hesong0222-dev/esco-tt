"""Rank/runtime study for tt/.  Usage: python bench/rank_study.py out.csv "d,k,m,seed,eps" ...  (runs jobs in parallel, appends rows)."""
import sys, os, json, time, csv
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tt import esco_tt, quicksim


def run(job):
    d, k, m, seed, eps, rmax = job
    tr = quicksim.simulate(k, d, quicksim.migration_matrix(d, m), [1.0] * d, seed=seed)
    try:
        r = esco_tt.loglik_detail(tr, eps=eps, rmax=rmax, verbose=os.environ.get("VERBOSE") == "1")
        return dict(d=d, k=k, m=m, seed=seed, eps=eps, rmax=rmax, loglik=r["loglik"], max_rank=r["max_rank"],
                    runtime=round(r["runtime"], 1), n_steps=r["n_steps"], n_mig=tr["meta"]["n_mig"], tmrca=round(tr["meta"]["tmrca"], 3))
    except Exception as e:  # noqa
        return dict(d=d, k=k, m=m, seed=seed, eps=eps, rmax=rmax, loglik=float("nan"), max_rank=-1, runtime=-1, n_steps=-1,
                    n_mig=tr["meta"]["n_mig"], tmrca=round(tr["meta"]["tmrca"], 3), err=str(e)[:80])


if __name__ == "__main__":
    out = sys.argv[1]
    jobs = []
    for a in sys.argv[2:]:
        f = a.split(",")
        jobs.append((int(f[0]), int(f[1]), float(f[2]), int(f[3]), float(f[4]), int(f[5]) if len(f) > 5 else None))
    nproc = int(os.environ.get("NPROC", "6"))
    fields = ["d", "k", "m", "seed", "eps", "rmax", "loglik", "max_rank", "runtime", "n_steps", "n_mig", "tmrca", "err"]
    new = not os.path.exists(out)
    with open(out, "a") as f, Pool(nproc) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for row in pool.imap_unordered(run, jobs):
            w.writerow(row)
            f.flush()
            print(row, flush=True)
