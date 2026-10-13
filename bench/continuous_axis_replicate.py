"""Replication with the ORIGINAL generator (sim/simulate.generate, pairwise m fixed): new seeds for the cells that
carry the "d>=3 slow underestimates" pattern, dense exact.  d in {2,3,4} slow, k = 10 (d=4 too) and k = 14 (d=2,3), 40 reps.
Seed = 88000000 + 100000 d + 1000 k + rep.  Output: bench/continuous_axis_replicate.csv
Usage: OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python bench/continuous_axis_replicate.py"""
import csv
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "sim")]
import simulate  # noqa: E402
from approx import independence as ind  # noqa: E402
from approx.mascot import loglik as mascot  # noqa: E402
from dense import esco_dense  # noqa: E402
from regime_metrics import metrics  # noqa: E402

CELLS = [(2, 10), (3, 10), (4, 10), (2, 14), (3, 14)]


def main():
    out, t0 = [], time.time()
    for d, k in CELLS:
        for rep in range(40):
            seed = 88000000 + 100000 * d + 1000 * k + rep
            tr = simulate.generate(d, k, "slow", rep, seed)
            ex = esco_dense.loglik(tr)
            r = dict(tree=f"rep_d{d}_k{k}_slow_r{rep}", d=d, k=k, regime="slow", rep=rep, seed=seed,
                     peak=esco_dense.max_lineages(tr), exact=ex, mascot=mascot(tr), lisco=ind.loglik(tr, "lisco"),
                     sisco=ind.loglik(tr, "sisco"), **metrics(tr))
            out.append(r)
            print(f"[{time.time() - t0:5.0f}s] {r['tree']:22s} peak {r['peak']} mascot-exact {r['mascot'] - ex:+.3f}", flush=True)
    with open(os.path.join(HERE, "continuous_axis_replicate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
