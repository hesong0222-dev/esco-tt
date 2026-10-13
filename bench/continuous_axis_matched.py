"""Matched-outflow experiment (small trees, dense exact; no TT): separates d from per-lineage migration scale.

In sim/ the PAIRWISE rate is fixed per regime, so the per-lineage outflow m_out = sum_b M[a][b] grows as (d-1).
Here the per-lineage outflow is fixed instead:  M[a][b] = g * cbar / (d-1) * U(0.5,1.5),  cbar = mean_a 1/Ne[a],
so trees with different d share the same Ne*m_out distribution.  Everything else as sim/simulate.py
(Ne[a] = 1 + 0.5a, heterochronous sampling, uniform demes, k = 10).
g in {0.05, 0.1, 0.2, 0.4, 0.8, 1.6}, d in {2, 3, 4}, 40 reps -> 720 trees (seed = 77000000 + 10000 d + 100 gi + rep).
Output: bench/continuous_axis_matched.csv.  Usage: OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python bench/continuous_axis_matched.py
"""
import csv
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "sim")]
import simulate  # noqa: E402
from approx import independence as ind  # noqa: E402
from approx.mascot import loglik as mascot  # noqa: E402
from dense import esco_dense  # noqa: E402
from regime_metrics import metrics  # noqa: E402

G = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
K, REPS = 10, 40


def main():
    out = []
    t0 = time.time()
    for d in (2, 3, 4):
        for gi, g in enumerate(G):
            for rep in range(REPS):
                seed = 77000000 + 10000 * d + 100 * gi + rep
                rng = np.random.default_rng(seed)
                Ne = [1.0 + 0.5 * a for a in range(d)]
                cbar = float(np.mean(1.0 / np.array(Ne)))
                M = np.zeros((d, d))
                for a in range(d):
                    for b in range(d):
                        if a != b:
                            M[a, b] = g * cbar / (d - 1) * rng.uniform(0.5, 1.5)
                times, demes = simulate.make_samples(K, d, rng)
                ev = simulate.simulate_tree(M, Ne, times, demes, rng)
                tr = {"demes": d, "M": M.tolist(), "Ne": Ne, "events": ev}
                ex = esco_dense.loglik(tr)
                r = dict(tree=f"m_d{d}_g{g}_r{rep}", d=d, k=K, g=g, rep=rep, seed=seed,
                         peak=esco_dense.max_lineages(tr), exact=ex, mascot=mascot(tr),
                         lisco=ind.loglik(tr, "lisco"), sisco=ind.loglik(tr, "sisco"),
                         **metrics(tr))
                out.append(r)
                print(f"[{time.time() - t0:5.0f}s] {r['tree']:18s} exact {ex:9.4f} mascot-exact {r['mascot'] - ex:+.3f}",
                      flush=True)
    with open(os.path.join(HERE, "continuous_axis_matched.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
