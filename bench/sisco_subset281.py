"""SISCO / LISCO / MASCOT summaries on the same simulated set as the paper's MASCOT error map (2026-10-11).

bench/sisco_eval.csv 'small' has 239 trees: the 228 with a local dense value (bench/dense_vs_mascot.csv) plus 11
four-deme trees with 13-14 lineages whose dense values come from kaggle/dense_big.csv. The error map (Fig. 4,
stats.py) uses the 228. This script drops the 11 and prints the pooled and per-regime summaries for 228 + 53 large = 281.
No new likelihood evaluations. Usage: .venv/bin/python bench/sisco_subset281.py"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(os.path.join(HERE, "sisco_eval.csv"))
D = pd.read_csv(os.path.join(HERE, "dense_vs_mascot.csv"))
local = set(D[D.ll_dense.notna()].file.str.replace(".json", "", regex=False).str.split("/").str[-1])
sim = S[(S.set == "large") | ((S.set == "small") & S.tree.isin(local))]
A = sim[["mascot_minus_exact", "lisco_minus_exact", "sisco_minus_exact"]].abs()
print(f"n={len(sim)} (small {sum(sim.set == 'small')}, large {sum(sim.set == 'large')})")
print("median abs M/L/S", A.median().round(3).tolist(), "MASCOT closest %.3f" % np.mean(A.values.argmin(1) == 0))
for r in ("slow", "medium", "fast"):
    x = A[sim.regime == r]
    print(r, len(x), "median", x.median().round(3).tolist(), "p90", x.quantile(0.9).round(2).tolist())
for st in ("small", "large"):
    for d in (3, 4):
        q = sim[(sim.set == st) & (sim.d == d) & (sim.regime == "slow")]
        print("slow", st, "d", d, "n", len(q), "SISCO median %.3f worst %.2f" % (q.sisco_minus_exact.median(), q.sisco_minus_exact.min()))
