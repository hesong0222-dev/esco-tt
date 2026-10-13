"""Summarize bench/dense_vs_mascot.csv: MASCOT - dense log-likelihood bias by regime, d, k."""
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in csv.DictReader(open(os.path.join(HERE, "dense_vs_mascot.csv"))) if r["mascot_minus_dense"]]


def stats(xs):
    x = np.array(xs)
    return f"{len(x):3d} | {x.mean():+.3f} | {np.median(x):+.3f} | {np.abs(x).mean():.3f} | {x.min():+.3f} | {x.max():+.3f}"


hdr = "| group | n | mean | median | mean abs | min | max |\n|---|---|---|---|---|---|---|"
out = []
out.append("### By regime\n" + hdr)
for reg in ("slow", "medium", "fast"):
    out.append(f"| {reg} | " + stats([float(r["mascot_minus_dense"]) for r in rows if r["regime"] == reg]) + " |")
out.append("\n### By regime x d\n" + hdr)
for reg in ("slow", "medium", "fast"):
    for d in ("2", "3", "4"):
        xs = [float(r["mascot_minus_dense"]) for r in rows if r["regime"] == reg and r["d"] == d]
        if xs:
            out.append(f"| {reg}, d={d} | " + stats(xs) + " |")
out.append("\n### By regime x k (mean MASCOT - dense; all d pooled)\n")
ks = sorted({int(r["k"]) for r in rows})
out.append("| regime | " + " | ".join(f"k={k}" for k in ks) + " |")
out.append("|---|" + "---|" * len(ks))
for reg in ("slow", "medium", "fast"):
    cells = []
    for k in ks:
        xs = [float(r["mascot_minus_dense"]) for r in rows if r["regime"] == reg and int(r["k"]) == k]
        cells.append(f"{np.mean(xs):+.3f}" if xs else "-")
    out.append(f"| {reg} | " + " | ".join(cells) + " |")
# per-coalescence bias (normalize by k-1 coalescences)
out.append("\n### Bias per coalescent event, (MASCOT - dense)/(k-1), by regime\n" + hdr)
for reg in ("slow", "medium", "fast"):
    out.append(f"| {reg} | " + stats([float(r["mascot_minus_dense"]) / (int(r["k"]) - 1) for r in rows if r["regime"] == reg]) + " |")
nskip = sum(1 for r in csv.DictReader(open(os.path.join(HERE, "dense_vs_mascot.csv"))) if not r["mascot_minus_dense"])
out.append(f"\n(dense skipped for {nskip} trees whose peak state count exceeds the dense cap 4^12.)")
print("\n".join(out))
