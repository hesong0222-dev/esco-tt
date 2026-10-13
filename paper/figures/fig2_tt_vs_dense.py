"""F2: |TT - dense| vs number of dense states. Sources: bench/tt_vs_dense.csv (eps 1e-8, local),
kaggle/xcheck.csv + kaggle/dense_big.csv (eps 1e-6/1e-7, 13-14 lineages, d=4); d4_k14_medium_r1 eps 1e-7 value
from the OpenBLAS rerun recorded in LOG.md (2026-10-06 03:38)."""
import numpy as np
from common import *

t = pd.read_csv(f"{BENCH}/tt_vs_dense.csv")
t["regime"] = t.file.map(regime_of)
t["states"] = t.d.astype(float) ** t.kmax

dense = pd.read_csv(f"{KAGGLE}/dense_big.csv", header=None, names=["file", "alive", "dense", "sec"])
x = pd.read_csv(f"{KAGGLE}/xcheck.csv", header=None, names=["file", "eps", "ll", "sec"])
x = x[x.eps > 0].copy()
x["ll"] = x.ll.astype(float)
x = x.merge(dense, on="file")
# the eps=1e-7 value for d4_k14_medium_r1 in xcheck.csv (-48.5497) came from a run corrupted by the Apple
# Accelerate LAPACK heap bug; LOG.md 2026-10-06 03:38 records the OpenBLAS rerun -41.30912114229258.
bad = (x.file == "d4_k14_medium_r1.json") & (x.eps == 1e-7)
x.loc[bad, "ll"] = -41.30912114229258
x["err"] = (x.ll - x.dense).abs()
x["states"] = 4.0 ** x.alive
x["regime"] = x.file.map(regime_of)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
for r in ("slow", "medium", "fast"):
    g = t[t.regime == r]
    ax.scatter(g.states, g["diff"].abs().clip(lower=1e-10), s=10, c=REG_COL[r], label=f"{r} (Richardson)", alpha=.8, lw=0)
    ax.scatter(g.states, g.diff_h.abs().clip(lower=1e-10), s=10, facecolors="none", edgecolors=REG_COL[r], lw=.6, alpha=.6)
ax.scatter([], [], s=10, facecolors="none", edgecolors="k", lw=.6, label="plain step h")
for eps, mk, off in ((1e-6, "^", 0.75), (1e-7, "s", 1.33)):
    g = x[x.eps == eps]
    ax.scatter(g.states * off, g.err, s=22, marker=mk, c=[REG_COL[r] for r in g.regime], edgecolors="k", lw=.5)
    ax.scatter([], [], s=22, marker=mk, c="0.7", edgecolors="k", lw=.5, label=f"Kaggle d=4, eps={eps:.0e}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("dense state count $d^{k_{max}}$")
ax.set_ylabel("|logL(TT) − logL(dense)|")
ax.set_title("a  TT vs dense exact likelihood")
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=6)
ax.set_ylim(1e-10, 1e-1)

# b: estimated vs actual error (eps 1e-8 set)
ax = axes[1]
est = (t.err_step + t.err_trunc).clip(lower=1e-11)
for r in ("slow", "medium", "fast"):
    m = t.regime == r
    ax.scatter(est[m], t["diff"].abs()[m].clip(lower=1e-11), s=9, c=REG_COL[r], lw=0, alpha=.8, label=r)
lim = [1e-11, 1e-3]
ax.plot(lim, lim, "k-", lw=.6); ax.plot(lim, [3 * v for v in lim], "k--", lw=.6)
ax.text(2e-6, 1.2e-4, "3×", fontsize=7); ax.text(4e-5, 1.5e-5, "1×", fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("internal error estimate (step + truncation)")
ax.set_ylabel("actual |error| (Richardson value)")
ax.set_title("b  self-estimate vs actual (eps=1e-8)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig2_tt_vs_dense")
