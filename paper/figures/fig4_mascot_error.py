"""F4: MASCOT - exact log-likelihood at the generating parameters, by migration regime and d.
Sources: bench/dense_vs_mascot.csv (228 trees, dense exact, peak <= 4^12 states) and bench/large_sweep.csv
(47 trees, TT eps 1e-6; black ring = tree with a paired h/2 run, |TT(h) - TT(h/2)| <= 1.0e-2, i.e. smaller than
the marker; that difference mixes step and accumulated truncation error and is only an error proxy). NOTE: differences at fixed true
parameters are not inference bias."""
import numpy as np
from common import *

D = pd.read_csv(f"{BENCH}/dense_vs_mascot.csv").dropna(subset=["ll_dense"])
L = pd.read_csv(f"{BENCH}/large_sweep.csv")
L["gap"] = L.mascot - L.tt
L["err"] = (L.tt - L.sm2).abs()
rng = np.random.default_rng(0)
REGS = ("slow", "medium", "fast")


def panel(ax, df, ycol, ds, title, errcol=None):
    xt, xl = [], []
    for i, r in enumerate(REGS):
        for j, d in enumerate(ds):
            x0 = i * (len(ds) + 1) + j
            g = df[(df.regime == r) & (df.d == d)]
            xs = x0 + rng.uniform(-.25, .25, len(g))
            ax.scatter(xs, g[ycol], s=9, c=REG_COL[r], lw=0, alpha=.85)
            if errcol is not None:
                m = g[errcol].notna()
                ax.scatter(xs[m.values], g[ycol][m], s=16, facecolors="none", edgecolors="k", lw=.6)
            if len(g):
                ax.hlines(g[ycol].median(), x0 - .35, x0 + .35, color="k", lw=1.2)
            xt.append(x0); xl.append(f"d={d}")
        ax.text(i * (len(ds) + 1) + (len(ds) - 1) / 2, 1.02, r, transform=ax.get_xaxis_transform(), ha="center",
                color=REG_COL[r], fontsize=8, weight="bold")
    ax.axhline(0, color="0.5", lw=.6)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("logL(MASCOT) − logL(exact)")
    ax.set_title(title, pad=14)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [3, 2]})
panel(axes[0], D, "mascot_minus_dense", (2, 3, 4), "a  small trees (≤ 15 lineages, dense exact)")
panel(axes[1], L, "gap", (3, 4), "b  large trees (15–35 lineages, TT eps=1e-6)", errcol="err")
axes[1].scatter([], [], s=16, facecolors="none", edgecolors="k", lw=.6, label=f"paired h/2 run: |Δ| ≤ {L.err.max():.2g}")
axes[1].legend(frameon=False, loc="lower right", fontsize=6)
for ax in axes:
    ax.set_ylim(-8, 1.5)
    ax.set_yticks([-5, -1, -0.1, 0, 0.1, 1]); ax.set_yticklabels(["−5", "−1", "−0.1", "0", "0.1", "1"])
fig.tight_layout()
save(fig, "fig4_mascot_error")
