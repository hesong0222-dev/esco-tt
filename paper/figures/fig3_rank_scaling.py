"""F3: TT rank and runtime scaling. Sources: bench/rank_study.csv (isochronous quicksim trees, Ne=1, symmetric M,
seed 1 unless noted; wall times on a shared, heavily loaded Mac -> relative only) and bench/large_sweep.csv
(heterochronous sim/data/large trees, eps 1e-6, Kaggle CPU, ~1 core per job)."""
from common import *

R = pd.read_csv(f"{BENCH}/rank_study.csv")
R = R[R.seed == 1]
L = pd.read_csv(f"{BENCH}/large_sweep.csv")
EPS_LS = {1e-8: "-", 1e-6: "--", 1e-4: ":"}
EPS_MK = {1e-8: "o", 1e-6: "s", 1e-4: "^"}

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for col, (ycol, ylab, title) in enumerate((("max_rank", "max TT rank", "a  rank vs k (m = 1)"),
                                           ("runtime", "wall time (s, loaded machine)", "b  runtime vs k (m = 1)"))):
    ax = axes[col]
    for d in (2, 3, 4, 6):
        for eps in (1e-8, 1e-6, 1e-4):
            g = R[(R.d == d) & (R.m == 1.0) & (R.eps == eps)].sort_values("k")
            if len(g) == 0:
                continue
            dx = 1.2 if d == 6 else 0.0  # d = 6 points offset to the right so they do not hide d = 4 (k = 20)
            ax.plot(g.k + dx, g[ycol], EPS_LS[eps], marker=EPS_MK[eps], ms=3.5, color=D_COL[d], lw=1,
                    mfc="white" if d == 6 else D_COL[d], zorder=3 if d == 6 else 2)
    ax.set_yscale("log"); ax.set_xlabel("isochronous lineages k"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.set_xticks([10, 20, 30, 40, 60])
h = [plt.Line2D([], [], color=D_COL[d], lw=1.5, label=f"d={d}") for d in (2, 3, 4, 6)]
h += [plt.Line2D([], [], color="k", ls=EPS_LS[e], marker=EPS_MK[e], ms=3.5, lw=1, label=f"eps={e:.0e}") for e in (1e-8, 1e-6, 1e-4)]
fig.legend(handles=h, frameon=False, fontsize=7, loc="lower center", ncol=7, bbox_to_anchor=(0.36, -0.06))

ax = axes[2]
MK = {3: "o", 4: "s"}
for d in (3, 4):
    for r in ("slow", "medium", "fast"):
        g = L[(L.d == d) & (L.regime == r)]
        ax.scatter(g.alive, g["rank"], marker=MK[d], s=16, c=REG_COL[r], edgecolors="k" if d == 4 else "none", lw=.4)
h = [plt.Line2D([], [], ls="", marker="o", color=REG_COL[r], label=r) for r in ("slow", "medium", "fast")]
h += [plt.Line2D([], [], ls="", marker=MK[d], mfc="w", mec="k", label=f"d={d}") for d in (3, 4)]
ax.legend(handles=h, frameon=False, fontsize=6, loc="upper left", ncol=2)
ax.set_yscale("log"); ax.set_ylim(5, 400)
ax.set_xlabel("peak simultaneous lineages"); ax.set_ylabel("max TT rank (eps = 1e-6)")
ax.set_title("c  large heterochronous trees")
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig3_rank_scaling")
