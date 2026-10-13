"""F7: AIV H7 (133 tips, 3 bird-order demes), three MCC summary trees of Mueller et al. (2017): every local peak's
MASCOT maximum and exact (TT eps 1e-6) maximum, each relative to that engine's best asymmetric peak on the same tree.
Sources (values copied, logL):
  masco: bench/realdata2_masco_refined.md tables "Asymmetric model" and "Symmetric model" (refined exact optima);
  sisco, lisco: bench/realdata2_sisco_lisco.md "Asymmetric peaks per tree" and "Symmetric peaks" tables.
  bound=True: the exact value is a lower bound of that peak's maximum (final values 2026-10-11 22:40: masco cha root,
  every step rejected; sisco cha root and lisco sym regular, cut twice by the 12 h Kaggle limit)."""
import numpy as np
from common import *

PEAKS = ["ans-root hub", "gal root", "cha root", "sym hub", "sym regular"]
# tree: peak -> (MASCOT max, exact max, exact_is_lower_bound)
D = {
    "masco": {"ans-root hub": (-281.478, -281.507, False), "gal root": (-284.569, -286.180, False),
              "cha root": (-285.041, -292.239, True), "sym hub": (-288.531, -296.304, False),
              "sym regular": (-294.624, -301.930, False)},
    "sisco": {"ans-root hub": (-266.604, -266.621, False), "gal root": (-264.705, -266.060, False),
              "cha root": (-264.670, -270.625, True), "sym hub": (-276.101, -283.237, False),
              "sym regular": (-274.067, -281.047, False)},
    "lisco": {"ans-root hub": (-261.846, -261.906, False), "gal root": (-257.131, -258.572, False),
              "cha root": (-256.886, -262.365, False), "sym hub": (-271.007, -277.873, False),
              "sym regular": (-267.082, -274.044, True)},
}
ASYM = PEAKS[:3]
C_M, C_E = "#888888", "#c51b7d"
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
x = np.arange(len(PEAKS))
for ax, lab, (tree, P) in zip(axes, "abc", D.items()):
    bm = max(P[p][0] for p in ASYM); be = max(P[p][1] for p in ASYM)
    ym = np.array([P[p][0] - bm for p in PEAKS]); ye = np.array([P[p][1] - be for p in PEAKS])
    bnd = np.array([P[p][2] for p in PEAKS])
    ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
    ax.axvline(2.5, color="#dddddd", lw=0.6, ls=":", zorder=0)
    ax.scatter(x - 0.12, ym, s=22, color=C_M, label="MASCOT max", zorder=3)
    ax.scatter(x[~bnd] + 0.12, ye[~bnd], s=22, color=C_E, label="exact max (refined)", zorder=3)
    ax.scatter(x[bnd] + 0.12, ye[bnd], s=22, facecolors="white", edgecolors=C_E, lw=1.0,
               label="exact, lower bound", zorder=3)
    for xi, yi in zip(x[bnd] + 0.12, ye[bnd]):
        ax.annotate("", xy=(xi, yi + 1.6), xytext=(xi, yi + 0.25),
                    arrowprops=dict(arrowstyle="->", color=C_E, lw=0.8))
    bestm = PEAKS[int(np.argmax(ym[:3]))]; beste = PEAKS[int(np.argmax(ye[:3]))]
    flip = "same best peak" if bestm == beste else "best peak differs"
    ax.set_title(f"{lab}  {tree} MCC tree\n{flip}", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(PEAKS, rotation=35, ha="right")
    ax.set_ylim(-22, 2.5)
axes[0].set_ylabel("log L − best asymmetric peak\n(of the same engine)")
axes[0].legend(frameon=False, loc="lower left", fontsize=6.5)
save(fig, "fig7_aiv_peaks")
