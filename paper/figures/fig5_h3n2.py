"""F5: H3N2 (Taming the BEAST MASCOT tutorial MCC tree, 24 tips, 3 demes): MASCOT vs exact (dense).
Source: bench/realdata.csv. (a) asymmetric-model MLEs (nonzero parameters); (b) profile of m10 (NY<-HK backward rate),
best of the two warm-started chains per grid point of the main chain (as in bench/realdata.md), relative to each engine's own maximum."""
import numpy as np
from common import *

R = pd.read_csv(f"{BENCH}/realdata.csv")
fit = R[(R.tree == "mcc") & (R.kind == "fit") & (R.model == "asym")].set_index("engine")
params = [("m01", "m HK→NY"), ("m10", "m NY→HK"), ("m21", "m NZ→NY"), ("Ne0", "Ne HK"), ("Ne2", "Ne NZ")]  # backward a→b
ENG = {"mascot": ("MASCOT", "#888888"), "dense": ("exact", "#c51b7d")}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.3]})
ax = axes[0]
xs = np.arange(len(params))
for i, (e, (lab, c)) in enumerate(ENG.items()):
    v = [fit.loc[e, p] for p, _ in params]
    ax.bar(xs + (i - .5) * .38, v, .38, color=c, label=f"{lab} (logL {fit.loc[e, 'loglik']:.2f})")
ax.set_xticks(xs); ax.set_xticklabels([n for _, n in params], rotation=30, ha="right")
ax.set_ylabel("MLE (backward rates a→b, /yr; Ne in yr)")
ax.set_title("a  asymmetric model, MCC tree")
ax.legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.62))

ax = axes[1]
P = R[(R.tree == "mcc") & (R.kind == "profile_m10")]
for e, (lab, c) in ENG.items():
    Pe = P[P.engine == e]
    grid = Pe[Pe.start.str.startswith("prof_")].m10.unique()  # main-chain grid; alt chain only used where both exist
    g = Pe[Pe.m10.isin(grid)].groupby("m10").loglik.max()
    best = fit.loc[e, "loglik"]
    x = np.r_[g.index.values, fit.loc[e, "m10"]]; y = np.r_[g.values - best, 0.0]
    o = np.argsort(x)
    ax.plot(x[o], y[o], "-o", ms=3, color=c, label=lab)
ax.axhline(-1.92, color="k", ls="--", lw=.7)
ax.text(1.1, -2.1, "χ²₁ 95% threshold (−1.92)", fontsize=6.5)
ax.set_ylim(-2.2, 0.1)
ax.set_xscale("log"); ax.set_xlabel("fixed m10 (backward NY→HK rate, /yr)")
ax.set_ylabel("profile ΔlogL")
ax.set_title("b  profile likelihood of m10")
ax.annotate("root NY mode", xy=(0.2, -1.8), xytext=(0.12, -1.35), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=.5))
ax.legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout()
save(fig, "fig5_h3n2")
