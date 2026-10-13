"""F8: MLE study. Per-tree paired difference Delta = log m_hat(MASCOT) - log m_hat(exact) by setting, with the
pre-registered median and bootstrap 95% CI and the decision band (+-0.05 equivalence, +-0.10 systematic threshold).
Sources: bench/mle_study.csv (col Delta_m, one row per tree), bench/mle_study_summary.csv (metric Delta_log_m: median, lo,
hi, decision), both written by bench/mle_summarize.py (run 2026-10-10)."""
import numpy as np
from common import *

T = pd.read_csv(f"{BENCH}/mle_study.csv")
S = pd.read_csv(f"{BENCH}/mle_study_summary.csv")
S = S[S.metric == "Delta_log_m"].set_index("setting")
SET = [("d3k30slow", 3, 30, "slow"), ("d3k50slow", 3, 50, "slow"), ("d4k30slow", 4, 30, "slow"),
       ("d4k50slow", 4, 50, "slow"), ("d3k30medium", 3, 30, "medium")]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.axhspan(-0.05, 0.05, color="#eeeeee", zorder=0)
for yv in (-0.10, 0.10):
    ax.axhline(yv, color="#bbbbbb", lw=0.6, ls="--", zorder=0)
ax.axhline(0, color="#999999", lw=0.6, zorder=0)
rng = np.random.default_rng(1)
for i, (key, d, k, reg) in enumerate(SET):
    v = T[(T.d == d) & (T.k == k) & (T.regime == reg)].Delta_m.dropna().values
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=REG_COL[reg], alpha=0.6, lw=0, zorder=2)
    r = S.loc[key]
    ax.errorbar(i + 0.32, r["median"], yerr=[[r["median"] - r["lo"]], [r["hi"] - r["median"]]], fmt="o", ms=4,
                color="black", capsize=2, lw=1, zorder=3)
    ax.text(i, -0.62, f"n={len(v)}\n{r['decision'].replace('difference', 'diff.')}",
            ha="center", va="top", fontsize=6)
ax.set_xticks(range(len(SET)))
ax.set_xticklabels([f"d={d}, {k} tips\n{reg}" for _, d, k, reg in SET])
ax.set_ylim(-0.75, 0.6)
ax.set_ylabel("Δ = log m̂(MASCOT) − log m̂(exact)")
ax.set_title("Paired difference in migration-rate estimates (median, 95% CI)")
save(fig, "fig8_mle_delta")
