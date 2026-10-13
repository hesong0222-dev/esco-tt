"""Is MASCOT's "d>=3 underestimates under slow migration" a d effect or a migration-scale effect?

A. existing design: the 292 sim trees with an exact value (bench/sisco_eval.csv sets small+large); migration measures
   from bench/regime_metrics.py.  Pairwise M fixed per regime -> Ne*m_out grows with d (confounded).
B. matched design: bench/continuous_axis_matched.csv (720 small trees, per-lineage outflow fixed across d, dense exact).
Response: e = (MASCOT - exact)/(k-1), signed, per coalescence.  OLS with a quadratic in log10(Ne*m_out) and d dummies;
95% CIs from 2000 tree-level bootstrap resamples.
Usage: .venv/bin/python bench/continuous_axis.py  (prints the tables of bench/continuous_axis.md, writes
bench/continuous_axis.csv and paper/figures/fig_continuous_axis.{png,pdf})
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "paper", "figures")]
from regime_metrics import metrics  # noqa: E402

rng = np.random.default_rng(0)


def tree_path(r):
    if r.set == "small":
        return os.path.join(ROOT, "sim", "data", r.tree + ".json")
    sub = "large3" if r.d == 3 else ("large4x" if r.tree.endswith(("r2", "r3")) else "large")
    return os.path.join(ROOT, "sim", "data", sub, r.tree + ".json")


def load_existing():
    S = pd.read_csv(os.path.join(HERE, "sisco_eval.csv"))
    S = S[S.set.isin(["small", "large"])].copy()
    mets = [metrics(json.load(open(tree_path(r)))) for r in S.itertuples()]
    S = pd.concat([S.reset_index(drop=True), pd.DataFrame(mets)], axis=1)
    return S


def design(df, use_d=True, use_m=True):
    x = np.log10(df.nm.values)
    cols, names = [np.ones(len(df))], ["const"]
    if use_m:
        cols += [x, x ** 2]
        names += ["log10Nm", "log10Nm^2"]
    if use_d:
        for d in (3, 4):
            cols.append((df.d.values == d).astype(float))
            names.append(f"d={d}")
    return np.column_stack(cols), names


def ols(df, y, **kw):
    X, names = design(df, **kw)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    boots = []
    n = len(y)
    for _ in range(2000):
        i = rng.integers(0, n, n)
        boots.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return names, beta, lo, hi, r2


def report_ols(tag, df, y):
    print(f"\n**{tag}** (n = {len(df)}, response = (MASCOT − exact)/(k−1))\n")
    print("| model | R² | coefficients (95% bootstrap CI) |")
    print("|---|---|---|")
    for lab, kw in (("d only", dict(use_m=False)), ("Nm only", dict(use_d=False)), ("Nm + d", {})):
        names, b, lo, hi, r2 = ols(df, y, **kw)
        txt = "; ".join(f"{nm} {bb:+.4f} [{l:+.4f}, {h:+.4f}]" for nm, bb, l, h in zip(names, b, lo, hi) if nm != "const")
        print(f"| {lab} | {r2:.3f} | {txt} |")


def binned(df, col, edges):
    print(f"\n| {col} bin | " + " | ".join(f"d={d}: n, median e, frac<0" for d in (2, 3, 4)) + " |")
    print("|---|---|---|---|")
    for lo, hi in zip(edges[:-1], edges[1:]):
        cells = []
        for d in (2, 3, 4):
            g = df[(df.d == d) & (df[col] >= lo) & (df[col] < hi)]
            cells.append(f"{len(g)}, {g.e.median():+.4f}, {np.mean(g.e < 0):.0%}" if len(g) else "—")
        print(f"| [{lo:g}, {hi:g}) | " + " | ".join(cells) + " |")


def med_diff_ci(a, b, nb=2000):
    a, b = np.asarray(a), np.asarray(b)
    ds = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(nb)]
    return np.median(a) - np.median(b), *np.percentile(ds, [2.5, 97.5])


def main():
    S = load_existing()
    S["e"] = S.mascot_minus_exact / (S.k - 1)
    print("## A. existing design: migration measures by d × regime (medians)\n")
    print("| d | regime | n | Ne·m_out | outflow/coal rate | E[mig]/coal | Fitch/coal | median (MASCOT−exact) | median per coal |")
    print("|---|---|---|---|---|---|---|---|---|")
    for reg in ("slow", "medium", "fast"):
        for d in (2, 3, 4):
            g = S[(S.d == d) & (S.regime == reg)]
            print(f"| {d} | {reg} | {len(g)} | {g.nm.median():.3f} ({g.nm.min():.2f}–{g.nm.max():.2f}) | {g.ratio.median():.2f} |"
                  f" {g.emig.median():.2f} | {g.fitch.median():.2f} | {g.mascot_minus_exact.median():+.3f} | {g.e.median():+.4f} |")
    report_ols("A. existing design, all regimes", S, S.e.values)
    slowish = S[S.nm < 1.0]
    report_ols("A. existing design, Ne·m_out < 1", slowish, slowish.e.values)
    edges = [0.03, 0.1, 0.2, 0.4, 1, 3, 10, 100]
    print("\nBinned by Ne·m_out (existing design):")
    binned(S, "nm", edges)

    Mt = pd.read_csv(os.path.join(HERE, "continuous_axis_matched.csv"))
    Mt["e"] = (Mt.mascot - Mt.exact) / (Mt.k - 1)
    for m in ("lisco", "sisco"):
        Mt[f"e_{m}"] = (Mt[m] - Mt.exact) / (Mt.k - 1)
    print("\n## B. matched design (per-lineage outflow fixed across d; k = 10, 40 reps per cell)\n")
    print("| g | d | Ne·m_out median | Fitch/coal | median MASCOT−exact | IQR | frac < 0 | median abs | median abs LISCO | median abs SISCO |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g in sorted(Mt.g.unique()):
        for d in (2, 3, 4):
            x = Mt[(Mt.g == g) & (Mt.d == d)]
            raw = x.mascot - x.exact
            q1, q3 = np.percentile(raw, [25, 75])
            print(f"| {g:g} | {d} | {x.nm.median():.3f} | {x.fitch.median():.2f} | {raw.median():+.3f} | {q1:+.2f} … {q3:+.2f} |"
                  f" {np.mean(raw < 0):.0%} | {raw.abs().median():.3f} | {(x.lisco - x.exact).abs().median():.3f} |"
                  f" {(x.sisco - x.exact).abs().median():.3f} |")
    print("\nMedian difference of MASCOT−exact, d=4 minus d=2 and d=3 minus d=2, at equal g (95% bootstrap CI):\n")
    print("| g | d=3 − d=2 | d=4 − d=2 |")
    print("|---|---|---|")
    for g in sorted(Mt.g.unique()):
        base = (Mt.mascot - Mt.exact)[(Mt.g == g) & (Mt.d == 2)]
        cells = []
        for d in (3, 4):
            v, lo, hi = med_diff_ci((Mt.mascot - Mt.exact)[(Mt.g == g) & (Mt.d == d)], base)
            cells.append(f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"| {g:g} | " + " | ".join(cells) + " |")
    report_ols("B. matched design", Mt, Mt.e.values)
    print("\nBinned by Ne·m_out (matched design):")
    binned(Mt, "nm", edges)

    R = pd.read_csv(os.path.join(HERE, "continuous_axis_replicate.csv"))
    R["err"] = R.mascot - R.exact
    print("\n## C. replication with the original generator (slow, new seeds, 40 per cell, dense exact)\n")
    print("| d | k | n | Ne·m_out | median MASCOT−exact [95% CI] | mean | frac < 0 | median abs | min |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (d, k), g in R.groupby(["d", "k"]):
        e = g.err.values
        b = [np.median(rng.choice(e, len(e))) for _ in range(2000)]
        print(f"| {d} | {k} | {len(e)} | {g.nm.median():.3f} | {np.median(e):+.3f} [{np.percentile(b, 2.5):+.3f}, "
              f"{np.percentile(b, 97.5):+.3f}] | {e.mean():+.3f} | {np.mean(e < 0):.0%} | {np.median(np.abs(e)):.3f} | {e.min():+.2f} |")

    # pooled: existing + matched, does d survive?
    P = pd.concat([S[["d", "nm", "e"]], Mt[["d", "nm", "e"]]], ignore_index=True)
    report_ols("A+B pooled", P, P.e.values)

    out = pd.concat([S.assign(design="existing")[["design", "set", "tree", "d", "k", "regime", "nm", "ratio", "emig", "fitch",
                                                  "mascot_minus_exact", "lisco_minus_exact", "sisco_minus_exact"]],
                     Mt.assign(design="matched", set="matched", regime="g=" + Mt.g.astype(str),
                               mascot_minus_exact=Mt.mascot - Mt.exact, lisco_minus_exact=Mt.lisco - Mt.exact,
                               sisco_minus_exact=Mt.sisco - Mt.exact)[
                         ["design", "set", "tree", "d", "k", "regime", "nm", "ratio", "emig", "fitch",
                          "mascot_minus_exact", "lisco_minus_exact", "sisco_minus_exact"]]])
    out.to_csv(os.path.join(HERE, "continuous_axis.csv"), index=False)
    figure(S, Mt)


def figure(S, Mt):
    from common import plt, D_COL, save
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axes[0]
    for d in (2, 3, 4):
        for s, mk in (("small", "o"), ("large", "s")):
            g = S[(S.d == d) & (S.set == s)]
            if len(g):
                ax.scatter(g.nm, g.e, s=8 if s == "small" else 14, marker=mk, c=D_COL[d], lw=0, alpha=.75,
                           label=f"d={d}" if s == "small" else None)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.01)
    ax.axhline(0, color="0.5", lw=.6)
    ax.set_xlabel("Ne·m_out (per-lineage outflow × Ne)"); ax.set_ylabel("(MASCOT − exact) / (k−1)")
    ax.set_title("a  sim design (pairwise m fixed)", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax = axes[1]
    for j, d in enumerate((2, 3, 4)):
        x = Mt[Mt.d == d]
        gs = sorted(x.g.unique())
        med = [x[x.g == g].e.median() for g in gs]
        q1 = [np.percentile(x[x.g == g].e, 25) for g in gs]
        q3 = [np.percentile(x[x.g == g].e, 75) for g in gs]
        nm = [x[x.g == g].nm.median() for g in gs]
        xs = np.array(nm) * (1 + 0.06 * (j - 1))
        ax.errorbar(xs, med, yerr=[np.subtract(med, q1), np.subtract(q3, med)], fmt="o-", ms=3, lw=1, capsize=1.5,
                    color=D_COL[d], label=f"d={d}")
    ax.set_xscale("log"); ax.axhline(0, color="0.5", lw=.6)
    ax.set_xlabel("Ne·m_out (median per cell)"); ax.set_ylabel("median, IQR")
    ax.set_title("b  matched outflow (k=10)", loc="left")
    ax.legend(frameon=False, fontsize=6)
    ax = axes[2]
    for m, ls in (("mascot", "-"), ("lisco", "--"), ("sisco", ":")):
        for d in (2, 4):
            x = Mt[Mt.d == d]
            gs = sorted(x.g.unique())
            ax.plot([x[x.g == g].nm.median() for g in gs], [(x[x.g == g][m] - x[x.g == g].exact).abs().median() for g in gs],
                    ls, color=D_COL[d], lw=1.1, label=f"{m.upper()} d={d}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Ne·m_out (median per cell)"); ax.set_ylabel("median |approx − exact|")
    ax.set_title("c  matched: |error|", loc="left")
    ax.legend(frameon=False, fontsize=5, ncol=1)
    fig.tight_layout()
    save(fig, "fig_continuous_axis")


if __name__ == "__main__":
    main()
