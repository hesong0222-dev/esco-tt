"""Importance-reweight the H3N2 MASCOT posterior sample (Taming-the-BEAST tutorial run) to the exact-ESCO posterior.

Input: bench/h3n2_posterior_eval.csv (per logged state: exact, converged MASCOT, the run's own logged structured-
coalescent density `beast_logged` = column `Mascot` of the log, root-deme distributions) + data/raw/H3N2.log
(Tree.height, clockRate). The chain's states are a sample from pi_S ∝ prior * treeLik * exp(l_S), l_S = beast_logged.
Two posteriors over the same states (priors and sequence likelihood cancel in every ratio):
  C  converged MASCOT model (approx/mascot.py)                w ∝ exp(l_ours  - l_S)
  E  exact ESCO                                               w ∝ exp(l_exact - l_S)
The model-error step is C -> E, weights ∝ exp(l_exact - l_ours) relative to C.
Root deme: posterior mean of the per-state root distribution of the model in use (MASCOT root for C; exact for E).
Usage: .venv/bin/python bench/h3n2_posterior_reweight.py [burnin_fraction=0.1]
Writes bench/h3n2_posterior_reweight.csv (one row per posterior x quantity) and prints the tables of the .md.
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE]
from beast_io import load_log  # noqa: E402

DEM = {"Hong_Kong": "HK", "New_York": "NY", "New_Zealand": "NZ"}
rng = np.random.default_rng(0)


def kish(w):
    w = w / w.sum()
    return 1.0 / (w ** 2).sum()


def wquant(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return np.interp(q, c, x[o])


def whpd(x, w, mass=0.95):
    o = np.argsort(x)
    xs, ws = x[o], w[o] / w.sum()
    c = np.concatenate([[0], np.cumsum(ws)])
    best = (xs[0], xs[-1])
    j = 0
    for i in range(len(xs)):
        while j < len(xs) and c[j + 1] - c[i] < mass:
            j += 1
        if j >= len(xs):
            break
        if xs[j] - xs[i] < best[1] - best[0]:
            best = (xs[i], xs[j])
    return best


def load(burn):
    E = pd.read_csv(os.path.join(HERE, "h3n2_posterior_eval.csv"))
    L = pd.DataFrame(load_log(os.path.join(ROOT, "data", "raw", "H3N2.log")))
    L = L[["Sample", "Tree.height", "clockRate"]].astype(float).rename(columns={"Sample": "sample"})
    E = E.merge(L, on="sample")
    E = E[E["sample"] >= burn * 2_000_000].sort_values("sample").reset_index(drop=True)
    return E


def main():
    burn = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    E = load(burn)
    n = len(E)
    lB, lC, lE = E.beast_logged.values, E.mascot.values, E.exact.values  # lB = the run's own target density
    W = {"C": np.exp((lC - lB) - (lC - lB).max()), "E": np.exp((lE - lB) - (lE - lB).max())}
    wCE = np.exp((lE - lC) - (lE - lC).max())
    print(f"states after {burn:.0%} burn-in: {n} (of 401); exact engine counts: {E.engine.value_counts().to_dict()}")
    print(f"l_exact - l_ours (MASCOT model error, sign flipped): median {np.median(lE - lC):+.3f}, "
          f"range {np.min(lE - lC):+.3f} … {np.max(lE - lC):+.3f}, sd {np.std(lE - lC):.3f}")
    print(f"Kish ESS: sample->C {kish(W['C']):.1f}, sample->E {kish(W['E']):.1f}, C->E {kish(wCE):.1f} (n = {n});"
          f" max normalised weight sample->E {W['E'].max() / W['E'].sum():.3f}")
    # weight-ESS uncertainty: bootstrap of states
    bs = [kish(W["E"][rng.integers(0, n, n)]) for _ in range(1000)]
    print(f"  sample->E ESS bootstrap 95%: {np.percentile(bs, 2.5):.1f} … {np.percentile(bs, 97.5):.1f}")
    corr = np.corrcoef(lE - lC, E["Tree.height"])[0, 1]
    print(f"corr(l_exact - l_ours, tree height) = {corr:+.2f}")

    qs = [f"Ne_{a}" for a in DEM] + [f"m_{a}_to_{b}" for a in DEM for b in DEM if a != b] + ["Tree.height", "clockRate"]
    rows = []
    print("\n| quantity | posterior | mean | median | 95% HPD | shift of mean vs C |")
    print("|---|---|---|---|---|---|")
    for q in qs:
        x = E[q].values.astype(float)
        means = {}
        for k in ("C", "E"):
            w = W[k]
            m = float((w * x).sum() / w.sum())
            means[k] = m
            lo, hi = whpd(x, w)
            med = wquant(x, w, 0.5)
            rows.append(dict(quantity=q, posterior=k, mean=m, median=med, hpd_lo=lo, hpd_hi=hi,
                             ess=kish(w)))
            sC = f"{(m / means['C'] - 1) * 100:+.1f}%" if k == "E" else ""
            lab = q.replace("Hong_Kong", "HK").replace("New_York", "NY").replace("New_Zealand", "NZ")
            print(f"| {lab} | {k} | {m:.4g} | {med:.4g} | {lo:.3g} – {hi:.3g} | {sC} |")
    print("\nRoot-deme posterior probability (mean of per-state root distribution):\n")
    print("| posterior | root model | HK | NY | NZ |")
    print("|---|---|---|---|---|")
    for k, model in (("C", "mascot"), ("E", "mascot"), ("E", "exact")):
        w = W[k] / W[k].sum()
        v = [float((w * E[f"root_{model}_{a}"]).sum()) for a in ("HK", "NY", "NZ")]
        rows.append(dict(quantity=f"root_{model}", posterior=k, mean=None, median=None, hpd_lo=None, hpd_hi=None,
                         ess=kish(W[k]), root_HK=v[0], root_NY=v[1], root_NZ=v[2]))
        print(f"| {k} | {model} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} |")
    # per-state root change at the same state
    d = (E.root_exact_HK - E.root_mascot_HK).abs()
    print(f"\nper-state |P_exact(root HK) - P_MASCOT(root HK)|: median {d.median():.3f}, max {d.max():.3f}")
    # bootstrap of the E-posterior mean shift for key quantities (resampling states)
    print("\nBootstrap (1000 resamples of states) of the C->E relative mean shift:\n")
    print("| quantity | shift | 95% CI |")
    print("|---|---|---|")
    for q in qs:
        x = E[q].values.astype(float)
        out = []
        for _ in range(1000):
            i = rng.integers(0, n, n)
            mc = (W["C"][i] * x[i]).sum() / W["C"][i].sum()
            me = (W["E"][i] * x[i]).sum() / W["E"][i].sum()
            out.append(me / mc - 1)
        mc = (W["C"] * x).sum() / W["C"].sum()
        me = (W["E"] * x).sum() / W["E"].sum()
        lab = q.replace("Hong_Kong", "HK").replace("New_York", "NY").replace("New_Zealand", "NZ")
        print(f"| {lab} | {(me / mc - 1) * 100:+.1f}% | {np.percentile(out, 2.5) * 100:+.1f}% … {np.percentile(out, 97.5) * 100:+.1f}% |")
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    with open(os.path.join(HERE, "h3n2_posterior_reweight.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "posterior", "mean", "median", "hpd_lo", "hpd_hi", "ess",
                                          "root_HK", "root_NY", "root_NZ"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
