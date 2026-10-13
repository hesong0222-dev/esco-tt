"""Analysis of the MTT comparison.

usage: .venv/bin/python bench/mtt/analyze.py [KAGGLE_OUT_DIR]
inputs : bench/mtt/out/<tree>_local.npz (local MASCOT chain + IS-exact), and if given, the full kernel's output
         (<tree>_mtt_c{1,2}/<tree>.log, <tree>_exact_s{1,2}.npz, <tree>_mascot_s1.npz, full_summary.json).
outputs: bench/mtt_compare.csv (one row per tree x parameter x method) and bench/mtt/results.md (tables).
Exact reference = pooled Kaggle exact chains if present, else local importance-reweighted MASCOT draws.
KS = two-sample KS distance to the exact reference; KS_ref = 1.36 sqrt(1/ESS_method + 1/ESS_exact) is the 5%
critical value one would expect from Monte Carlo error alone (ESS in place of n).
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

KDIR = sys.argv[1] if len(sys.argv) > 1 else None


def find(pattern):
    if not KDIR:
        return []
    return sorted(glob.glob(os.path.join(KDIR, "**", pattern), recursive=True))


def mtt_chains(tree, d):
    chains, secs = [], []
    for lf in find(f"{tree}_mtt_c*/{tree}.log"):
        rows = [l.rstrip("\n").split("\t") for l in open(lf) if not l.startswith("#")]
        head, data = rows[0], [r for r in rows[1:] if len(r) >= len(rows[0]) - 1 and r[0].strip()]
        cols = [head.index(f"popSizes.{a+1}") for a in range(d)] + \
               [head.index(f"rateMatrix.{j+1}") for j in range(d * (d - 1))]
        X = np.log(np.array([[float(r[c]) for c in cols] for r in data]))
        X = X[len(X) // 10:]  # 10% burn-in
        info = json.load(open(os.path.join(os.path.dirname(lf), "run_info.json")))
        chains.append(X)
        secs.append(info["seconds"])
    return chains, secs


def am_chains(tree, engine):
    chains, secs = [], []
    for f in find(f"{tree}_{engine}_s*.npz"):
        z = np.load(f)
        chains.append(z["samples"][int(z["adapt_end"]):])
        secs.append(float(z["seconds"]))
    return chains, secs


def ess_rate(chains, secs, j):
    """Sum of per-chain ESS / total CPU hours (whole run incl. burn-in/adaptation)."""
    e = sum(C.ess(c[:, j]) for c in chains)
    return e, e / (sum(secs) / 3600.0)


rows, md = [], []
for tree in C.TREES:
    lf = os.path.join(C.HERE, "out", tree + "_local.npz")
    loc = np.load(lf) if os.path.exists(lf) else None
    t = C.load_tree(tree)
    d = len(t["Ne"])
    names = C.param_names(d)
    truth = np.exp(C.theta_true(t))
    methods = {}
    ex_c, ex_s = am_chains(tree, "exact")
    if ex_c:
        methods["exact_mcmc"] = (np.vstack(ex_c), None, ex_c, ex_s)
    if loc is not None:
        methods["exact_IS"] = (loc["is_draws"], loc["is_w"], None, None)
        methods["mascot_local"] = (loc["mascot"], None, [loc["mascot"]], [float(loc["mascot_seconds"])])
    ma_c, ma_s = am_chains(tree, "mascot")
    if ma_c:
        methods["mascot"] = (np.vstack(ma_c), None, ma_c, ma_s)
    mt_c, mt_s = mtt_chains(tree, d)
    if mt_c:
        methods["mtt"] = (np.vstack(mt_c), None, mt_c, mt_s)
    ref = "exact_mcmc" if "exact_mcmc" in methods else ("exact_IS" if "exact_IS" in methods else None)
    if ref is None:
        continue
    RX, RW = methods[ref][0], methods[ref][1]
    RW = np.ones(len(RX)) if RW is None else RW
    if ref == "exact_IS":
        ess_ref = [min(RW.sum() ** 2 / (RW ** 2).sum(), C.ess(loc["mascot"][:, j])) for j in range(len(names))]
    else:
        ess_ref = [ess_rate(methods[ref][2], methods[ref][3], j)[0] for j in range(len(names))]
    md.append(f"\n### {tree} (d={d}; reference = {ref})\n")
    md.append("| param | truth | method | mean | 95% interval | KS | KS_ref | overlap | ESS | ESS/CPU-h | R-hat |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m, (X, W, ch, secs) in methods.items():
        S = C.summarize(X, W)
        for j, nm in enumerate(names):
            ks = C.ks_weighted(RX[:, j], RW, X[:, j], W) if m != ref else 0.0
            ov = C.overlap(RX[:, j], RW, X[:, j], W) if m != ref else 1.0
            if ch is not None:
                e, er = ess_rate(ch, secs, j)
                rh = C.rhat([c[:, j] for c in ch]) if len(ch) > 1 else float("nan")
            else:
                e = ess_ref[j] if m == ref else float("nan")
                er, rh = float("nan"), float("nan")
            ksr = 1.36 * np.sqrt(1 / max(e, 1) + 1 / max(ess_ref[j], 1)) if m != ref else float("nan")
            rows.append({"tree": tree, "param": nm, "truth": truth[j], "method": m, "mean": S[j][0], "lo": S[j][1],
                         "hi": S[j][2], "ks": ks, "ks_ref": ksr, "overlap": ov, "ess": e, "ess_per_cpu_h": er,
                         "rhat": rh, "ref": ref})
            md.append(f"| {nm} | {truth[j]:.3g} | {m} | {S[j][0]:.3g} | [{S[j][1]:.3g}, {S[j][2]:.3g}] | {ks:.3f} | "
                      f"{ksr:.3f} | {ov:.3f} | {e:.0f} | {er:.0f} | {rh:.3f} |")
    if loc is not None:
        lw = np.log(loc["is_w"])
        md.append(f"\nlocal: MASCOT chain {int(loc['mascot_n_iter'])} it, {float(loc['mascot_seconds']):.0f} s, "
                  f"accept {float(loc['mascot_accept']):.2f}; IS {len(lw)} draws, weight-ESS "
                  f"{loc['is_w'].sum()**2/(loc['is_w']**2).sum():.0f}, sd(log w) {np.std(lw):.3f}")
    for f in find(f"{tree}_mtt_c*/run_info.json"):
        md.append(f"- MTT {os.path.basename(os.path.dirname(f))}: {json.dumps(json.load(open(f)))}")

out_csv = os.path.join(C.ESCO, "bench", "mtt_compare.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
open(os.path.join(C.HERE, "results.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("wrote", out_csv)
