"""Tables for bench/sisco.md from bench/sisco_eval.csv (+ bench/h3n2_posterior_eval.csv if present).
Usage: .venv/bin/python bench/sisco_summarize.py"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ("mascot", "lisco", "sisco")


def load():
    rows = list(csv.DictReader(open(os.path.join(HERE, "sisco_eval.csv"))))
    for r in rows:
        r["d"] = int(r["d"])
        for m in METHODS:
            r[m + "_minus_exact"] = float(r[m + "_minus_exact"])
    p = os.path.join(HERE, "h3n2_posterior_eval.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if int(r["sample"]) < 200000:  # 10% burn-in (chain 2e6), as in h3n2_posterior_reweight
                continue
            ex = float(r["exact"])
            rows.append(dict(set="h3n2post", tree=r["sample"], d=3, regime="real",
                             **{m + "_minus_exact": float(r[m]) - ex for m in METHODS}))
    return rows


def cell(e):
    e = np.asarray(e)
    return (f"{np.median(e):+.3f}", f"{e.min():+.2f} … {e.max():+.2f}", f"{np.median(np.abs(e)):.3f}",
            f"{np.mean(np.abs(e) > 0.1):.0%}", f"{np.abs(e).max():.2f}")


def main():
    rows = load()
    groups = []
    for s in ("small", "large"):
        for d in (2, 3, 4):
            for reg in ("slow", "medium", "fast"):
                g = [r for r in rows if r["set"] == s and r["d"] == d and r["regime"] == reg]
                if g:
                    groups.append((s, d, reg, g))
    for s, lab in (("h3n2", "H3N2 fit points"), ("h3n2post", "H3N2 posterior states (post burn-in)"),
                   ("aiv", "AIV host-order fit points")):
        g = [r for r in rows if r["set"] == s]
        if g:
            groups.append((lab, 3, "real", g))
    print("| set | d | regime | n | method | median (approx − exact) | range | median abs | abs>0.1 | max abs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s, d, reg, g in groups:
        for m in METHODS:
            c = cell([r[m + "_minus_exact"] for r in g])
            print(f"| {s} | {d} | {reg} | {len(g)} | {m.upper()} | " + " | ".join(c) + " |")
    print()
    print("Closest approximation per tree/point (|approx − exact| smallest):")
    print()
    print("| set | d | regime | n | MASCOT best | LISCO best | SISCO best | median abs ratio LISCO/MASCOT | SISCO/MASCOT |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s, d, reg, g in groups:
        a = np.array([[abs(r[m + "_minus_exact"]) for m in METHODS] for r in g])
        best = np.bincount(a.argmin(1), minlength=3)
        print(f"| {s} | {d} | {reg} | {len(g)} | {best[0]} | {best[1]} | {best[2]} | "
              f"{np.median(a[:, 1] / a[:, 0]):.2f} | {np.median(a[:, 2] / a[:, 0]):.2f} |")
    # pooled over all simulated trees
    sim = [r for r in rows if r["set"] in ("small", "large")]
    a = np.array([[abs(r[m + "_minus_exact"]) for m in METHODS] for r in sim])
    print()
    print(f"Pooled simulated trees n={len(sim)}: median abs MASCOT {np.median(a[:, 0]):.3f}, LISCO {np.median(a[:, 1]):.3f},"
          f" SISCO {np.median(a[:, 2]):.3f}; MASCOT closest on {np.mean(a.argmin(1) == 0):.0%},"
          f" |SISCO| > |MASCOT| on {np.mean(a[:, 2] > a[:, 0]):.0%}, |LISCO| > |MASCOT| on {np.mean(a[:, 1] > a[:, 0]):.0%}")
    for reg in ("slow", "medium", "fast"):
        b = np.array([[abs(r[m + "_minus_exact"]) for m in METHODS] for r in sim if r["regime"] == reg])
        print(f"  {reg}: n={len(b)} median abs M/L/S {np.median(b[:, 0]):.3f}/{np.median(b[:, 1]):.3f}/{np.median(b[:, 2]):.3f};"
              f" 90th pct {np.percentile(b[:, 0], 90):.2f}/{np.percentile(b[:, 1], 90):.2f}/{np.percentile(b[:, 2], 90):.2f}")


if __name__ == "__main__":
    main()
