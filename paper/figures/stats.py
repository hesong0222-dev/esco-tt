"""Print every summary number quoted in paper/draft.md that is derived (not copied) from a CSV.
Usage: .venv/bin/python paper/figures/stats.py"""
import os
import numpy as np, pandas as pd
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench")

t = pd.read_csv(f"{B}/tt_vs_dense.csv")
t["ad"] = t["diff"].abs(); t["adh"] = t["diff_h"].abs(); t["est"] = t.err_step + t.err_trunc
t["regime"] = t.file.str.extract(r"_(slow|medium|fast)_")
print("== tt_vs_dense.csv (eps=1e-8)")
print("n", len(t), "max|diff| rich", t.ad.max(), "at", t.loc[t.ad.idxmax(), "file"], "median", t.ad.median())
print("max|diff_h|", t.adh.max(), "median|diff_h|", t.adh.median())
for d, g in t.groupby("d"):
    print(f" d={d} n={len(g)} kmax={g.kmax.max()} max|diff|={g.ad.max():.2e} max|diff_h|={g.adh.max():.2e} maxrank={g.max_rank.max()}")
for r, g in t.groupby("regime"):
    print(f" {r} n={len(g)} max|diff|={g.ad.max():.2e} median={g.ad.median():.2e}")
print("coverage 1x", (t.ad <= t.est).mean(), "3x", (t.ad <= 3 * t.est).mean())
unc = t[t.ad > 3 * t.est]; print("uncovered at 3x:", len(unc), unc.regime.value_counts().to_dict())
print("Richardson worse than plain h:", (t.ad > t.adh).mean())

o = pd.read_csv(f"{B}/tt_vs_dense_openblas.csv")
m = t.merge(o, on="file", suffixes=("", "_o"))
print("== OpenBLAS rerun: common", len(m), "max|ll_tt-ll_tt_o|", (m.ll_tt - m.ll_tt_o).abs().max())

L = pd.read_csv(f"{B}/large_sweep.csv")
L["gap"] = L.mascot - L.tt
print("== large_sweep.csv  n", len(L), "alive range", L.alive.min(), L.alive.max())
for (d, r), g in L.groupby(["d", "regime"]):
    print(f" d={d} {r:6s} n={len(g)} gap median {g.gap.median():+.3f} [{g.gap.min():+.2f},{g.gap.max():+.2f}] "
          f"rank median {g['rank'].median():.0f} [{g['rank'].min()},{g['rank'].max()}] sec median {g.sec.median():.0f} [{g.sec.min():.0f},{g.sec.max():.0f}] alive {g.alive.min()}-{g.alive.max()}")
s2 = L.dropna(subset=["sm2"]); s2 = s2.assign(dh=(s2.tt - s2.sm2).abs())
print("h vs h/2: n", len(s2)); print(s2[["tree", "dh", "gap"]].to_string())
e7 = L.dropna(subset=["e7"]); print("eps1e-7 pairs:"); print((e7.tt - e7.e7).to_string(), e7.tree.tolist())
print("max |gap| / max |tt-sm2| per tree:", (s2.gap.abs() / s2.dh).min())
print("all-trees |gap|>0.1 fraction:", (L.gap.abs() > 0.1).mean())

D = pd.read_csv(f"{B}/dense_vs_mascot.csv").dropna(subset=["ll_dense"])
print("== dense_vs_mascot n", len(D))
for r, g in D.groupby("regime"):
    q = g.mascot_minus_dense
    print(f" {r} n={len(g)} median {q.median():+.3f} IQR [{q.quantile(.25):+.3f},{q.quantile(.75):+.3f}] min {q.min():+.3f} max {q.max():+.3f} frac|.|>0.5 {(q.abs()>0.5).mean():.2f}")
