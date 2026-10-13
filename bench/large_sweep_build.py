"""Rebuild bench/large_sweep.csv from every Kaggle headline.py JSON under kaggle/out/ plus the MASCOT CSVs.
Usage: .venv/bin/python bench/large_sweep_build.py
Columns: tree,d,k,regime,alive,tt,rank,sec (eps 1e-6, h) | sm2,sm2_rank,sm2_sec (eps 1e-6, h/2) | e7 (eps 1e-7, h) |
mascot | err_h2 = |tt - sm2|. A tree appears only if its main eps 1e-6 run exists."""
import glob, json, os, re
import pandas as pd

E = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
main, sm2, e7 = {}, {}, {}
for f in sorted(glob.glob(f"{E}/kaggle/out/*/*.json")):
    j = json.load(open(f))
    if not isinstance(j, dict) or not {"file", "eps", "steps_mult", "loglik"} <= set(j) or not re.match(r"d\d_k\d+_", j["file"]):
        continue
    t = j["file"][:-5]
    if j["eps"] == 1e-6 and j["steps_mult"] == 1: main[t] = j
    elif j["eps"] == 1e-6 and j["steps_mult"] == 2: sm2[t] = j
    elif j["eps"] == 1e-7 and j["steps_mult"] == 1: e7[t] = j
masc = {}
for f in glob.glob(f"{E}/kaggle/large*_mascot.csv"):
    for line in open(f):
        a = line.strip().split(",")
        masc[a[0][:-5]] = float(re.search(r"-?[\d.]+(e-?\d+)?", a[2].replace("np.float64", "")).group(0))
rows = []
for t, j in sorted(main.items()):
    d, k, reg = re.match(r"d(\d)_k(\d+)_(\w+)_r\d+", t).groups()
    s = sm2.get(t, {})
    rows.append(dict(tree=t, d=int(d), k=int(k), regime=reg, alive=j["max_alive"], tt=j["loglik"], rank=j["max_rank"],
                     sec=j["runtime"], sm2=s.get("loglik"), e7=e7.get(t, {}).get("loglik"), mascot=masc.get(t),
                     sm2_rank=s.get("max_rank"), sm2_sec=s.get("runtime")))
L = pd.DataFrame(rows)
L["err_h2"] = (L.tt - L.sm2).abs()
L.to_csv(f"{E}/bench/large_sweep.csv", index=False)
print("main", len(main), "sm2", len(sm2), "e7", len(e7), "rows", len(L), "with sm2", L.sm2.notna().sum(),
      "with e7", L.e7.notna().sum(), "sm2 without main", sorted(set(sm2) - set(main)))

# per d x regime summary table (pasted into bench/large_sweep.md)
L["gap"] = L.mascot - L.tt
lines = ["| d | regime | n | MASCOT−exact median | range | rank median | rank max | n paired (h/2) | max abs(h − h/2) |",
         "|---|---|---|---|---|---|---|---|---|"]
for (d, r), g in L.groupby(["d", pd.Categorical(L.regime, ["slow", "medium", "fast"], ordered=True)], observed=True):
    e = g.err_h2.dropna()
    lines.append(f"| {d} | {r} | {len(g)} | {g.gap.median():+.3f} | {g.gap.min():+.2f} … {g.gap.max():+.2f} | "
                 f"{g['rank'].median():.0f} | {g['rank'].max()} | {len(e)} | {e.max():.1e} |" if len(e) else
                 f"| {d} | {r} | {len(g)} | {g.gap.median():+.3f} | {g.gap.min():+.2f} … {g.gap.max():+.2f} | "
                 f"{g['rank'].median():.0f} | {g['rank'].max()} | 0 | – |")
print("\n".join(lines))
