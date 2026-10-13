"""Collect Kaggle outputs (kaggle/out/realdata_k*/) into bench/realdata.csv and print summary tables."""
import csv
import glob
import json
import os

from scipy.stats import chi2

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{R}/kaggle/out"
DEMES = ["HK", "NY", "NZ"]
rows = []


def add(tree, kind, f, extra=None):
    r = {"tree": tree, "kind": kind, "engine": f["engine"], "model": f["model"], "loglik": f["loglik"]}
    r.update({k: v for k, v in f["params"].items()})
    for i, p in enumerate(f["root_probs"]):
        r[f"root_{DEMES[i]}"] = p
    if extra:
        r.update(extra)
    rows.append(r)


k1 = f"{OUT}/realdata_k1"
for name in ["mascot_sym", "mascot_asym", "dense_sym", "dense_asym_a"]:
    add("mcc", "fit", json.load(open(f"{k1}/{name}.json")), {"start": name})
for p in sorted(glob.glob(f"{OUT}/realdata_k[34]/post*_*_*.json")):
    b = os.path.basename(p)[:-5]
    add(b.split("_")[0], "fit", json.load(open(p)), {"start": b})
for p in sorted(glob.glob(f"{OUT}/realdata_k[25]/prof*_*.json")):
    d = json.load(open(p))
    for q in d["profile"]:
        add("mcc", f"profile_{d['param']}", {"engine": d["engine"], "model": d["model"], "loglik": q["loglik"],
                                              "params": q["params"], "root_probs": q["root_probs"]},
            {"start": os.path.basename(p)[:-5]})

keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(f"{R}/bench/realdata.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

fits = {}  # best of several starts per (tree, engine, model)
for r in rows:
    if r["kind"] == "fit":
        k = (r["tree"], r["engine"], r["model"])
        if k not in fits or r["loglik"] > fits[k]["loglik"]:
            fits[k] = r
print("tree engine  ll_sym     ll_asym    LR     p(chi2_3)  rootmax_sym  rootmax_asym")
for tree in sorted({k[0] for k in fits}):
    for eng in ["mascot", "dense"]:
        s, a = fits.get((tree, eng, "sym")), fits.get((tree, eng, "asym"))
        if not (s and a):
            continue
        lr = 2 * (a["loglik"] - s["loglik"])
        rs = max(DEMES, key=lambda x: s[f"root_{x}"])
        ra = max(DEMES, key=lambda x: a[f"root_{x}"])
        print(f"{tree:7s} {eng:6s} {s['loglik']:.4f} {a['loglik']:.4f} {lr:6.3f} {chi2.sf(lr, 3):.3f}  "
              f"{rs}({s[f'root_{rs}']:.2f}) {ra}({a[f'root_{ra}']:.2f})")
print()
for tree in sorted({k[0] for k in fits}):
    for m in ["sym", "asym"]:
        for eng in ["mascot", "dense"]:
            r = fits.get((tree, eng, m))
            if r:
                ps = {k: round(v, 3) for k, v in r.items() if k[:1] in "mN" and k != "model" and isinstance(v, float)}
                print(tree, m, eng, round(r["loglik"], 4), ps,
                      [round(r[f"root_{x}"], 3) for x in DEMES])

print()
print("profile of m10 (asym, MCC tree): max over warm-start chains; dLL = loglik - global max of that engine")
gmax = {e: fits[("mcc", e, "asym")]["loglik"] for e in ["mascot", "dense"]}
prof = {}
for r in rows:
    if r["kind"].startswith("profile"):
        k = (r["engine"], round(r["m10"], 6))
        if k not in prof or r["loglik"] > prof[k]["loglik"]:
            prof[k] = r
print("m10     mascot_dLL root(mascot)   dense_dLL root(dense)")
for v in sorted({k[1] for k in prof} - {3.0}):  # m10=3 was run only from the NY-root start (near the optimum anyway)
    out = f"{v:5.2f}"
    for e in ["mascot", "dense"]:
        r = prof.get((e, v))
        if r:
            rm = max(DEMES, key=lambda x: r[f"root_{x}"])
            out += f"  {r['loglik'] - gmax[e]:8.3f} {rm}({r[f'root_{rm}']:.2f})"
    print(out)
