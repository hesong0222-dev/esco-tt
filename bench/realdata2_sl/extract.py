"""Distinct MASCOT optima from the multistart + x0 fits + seeded fits -> one fit JSON per optimum,
named <tag>_mascot_<model>_L<round(-logL*1000)>.json (stable name).

Usage: extract.py <tree tag, e.g. sisco> ; writes those files and prints a table.
Two fits are the same optimum if their logL agree to 1e-3.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1]
DEMES = ["ans", "cha", "gal"]
for model in ["asym", "sym"]:
    fits = json.load(open(os.path.join(HERE, f"{tag}_ms_mascot_{model}.json")))
    import glob
    for p in [os.path.join(HERE, f"{tag}_mascot_{model}_x0.json")] + sorted(glob.glob(os.path.join(HERE, f"{tag}_mascot_{model}_seed_*.json"))):
        if os.path.exists(p):
            f = json.load(open(p))
            f["start"] = os.path.basename(p)
            fits.append(f)
    groups = []
    for f in sorted(fits, key=lambda f: -f["loglik"]):
        for g in groups:
            if abs(g[0]["loglik"] - f["loglik"]) < 1e-3:
                g.append(f)
                break
        else:
            groups.append([f])
    for k, g in enumerate(groups):
        f = g[0]
        f["n_starts"] = len(g)
        f["starts"] = [x.get("start", "multistart") for x in g]
        name = f"{tag}_mascot_{model}_L{int(round(-f['loglik'] * 1000))}.json"
        json.dump(f, open(os.path.join(HERE, name), "w"), indent=1)
        r = f["root_probs"]
        print(f"{name}: logL {f['loglik']:.5f}  n={len(g)}  root {DEMES[r.index(max(r))]} "
              + "/".join(f"{x:.3f}" for x in r) + "  " + " ".join(f"{a}={b:.4g}" for a, b in f["params"].items()))
