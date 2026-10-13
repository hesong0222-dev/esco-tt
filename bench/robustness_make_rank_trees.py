"""Rank-study v2 trees (REVIEW fix #4): sim/simulate.generate = heterochronous sampling, asymmetric per-pair M
(factor*cbar*U(0.5,1.5)), Ne = 1+0.5a.  d in {2,3,4}, k in {10,20,30,40}, slow/medium/fast, 3 seeds each.
Seed = 20261006 + 100000 d + 1000 k + 10 regime_idx + rep  (disjoint from sim/data seeds, base 20261003)."""
import os, sys, json
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "sim"))
from simulate import generate, validate

out = os.path.join(ROOT, "bench", "robustness", "rank_trees")
os.makedirs(out, exist_ok=True)
for d in (2, 3, 4):
    for k in (10, 20, 30, 40):
        for ri, reg in enumerate(("slow", "medium", "fast")):
            for rep in range(3):
                seed = 20261006 + 100000 * d + 1000 * k + 10 * ri + rep
                tr = generate(d, k, reg, rep, seed)
                validate(tr)
                alive = mx = 0
                for e in tr["events"]:
                    alive += 1 if e["type"] == "sample" else -1
                    mx = max(mx, alive)
                tr["meta"]["max_alive"] = mx
                json.dump(tr, open(os.path.join(out, f"d{d}_k{k:02d}_{reg}_s{rep}.json"), "w"))
print("done")
