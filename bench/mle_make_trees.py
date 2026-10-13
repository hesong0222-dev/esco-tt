"""Generate the MLE-study trees (deterministic seeds, see mle_common.make_tree) into bench/mle/trees/."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mle_common as C

SETTINGS = [(3, 30, "slow"), (3, 50, "slow"), (4, 30, "slow"), (4, 50, "slow"), (3, 30, "medium")]
NREP = int(os.environ.get("NREP", "40"))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mle", "trees")
os.makedirs(out, exist_ok=True)
for d, k, reg in SETTINGS:
    for r in range(NREP):
        tr = C.make_tree(d, k, reg, r)
        json.dump(tr, open(f"{out}/d{d}_k{k}_{reg}_r{r:02d}.json", "w"))
print("ok", len(os.listdir(out)))
