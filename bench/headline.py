"""Headline TT run on a sim/data/large tree (dense cannot do these).  Usage: headline.py tree.json eps steps_mult out.json"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tt import esco_tt
tree = json.load(open(sys.argv[1])); eps = float(sys.argv[2]); sm = int(sys.argv[3])
r = esco_tt.loglik_detail(tree, eps=eps, steps_mult=sm, verbose=True)
r.pop("root_dist", None)
r.update(file=os.path.basename(sys.argv[1]), eps=eps, steps_mult=sm)
n = m = 0
for e in tree["events"]:
    n += 1 if e["type"] == "sample" else -1; m = max(m, n)
r["max_alive"] = m
json.dump(r, open(sys.argv[4], "w"), indent=1); print(r)
