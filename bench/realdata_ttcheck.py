"""Evaluate dense, MASCOT and TT (eps 1e-6, 1e-7, 1e-7 with h/2) at the parameters of fit JSON files."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import realdata_fit as rf
from tt import esco_tt
tree = json.load(open(sys.argv[1]))
out = {}
for path in sys.argv[3:]:
    f = json.load(open(path))
    M, Ne = rf.unpack(f["model"], f["theta"])
    tr = dict(tree, M=M.tolist(), Ne=Ne.tolist())
    r = {"fit_engine": f["engine"], "model": f["model"]}
    t0 = time.time(); r["dense"], r["dense_root"] = rf.dense_root(tr); r["t_dense"] = time.time() - t0
    r["mascot"], r["mascot_root"] = rf.mascot_root(tr)
    for tag, kw in [("tt_1e-6", dict(eps=1e-6)), ("tt_1e-7", dict(eps=1e-7)), ("tt_1e-7_h2", dict(eps=1e-7, steps_mult=2))]:
        d = esco_tt.loglik_detail(tr, **kw)
        r[tag] = d["loglik"]; r[tag + "_rank"] = d["max_rank"]; r[tag + "_s"] = d["runtime"]
        r[tag + "_root"] = d.get("root_dist")
    print(os.path.basename(path), json.dumps(r), flush=True)
    out[os.path.basename(path)] = r
    json.dump(out, open(sys.argv[2], "w"), indent=1)
