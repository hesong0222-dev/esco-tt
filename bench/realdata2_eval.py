"""Evaluate MASCOT and exact TT (eps 1e-6 and a paired error estimate) at the parameters of fit JSON files.

Usage: realdata2_eval.py <tree.json> <out.json> <eps list, e.g. 1e-6,1e-7,1e-6h2> <fit.json> [fit.json ...]
("h2" = same eps with half the time step, steps_mult=2.)
Each TT run reports loglik, max rank, runtime and root distribution. Paired error estimate = |TT(eps) - TT(eps/10)|.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realdata_fit as rf
from tt import esco_tt

tree_path, out_path, epss = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
rf._init(tree_path)
out = json.load(open(out_path)) if os.path.exists(out_path) else {}
for path in sys.argv[4:]:
    f = json.load(open(path))
    M, Ne = rf.unpack(f["model"], f["theta"])
    tr = dict(rf.TREE, M=M.tolist(), Ne=Ne.tolist())
    key = os.path.basename(path)
    r = out.get(key, {"fit_engine": f["engine"], "model": f["model"], "fit_loglik": f["loglik"]})
    t0 = time.time(); r["mascot"], r["mascot_root"] = rf.mascot_root(tr); r["t_mascot"] = time.time() - t0
    for spec in epss:
        eps, sm = float(spec.replace("h2", "")), (2 if spec.endswith("h2") else 1)
        tag = f"tt_{eps:g}" + ("_h2" if sm == 2 else "")
        if tag in r:
            continue
        d = esco_tt.loglik_detail(tr, eps=eps, steps_mult=sm)
        r[tag], r[tag + "_rank"], r[tag + "_s"] = d["loglik"], d["max_rank"], d["runtime"]
        r[tag + "_root"] = d.get("root_dist")
        r[tag + "_rmax_cuts"] = d.get("rmax_cuts")
        print(key, tag, d["loglik"], "rank", d["max_rank"], f"{d['runtime']:.0f}s", flush=True)
        out[key] = r
        json.dump(out, open(out_path, "w"), indent=1)
    out[key] = r
    json.dump(out, open(out_path, "w"), indent=1)
