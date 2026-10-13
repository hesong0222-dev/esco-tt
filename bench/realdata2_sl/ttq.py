"""Resumable, single-process TT queue for the sisco/lisco replication of realdata2.

Every TT evaluation is appended to a JSONL cache as soon as it finishes, so a killed run resumes without
re-doing any finished evaluation. realdata2_refine.py is reused unchanged: its Pool is replaced by an
in-process serial map (one heavy process only) and rf.evaluate by the cached version.

  ttq.py eval   <tree.json> <out.json> <fit.json> [...]      MASCOT + TT(eps TT_EPS) at each fit's theta
  ttq.py refine <tree.json> <mascot_fit.json> <out.json>     realdata2_refine.py with cache, serial
Env: TT_EPS (default 1e-6), TT_SM (steps_mult, default 1), CACHE (jsonl path, required).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(BENCH))
sys.path.insert(0, BENCH)
import realdata_fit as rf  # noqa: E402
import realdata2_refine as rr  # noqa: E402
from tt import esco_tt  # noqa: E402
import numpy as np  # noqa: E402

CACHE = os.environ["CACHE"]
SM = int(os.environ.get("TT_SM", "1"))
_orig_evaluate = rf.evaluate
_cache = {}


def _key(model, theta):
    return json.dumps([model, rf.TT_EPS, SM, [round(float(x), 10) for x in theta]])


def _load():
    if os.path.exists(CACHE):
        for line in open(CACHE):
            if line.strip():
                r = json.loads(line)
                _cache[r["key"]] = r


def cached_evaluate(args):
    engine, model, theta = args
    if engine != "tt":
        return _orig_evaluate(args)
    k = _key(model, theta)
    if k in _cache:
        r = _cache[k]
        return r["loglik"], r["root"]
    M, Ne = rf.unpack(model, theta)
    tr = dict(rf.TREE, M=M.tolist(), Ne=Ne.tolist())
    d = esco_tt.loglik_detail(tr, eps=rf.TT_EPS, steps_mult=SM)
    rd = d.get("root_dist")
    if rd is not None:
        rd = np.maximum(np.array(rd), 0)
        rd = (rd / rd.sum()).tolist()
    r = {"key": k, "model": model, "eps": rf.TT_EPS, "sm": SM, "theta": [float(x) for x in theta],
         "loglik": d["loglik"], "root": rd, "rank": d["max_rank"], "sec": d["runtime"],
         "rmax_cuts": d.get("rmax_cuts"), "done": time.strftime("%F %T")}
    with open(CACHE, "a") as fh:
        fh.write(json.dumps(r) + "\n")
    _cache[k] = r
    print(f"[tt] {model} eps={rf.TT_EPS:g} sm={SM} logL={d['loglik']:.5f} rank={d['max_rank']} {d['runtime']:.0f}s",
          flush=True)
    return r["loglik"], rd


class SerialPool:
    def __init__(self, *a, initializer=None, initargs=(), **k):
        if initializer:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, f, xs):
        return [f(x) for x in xs]


rf.evaluate = cached_evaluate
rr.Pool = SerialPool


def cmd_eval(tree, out_path, fits):
    rf._init(tree)
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for p in fits:
        f = json.load(open(p))
        key = os.path.basename(p)
        tag = f"tt_{rf.TT_EPS:g}" + ("_h2" if SM == 2 else "")
        r = out.get(key, {"model": f["model"], "fit_engine": f["engine"], "fit_loglik": f["loglik"]})
        if "mascot" not in r:
            r["mascot"], r["mascot_root"] = rf.mascot_root(dict(rf.TREE, **dict(zip(
                ["M", "Ne"], [x.tolist() for x in rf.unpack(f["model"], f["theta"])]))))
        if tag not in r:
            ll, root = cached_evaluate(("tt", f["model"], f["theta"]))
            c = _cache[_key(f["model"], f["theta"])]
            r.update({tag: ll, tag + "_root": root, tag + "_rank": c["rank"], tag + "_s": c["sec"],
                      tag + "_rmax_cuts": c["rmax_cuts"]})
            print(key, tag, ll, flush=True)
        out[key] = r
        json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    _load()
    if sys.argv[1] == "eval":
        cmd_eval(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif sys.argv[1] == "refine":
        sys.argv = [sys.argv[0]] + sys.argv[2:5]
        rr.main()
