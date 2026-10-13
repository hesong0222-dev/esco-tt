"""Fit one simulated tree under MASCOT and under the exact structured coalescent (TT).

Usage: mle_fit.py <tree.json> <out.jsonl>        env: NPROC (TT pool size), PROF_ALL (1 = profile every param)
Appends one JSON line per finished stage to out.jsonl (a timeout keeps the finished stages).

Model: theta = (log m, log Ne_0..Ne_{d-1}); M = m (1 - I).  Stages per engine E:
  truth  : loglik at the true parameters (joint LR coverage)
  fit    : full MLE (L-BFGS-B in log space).  MASCOT: starts = truth and flat (m=0.3, Ne=1), best kept.
           TT: start = MASCOT optimum; if ll_TT(truth) is higher than the result, refit from truth, best kept.
  cross  : the other engine's loglik / root distribution at this engine's MLE
  prof_j : max loglik with parameter j fixed at its true value (per-parameter LR coverage)
  eqNe   : MLE with all Ne equal (2 params) -> LRT 'Ne differ between demes' (df = d-1)
Gradients: MASCOT forward differences h=1e-5 (smooth ODE). TT central differences h=0.05: TT rounding
(eps 1e-6) makes the surface piecewise smooth with jumps up to ~5e-3, so small steps are unusable.
"""
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize

import mle_common as C

BOUNDS = (-9.0, 6.0)
POOL = None
TREE = None


def _ev(args):
    engine, theta = args
    return C.evaluate(engine, TREE, np.asarray(theta))


def _init(tree):
    global TREE
    TREE = tree


def ev_many(engine, thetas):
    jobs = [(engine, list(map(float, t))) for t in thetas]
    if engine == "tt" and POOL is not None:
        return POOL.map(_ev, jobs)
    return [_ev(j) for j in jobs]


def maximize(engine, x0, free, log):
    """Maximize loglik over theta[free]; the other entries stay as in x0."""
    base = np.array(x0, float)
    nev = [0]

    def full(z):
        th = base.copy()
        th[free] = z
        return th

    def fg(z):
        n = len(z)
        if engine == "mascot":
            h = 1e-5
            pts = [full(z)] + [full(z + h * np.eye(n)[i]) for i in range(n)]
        else:
            h = 0.05
            pts = [full(z)] + [full(z + s * h * np.eye(n)[i]) for i in range(n) for s in (1, -1)]
        f = np.array([-r[0] if np.isfinite(r[0]) else 1e10 for r in ev_many(engine, pts)])
        nev[0] += len(pts)
        if engine == "mascot":
            g = (f[1:] - f[0]) / h
        else:
            g = (f[1::2] - f[2::2]) / (2 * h)
        print(f"    [{engine}] f={f[0]:.6f} |g|={np.linalg.norm(g):.2e} nev={nev[0]}", file=log, flush=True)
        return f[0], g

    r = minimize(fg, base[free], jac=True, method="L-BFGS-B", bounds=[BOUNDS] * len(free),
                 options={"maxiter": 60, "ftol": 1e-10, "gtol": 0.02 if engine == "tt" else 1e-5})
    th = full(r.x)
    ll, root = ev_many(engine, [th])[0]
    return {"loglik": ll, "theta": th.tolist(), "root": root, "nev": nev[0] + 1, "msg": str(r.message)}


def maximize_eqNe(engine, x0, log):
    """Equal-Ne submodel: z = (log m, log Ne)."""
    d = len(x0) - 1
    expand = lambda z: np.array([z[0]] + [z[1]] * d)
    nev = [0]

    def fg(z):
        h = 1e-5 if engine == "mascot" else 0.05
        if engine == "mascot":
            pts = [expand(z)] + [expand(z + h * np.eye(2)[i]) for i in range(2)]
        else:
            pts = [expand(z)] + [expand(z + s * h * np.eye(2)[i]) for i in range(2) for s in (1, -1)]
        f = np.array([-r[0] if np.isfinite(r[0]) else 1e10 for r in ev_many(engine, pts)])
        nev[0] += len(pts)
        g = (f[1:] - f[0]) / h if engine == "mascot" else (f[1::2] - f[2::2]) / (2 * h)
        print(f"    [{engine} eqNe] f={f[0]:.6f} |g|={np.linalg.norm(g):.2e}", file=log, flush=True)
        return f[0], g

    z0 = np.array([x0[0], float(np.mean(x0[1:]))])
    r = minimize(fg, z0, jac=True, method="L-BFGS-B", bounds=[BOUNDS] * 2,
                 options={"maxiter": 60, "ftol": 1e-10, "gtol": 0.02 if engine == "tt" else 1e-5})
    th = expand(r.x)
    ll, root = ev_many(engine, [th])[0]
    return {"loglik": ll, "theta": th.tolist(), "root": root, "nev": nev[0] + 1, "msg": str(r.message)}


def main():
    global POOL
    tree_path, out_path = sys.argv[1], sys.argv[2]
    tree = json.load(open(tree_path))
    _init(tree)
    meta = tree["meta"]
    d = tree["demes"]
    p = d + 1
    th0 = np.array(C.true_theta(d, meta["regime"]))
    tag = os.path.basename(tree_path).replace(".json", "")
    prof_all = os.environ.get("PROF_ALL", "1") == "1"
    log = sys.stderr
    nproc = int(os.environ.get("NPROC", "4"))
    if nproc > 1:
        POOL = Pool(nproc, initializer=_init, initargs=(tree,))

    def emit(rec, t0):
        rec.update(tree=tag, d=d, k=meta["k"], regime=meta["regime"], rep=meta["rep"], eps=C.EPS,
                   sec=time.time() - t0)
        with open(out_path, "a") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(f"{tag} {rec['engine']} {rec['stage']} ll={rec.get('loglik')} {rec['sec']:.0f}s", file=log, flush=True)

    def stages(engine, start):
        t0 = time.time()
        ll0, root0 = ev_many(engine, [th0])[0]
        emit({"engine": engine, "stage": "truth", "loglik": ll0, "root": root0}, t0)
        t0 = time.time()
        fits = [maximize(engine, s, list(range(p)), log) for s in start]
        if engine == "tt" and ll0 > fits[0]["loglik"]:
            fits.append(maximize(engine, th0, list(range(p)), log))
        fb = max(fits, key=lambda r: r["loglik"])
        fb.update(engine=engine, stage="fit", start_lls=[r["loglik"] for r in fits])
        emit(dict(fb), t0)
        t0 = time.time()
        other = "tt" if engine == "mascot" else "mascot"
        llx, rootx = ev_many(other, [fb["theta"]])[0]
        emit({"engine": engine, "stage": "cross", "other": other, "loglik": llx, "root": rootx}, t0)
        order = [0, "eq"] + (list(range(1, p)) if (prof_all or engine == "mascot") else [])
        for j in order:
            t0 = time.time()
            if j == "eq":
                r = maximize_eqNe(engine, fb["theta"], log)
                r.update(engine=engine, stage="eqNe")
            else:
                x0 = np.array(fb["theta"])
                x0[j] = th0[j]
                r = maximize(engine, x0, [i for i in range(p) if i != j], log)
                r.update(engine=engine, stage=f"prof_{j}", param=C.names(d)[j])
            emit(r, t0)
        return fb

    fm = stages("mascot", [th0, np.array([math.log(0.3)] + [0.0] * d)])
    stages("tt", [np.array(fm["theta"])])
    print(f"{tag} DONE", file=log, flush=True)
    if POOL is not None:
        POOL.close()


if __name__ == "__main__":
    main()
