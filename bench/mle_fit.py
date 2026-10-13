"""Fit one simulated tree under MASCOT and under the exact structured coalescent (TT).   [v2]

Usage: mle_fit.py <tree.json> <out.jsonl>
env: NPROC (TT pool size), PROF_ALL (1 = TT profile for every param), PRIOR (jsonl of earlier runs: finished stages
     of this tree are reused and skipped = resume), VALIDATE (1 = add validation stages), D4OPT (1 = noisy-surface
     optimizer settings for d=4 TT, default 1 in v2).
Appends one JSON line per finished stage to out.jsonl (a timeout keeps the finished stages).

Model: theta = (log m, log Ne_0..Ne_{d-1}); M = m (1 - I).  Stages per engine E:
  truth  : loglik at the true parameters (joint LR coverage)
  fit    : full MLE (L-BFGS-B in log space).  MASCOT: starts = truth and flat (m=0.3, Ne=1), best kept.
           TT: start = MASCOT optimum; if ll_TT(truth) is higher than the result, refit from truth, best kept.
  cross  : the other engine's loglik / root distribution at this engine's MLE
  prof_j : max loglik with parameter j fixed at its true value (per-parameter LR coverage)
  eqNe   : MLE with all Ne equal (2 params) -> LRT 'Ne differ between demes' (df = d-1)
  (VALIDATE) fit_flat : TT fit from the flat start (m=0.3, Ne=1), unrelated to truth and MASCOT
  (VALIDATE) eps7     : TT loglik with eps=1e-7 at the TT MLE and at the MASCOT MLE
Gradients: MASCOT forward differences h=1e-5 (smooth ODE). TT central differences h=0.05: TT rounding
(eps 1e-6) makes the surface piecewise smooth with jumps up to ~5e-3, so small steps are unusable.
v2 changes (2026-10-06 audit): resume via PRIOR; every optimizer returns the best-logL point it evaluated;
d=4 TT uses maxls 5, ftol 1e-5, gtol 0.1 (D4OPT); records carry base_fit_ll (the fit the stage was built on).
v3 (2026-10-08): TT evaluation cache. Every TT evaluation is appended to EVCACHE_OUT (default <out>.evals.jsonl)
as soon as it finishes; EVCACHE (comma-separated files) preloads earlier evaluations of this tree. L-BFGS-B is
deterministic, so a resumed stage replays the interrupted path from the cache and continues: same result as an
uninterrupted run, but a stage longer than one Kaggle session can now finish. Numbers are unchanged otherwise.
TT_STAGES=primary stops the TT engine after fit+cross (the pre-registered primary metrics); a later run with the
default (all) adds the remaining stages via PRIOR.
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

VERSION = 3
BOUNDS = (-9.0, 6.0)
POOL = None
TREE = None
CACHE = {}
CACHE_OUT = None


def _ev(args):
    engine, theta, eps = args
    return C.evaluate(engine, TREE, np.asarray(theta), eps)


def _init(tree):
    global TREE
    TREE = tree


def _key(engine, theta, eps):
    return (engine, C.EPS if eps is None else float(eps), tuple(theta))


def load_cache(paths, tag):
    for path in paths:
        if path and os.path.exists(path):
            for l in open(path):
                if l.strip():
                    r = json.loads(l)
                    if r["tree"] == tag:
                        CACHE[_key(r["engine"], r["theta"], r["eps"])] = (r["ll"], r["root"])


def ev_many(engine, thetas, eps=None):
    jobs = [(engine, list(map(float, t)), eps) for t in thetas]
    if engine != "tt":
        return [_ev(j) for j in jobs]
    out = [CACHE.get(_key(*j)) for j in jobs]
    todo = [i for i, o in enumerate(out) if o is None]
    it = POOL.imap_unordered(_ev_i, [(i, jobs[i]) for i in todo]) if POOL is not None else map(_ev_i, [(i, jobs[i]) for i in todo])
    for i, res in it:
        ll, root = float(res[0]), (None if res[1] is None else [float(x) for x in res[1]])
        out[i] = (ll, root)
        CACHE[_key(*jobs[i])] = out[i]
        if CACHE_OUT:
            with open(CACHE_OUT, "a") as fo:
                fo.write(json.dumps({"tree": TREE["_tag"], "engine": engine, "eps": _key(*jobs[i])[1],
                                     "theta": jobs[i][1], "ll": ll, "root": root}) + "\n")
    return out


def _ev_i(args):
    i, job = args
    return i, _ev(job)


def opt_options(engine, d):
    if engine == "mascot":
        return {"maxiter": 60, "ftol": 1e-10, "gtol": 1e-5}
    if d == 4 and os.environ.get("D4OPT", "1") == "1":
        return {"maxiter": 60, "ftol": 1e-5, "gtol": 0.1, "maxls": 5}
    return {"maxiter": 60, "ftol": 1e-10, "gtol": 0.02}


def optimize(engine, z0, to_full, log, tag=""):
    """Maximize loglik(to_full(z)) over z; returns the best evaluated point (not just the last iterate)."""
    n = len(z0)
    nev = [0]
    best = [-np.inf, None]
    h = 1e-5 if engine == "mascot" else 0.05

    def fg(z):
        if engine == "mascot":
            zs = [z] + [z + h * np.eye(n)[i] for i in range(n)]
        else:
            zs = [z] + [z + s * h * np.eye(n)[i] for i in range(n) for s in (1, -1)]
        res = ev_many(engine, [to_full(x) for x in zs])
        nev[0] += len(zs)
        lls = [r[0] if np.isfinite(r[0]) else -1e10 for r in res]
        for x, ll in zip(zs, lls):
            if ll > best[0]:
                best[0], best[1] = ll, np.array(x)
        f = -np.array(lls)
        g = (f[1:] - f[0]) / h if engine == "mascot" else (f[1::2] - f[2::2]) / (2 * h)
        print(f"    [{engine}{tag}] f={f[0]:.6f} |g|={np.linalg.norm(g):.2e} nev={nev[0]}", file=log, flush=True)
        return f[0], g

    opts = opt_options(engine, TREE["demes"])
    r = minimize(fg, np.asarray(z0, float), jac=True, method="L-BFGS-B", bounds=[BOUNDS] * n, options=opts)
    ll, root = ev_many(engine, [to_full(r.x)])[0]
    used_best = False
    z = r.x
    if best[0] > ll + 1e-12:
        z, used_best = best[1], True
        ll, root = ev_many(engine, [to_full(z)])[0]
    return {"loglik": ll, "theta": to_full(z).tolist(), "root": root, "nev": nev[0] + 1, "msg": str(r.message),
            "nit": int(r.nit), "used_best_seen": used_best, "opt": opts}



def maximize(engine, x0, free, log):
    base = np.array(x0, float)

    def full(z):
        th = base.copy()
        th[free] = z
        return th

    return optimize(engine, base[free], full, log)


def maximize_eqNe(engine, x0, log):
    d = len(x0) - 1
    return optimize(engine, np.array([x0[0], float(np.mean(x0[1:]))]),
                    lambda z: np.array([z[0]] + [z[1]] * d), log, " eqNe")


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
    tree["_tag"] = tag
    global CACHE_OUT
    CACHE_OUT = os.environ.get("EVCACHE_OUT", out_path + ".evals.jsonl")
    load_cache(os.environ.get("EVCACHE", "").split(",") + [CACHE_OUT], tag)
    tt_primary = os.environ.get("TT_STAGES", "all") == "primary"
    prof_all = os.environ.get("PROF_ALL", "1") == "1"
    validate = os.environ.get("VALIDATE", "0") == "1"
    log = sys.stderr
    nproc = int(os.environ.get("NPROC", "4"))
    if nproc > 1:
        POOL = Pool(nproc, initializer=_init, initargs=(tree,))
    import numpy, scipy
    print(f"numpy {numpy.__version__} scipy {scipy.__version__} mle_fit v{VERSION}; cached TT evals {len(CACHE)}",
          file=log, flush=True)

    # ---- prior records (resume): keep, per (engine, stage), the record built on the best fit
    prior = {}
    if os.environ.get("PRIOR") and os.path.exists(os.environ["PRIOR"]):
        recs = [json.loads(l) for l in open(os.environ["PRIOR"]) if l.strip()]
        recs = [r for r in recs if r["tree"] == tag]
        fits = {e: max([r for r in recs if r["engine"] == e and r["stage"] == "fit"], key=lambda r: r["loglik"],
                       default=None) for e in ("mascot", "tt")}
        for r in recs:
            f = fits.get(r["engine"])
            if r["stage"] == "truth" or (f is not None and r.get("_src") == f.get("_src")
                                         and r.get("base_fit_ll", f["loglik"]) == f["loglik"]):
                prior[(r["engine"], r["stage"])] = r
        print(f"{tag} resume: {len(prior)} prior stages {sorted(prior)}", file=log, flush=True)

    def emit(rec, t0, base_ll=None):
        rec.update(tree=tag, d=d, k=meta["k"], regime=meta["regime"], rep=meta["rep"], eps=C.EPS,
                   sec=time.time() - t0, version=VERSION)
        if base_ll is not None:
            rec["base_fit_ll"] = base_ll
        with open(out_path, "a") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(f"{tag} {rec['engine']} {rec['stage']} ll={rec.get('loglik')} {rec['sec']:.0f}s", file=log, flush=True)

    def stages(engine, start):
        if (engine, "truth") not in prior:
            t0 = time.time()
            ll0, root0 = ev_many(engine, [th0])[0]
            emit({"engine": engine, "stage": "truth", "loglik": ll0, "root": root0}, t0)
        else:
            ll0 = prior[(engine, "truth")]["loglik"]
        if (engine, "fit") in prior:
            fb = prior[(engine, "fit")]
        else:
            t0 = time.time()
            fits = [maximize(engine, s, list(range(p)), log) for s in start]
            if engine == "tt" and ll0 > fits[0]["loglik"]:
                fits.append(maximize(engine, th0, list(range(p)), log))
            fb = max(fits, key=lambda r: r["loglik"])
            fb.update(engine=engine, stage="fit", start_lls=[r["loglik"] for r in fits])
            emit(dict(fb), t0)
        bl = fb["loglik"]
        other = "tt" if engine == "mascot" else "mascot"
        if (engine, "cross") not in prior:
            t0 = time.time()
            llx, rootx = ev_many(other, [fb["theta"]])[0]
            emit({"engine": engine, "stage": "cross", "other": other, "loglik": llx, "root": rootx}, t0, bl)
        if engine == "tt" and tt_primary:
            return fb
        order = [0, "eq"] + (list(range(1, p)) if (prof_all or engine == "mascot") else [])
        for j in order:
            st = "eqNe" if j == "eq" else f"prof_{j}"
            if (engine, st) in prior:
                continue
            t0 = time.time()
            if j == "eq":
                r = maximize_eqNe(engine, fb["theta"], log)
            else:
                x0 = np.array(fb["theta"])
                x0[j] = th0[j]
                r = maximize(engine, x0, [i for i in range(p) if i != j], log)
                r["param"] = C.names(d)[j]
            r.update(engine=engine, stage=st)
            emit(r, t0, bl)
        return fb

    fm = stages("mascot", [th0, np.array([math.log(0.3)] + [0.0] * d)])
    ft = stages("tt", [np.array(fm["theta"])])
    if validate and not tt_primary:
        if ("tt", "fit_flat") not in prior:
            t0 = time.time()
            r = maximize("tt", np.array([math.log(0.3)] + [0.0] * d), list(range(p)), log)
            r.update(engine="tt", stage="fit_flat")
            emit(r, t0, ft["loglik"])
        if ("tt", "eps7") not in prior:
            t0 = time.time()
            (a, _), (b, _) = ev_many("tt", [ft["theta"], fm["theta"]], eps=1e-7)
            emit({"engine": "tt", "stage": "eps7", "loglik": a, "ll_at_mascot_mle": b,
                  "ll6_at_tt_mle": ft["loglik"]}, t0, ft["loglik"])
    print(f"{tag} DONE", file=log, flush=True)
    if POOL is not None:
        POOL.close()


if __name__ == "__main__":
    main()
