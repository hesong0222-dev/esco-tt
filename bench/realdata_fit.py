"""ML fits of the structured coalescent on a fixed real tree: MASCOT vs exact (dense) ESCO.

Parameters are optimized in log space with L-BFGS-B; gradients by forward finite differences
evaluated in parallel (multiprocessing). Models:
  asym: 6 backward migration rates M[a][b] (a!=b) + 3 Ne
  sym : 3 migration rates (M[a][b]=M[b][a]) + 3 Ne
Usage:
  realdata_fit.py fit  <engine> <model> <tree.json> <out.json> [x0.json]
  realdata_fit.py prof <engine> <model> <tree.json> <out.json> <param_index> <x0.json> <v1,v2,...>
engine in {mascot, dense}. Output JSON holds loglik, params, root-deme probabilities.
"""
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)  # kaggle flat layout

import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize

try:
    from dense import esco_dense
    from approx import mascot
except ImportError:  # flat layout
    import esco_dense
    import mascot

D = 3
OFF = [(a, b) for a in range(D) for b in range(D) if a != b]
SYM = [(0, 1), (0, 2), (1, 2)]


def set_demes(d):
    """Number of demes is taken from the tree (H3N2: 3; other datasets may have more)."""
    global D, OFF, SYM
    D = d
    OFF = [(a, b) for a in range(d) for b in range(d) if a != b]
    SYM = [(a, b) for a in range(d) for b in range(a + 1, d)]


def npar(model):
    return len(names(model))


def unpack(model, theta):
    v = np.exp(np.asarray(theta, float))
    M = np.zeros((D, D))
    if model == "asym":
        for k, (a, b) in enumerate(OFF):
            M[a, b] = v[k]
        Ne = v[len(OFF):]
    else:
        for k, (a, b) in enumerate(SYM):
            M[a, b] = M[b, a] = v[k]
        Ne = v[len(SYM):]
    return M, Ne


def names(model):
    if model == "asym":
        return [f"m{a}{b}" for a, b in OFF] + [f"Ne{a}" for a in range(D)]
    return [f"m{a}{b}" for a, b in SYM] + [f"Ne{a}" for a in range(D)]


def dense_root(tree):
    """esco_dense.loglik loop, also returning the root-lineage deme distribution."""
    from scipy.linalg import expm
    mdl = esco_dense.Model(tree["M"], tree["Ne"])
    d = mdl.d
    P = np.ones(())
    ids, t, logL = [], None, 0.0
    Dd, E = esco_dense._counts_D(0, mdl)
    for ev in tree["events"]:
        if t is not None and ev["t"] > t and ids:
            dt, n = ev["t"] - t, len(ids)
            if n == 1:
                P = expm(mdl.Q.T * dt) @ P
                P = P / P.sum()
            elif d ** n <= 4096:
                P, ls = esco_dense.propagate_expm(P, dt, mdl, Dd)
                logL += ls
            else:
                P, ls = esco_dense.propagate_unif(P, dt, mdl, Dd, E)
                logL += ls
        t = ev["t"]
        if ev["type"] == "sample":
            e = np.zeros(d)
            e[ev["deme"]] = 1.0
            P = np.multiply.outer(P, e)
            ids.append(ev["id"])
        else:
            i, j = ids.index(ev["a"]), ids.index(ev["b"])
            P = np.ascontiguousarray(np.diagonal(P, axis1=i, axis2=j) * mdl.inv_ne)
            ids = [x for x in ids if x not in (ev["a"], ev["b"])] + [ev["new"]]
            s = P.sum()
            if s <= 0:
                return -math.inf, None
            logL += math.log(s)
            P = P / s
        Dd, E = esco_dense._counts_D(len(ids), mdl)
    return logL, P.ravel().tolist()


def mascot_root(tree):
    """mascot.loglik loop, also returning the root-lineage deme probabilities."""
    from scipy.integrate import solve_ivp
    M = np.array(tree["M"], float)
    np.fill_diagonal(M, 0.0)
    lam = 1.0 / np.array(tree["Ne"], float)
    out = M.sum(axis=1)
    ids, p, logL, t = [], np.zeros((0, D)), 0.0, None
    for ev in tree["events"]:
        if t is not None and ev["t"] > t and ids:
            n = len(ids)
            sol = solve_ivp(mascot._rhs, (t, ev["t"]), np.concatenate([p.ravel(), [0.0]]), method="DOP853",
                            rtol=1e-10, atol=1e-12, args=(n, D, M.T, out, lam))
            y = sol.y[:, -1]
            p = np.maximum(y[:-1].reshape(n, D), 0.0)
            p /= p.sum(axis=1, keepdims=True)
            logL += y[-1]
        t = ev["t"]
        if ev["type"] == "sample":
            row = np.zeros(D)
            row[ev["deme"]] = 1.0
            p = np.vstack([p, row])
            ids.append(ev["id"])
        else:
            i, j = ids.index(ev["a"]), ids.index(ev["b"])
            w = lam * p[i] * p[j]
            s = w.sum()
            if s <= 0:
                return -math.inf, None
            logL += math.log(s)
            keep = [x for x in range(len(ids)) if x not in (i, j)]
            p = np.vstack([p[keep], w / s])
            ids = [ids[x] for x in keep] + [ev["new"]]
    return logL, p[-1].tolist()


TREE = None


def _init(path):
    global TREE
    TREE = json.load(open(path))
    set_demes(TREE.get("demes", 3))


def evaluate(args):
    engine, model, theta = args
    M, Ne = unpack(model, theta)
    tr = dict(TREE, M=M.tolist(), Ne=Ne.tolist())
    if engine == "tt":
        return tt_root(tr)
    ll, root = (dense_root if engine == "dense" else mascot_root)(tr)
    return ll, root


TT_EPS = float(os.environ.get("TT_EPS", "1e-6"))


def tt_root(tree, eps=None):
    """Exact likelihood by tensor trains (tt/esco_tt.py) at tolerance eps (default TT_EPS)."""
    from tt import esco_tt
    r = esco_tt.loglik_detail(tree, eps=TT_EPS if eps is None else eps)
    rd = r.get("root_dist")
    if rd is not None:
        rd = np.maximum(np.array(rd), 0)
        rd = (rd / rd.sum()).tolist()
    return r["loglik"], rd


def fit(engine, model, tree_path, x0, fixed=None, pool=None, h=1e-5, log=None):
    """Maximize loglik. fixed = (index, logvalue) holds one log-parameter constant (profiles)."""
    n_par = npar(model)
    free = [i for i in range(n_par) if not fixed or i != fixed[0]]

    def full(z):
        th = np.zeros(n_par)
        th[free] = z
        if fixed:
            th[fixed[0]] = fixed[1]
        return th

    cache = {}
    nev = [0]

    def fg(z):
        key = tuple(np.round(z, 12))
        if key in cache:
            return cache[key]
        pts = [full(z)] + [full(z + h * np.eye(len(z))[k]) for k in range(len(z))]
        res = pool.map(evaluate, [(engine, model, p) for p in pts])
        nev[0] += len(pts)
        f0 = -res[0][0]
        g = np.array([(-r[0] - f0) / h for r in res[1:]])
        if not np.isfinite(f0):
            f0, g = 1e10, np.zeros(len(z))
        cache[key] = (f0, g)
        if log:
            print(f"  [{engine}/{model}] f={f0:.6f} |g|={np.linalg.norm(g):.2e} nev={nev[0]}", file=log, flush=True)
        return f0, g

    z0 = np.asarray(x0, float)[free]
    bounds = [(-12.0, 7.0)] * len(free)
    r = minimize(fg, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-5})
    th = full(r.x)
    ll, root = evaluate((engine, model, th))
    return {"engine": engine, "model": model, "loglik": ll, "theta": th.tolist(),
            "params": dict(zip(names(model), np.exp(th).tolist())), "root_probs": root,
            "grad_norm": float(np.linalg.norm(r.jac)), "nev": nev[0], "msg": str(r.message),
            "fixed": fixed}


def main():
    cmd, engine, model, tree_path, out = sys.argv[1:6]
    nproc = int(os.environ.get("NPROC", os.cpu_count()))
    t0 = time.time()
    with Pool(nproc, initializer=_init, initargs=(tree_path,)) as pool:
        _init(tree_path)
        n_par = npar(model)
        if cmd == "fit":
            if len(sys.argv) > 6:
                x0 = json.load(open(sys.argv[6]))["theta"]
                if len(x0) != n_par:  # sym -> asym start
                    M, Ne = unpack("sym", x0)
                    x0 = [math.log(M[a, b]) for a, b in OFF] + list(np.log(Ne))
            else:
                x0 = [0.0] * n_par
            res = fit(engine, model, tree_path, x0, pool=pool, log=sys.stderr)
        else:
            idx = int(sys.argv[6])
            x0 = json.load(open(sys.argv[7]))["theta"]
            grid = [float(v) for v in sys.argv[8].split(",")]
            res = {"engine": engine, "model": model, "param": names(model)[idx], "profile": []}
            start = list(x0)
            for v in grid:
                f = fit(engine, model, tree_path, start, fixed=(idx, math.log(v)), pool=pool, log=sys.stderr)
                res["profile"].append({"value": v, "loglik": f["loglik"], "params": f["params"],
                                       "root_probs": f["root_probs"]})
                start = f["theta"]
                print(f"profile {names(model)[idx]}={v}: {f['loglik']:.6f}", file=sys.stderr, flush=True)
                json.dump(res, open(out, "w"), indent=1)
    res["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
