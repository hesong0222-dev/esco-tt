"""MLE study: MASCOT vs exact (TT) structured coalescent — shared pieces.

Model fitted (d+1 parameters, all in log space):
    M[a][b] = m for all a != b  (one backward migration rate),   Ne[a] free per deme.
    theta = (log m, log Ne_0, ..., log Ne_{d-1})
Engines: 'mascot' (approx/mascot.py ODEs) and 'tt' (tt/esco_tt.py, exact up to eps).
Both return (loglik, root-deme distribution).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
from scipy.integrate import solve_ivp

from approx import mascot
from tt import esco_tt
from sim import simulate as S

# ----------------------------------------------------------------------------- settings
NE = {3: [1.0, 1.5, 2.0], 4: [1.0, 1.5, 2.0, 2.5]}
FACTOR = {"slow": 0.1, "medium": 1.0}


def true_params(d, regime):
    Ne = NE[d]
    cbar = float(np.mean(1.0 / np.array(Ne)))
    return FACTOR[regime] * cbar, Ne


def true_theta(d, regime):
    m, Ne = true_params(d, regime)
    return [math.log(m)] + [math.log(x) for x in Ne]


def make_tree(d, k, regime, rep):
    seed = 20261006 + 100000 * d + 1000 * k + 10 * list(FACTOR).index(regime) + rep
    rng = np.random.default_rng(seed)
    m, Ne = true_params(d, regime)
    M = (m * (1 - np.eye(d))).tolist()
    times, demes = S.make_samples(k, d, rng)
    ev = S.simulate_tree(M, Ne, times, demes, rng)
    tr = {"demes": d, "M": M, "Ne": Ne, "events": ev,
          "meta": {"d": d, "k": k, "regime": regime, "rep": rep, "seed": seed}}
    S.validate(tr)
    return tr


def with_theta(tree, theta):
    d = tree["demes"]
    v = np.exp(np.asarray(theta, float))
    M = (v[0] * (1 - np.eye(d))).tolist()
    return dict(tree, M=M, Ne=v[1:].tolist())


def names(d):
    return ["m"] + [f"Ne{a}" for a in range(d)]


# ----------------------------------------------------------------------------- engines
def mascot_root(tree):
    """approx/mascot.loglik loop, also returning the root-lineage deme probabilities."""
    M = np.array(tree["M"], float)
    np.fill_diagonal(M, 0.0)
    D = M.shape[0]
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


EPS = float(os.environ.get("MLE_EPS", "1e-6"))


def tt_root(tree, eps=None):
    r = esco_tt.loglik_detail(tree, eps=EPS if eps is None else eps)
    rd = r.get("root_dist")
    if rd is not None:
        rd = np.maximum(np.array(rd), 0)
        rd = (rd / rd.sum()).tolist()
    return r["loglik"], rd


def evaluate(engine, tree, theta, eps=None):
    tr = with_theta(tree, theta)
    if engine == "mascot":
        return mascot_root(tr)
    return tt_root(tr, eps)
