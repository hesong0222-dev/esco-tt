"""Library API: exact structured-coalescent log-likelihood (tensor train or full vector) and the MASCOT approximation."""
import json
import math
import time
from collections import namedtuple

import numpy as np

from ._engines import esco_dense, esco_tt, mascot

Result = namedtuple("Result", "loglik error method info")
Result.__doc__ = """loglik: log-likelihood of the fixed time tree.
error: estimated numerical error in log units (tt: max(paired difference, empirical bound 0.81*eps*N_round), see
loglik; dense: 0.0, exact up to floating point; mascot: None, because MASCOT is an approximation whose model error is
unknown without an exact value).
method: 'tt', 'dense' or 'mascot'.  info: engine details (ranks, paired value, run time, root distribution, ...)."""

DENSE_MAX_STATES = 3 ** 14  # full vector of d^k doubles (~40 MB each); above this 'auto' uses the tensor train
ERROR_C = 0.81              # empirical constant of |logL_TT - logL_exact| <= c * eps * N_round (bench/robustness.md)


def max_lineages(tree):
    return esco_dense.max_lineages(tree)


def exact_method(tree):
    """'dense' if the full vector of d^k states is small enough (k = most lineages present at once), else 'tt'."""
    d = len(tree["Ne"]) if "Ne" in tree else len(tree.get("deme_names") or []) or tree["demes"]
    return "dense" if d ** max_lineages(tree) <= DENSE_MAX_STATES else "tt"


def _as_tree(tree, migration=None, Ne=None):
    if isinstance(tree, str):
        if tree.lstrip().startswith("{"):
            tree = json.loads(tree)
        else:
            with open(tree) as f:
                tree = json.load(f)
    tree = dict(tree)
    if migration is not None:
        tree["M"] = [list(map(float, r)) for r in np.asarray(migration, float)]
    if Ne is not None:
        tree["Ne"] = [float(x) for x in np.atleast_1d(np.asarray(Ne, float))]
    if "M" not in tree or "Ne" not in tree:
        raise ValueError("migration matrix and Ne are required (pass them or include 'M' and 'Ne' in the tree dict)")
    d = len(tree["Ne"])
    if np.shape(tree["M"]) != (d, d):
        raise ValueError(f"migration must be {d}x{d} to match Ne")
    if any(not (x > 0) for x in tree["Ne"]):
        raise ValueError(f"every Ne must be positive, got {tree['Ne']}")
    tree["demes"] = d
    tree["events"] = sorted(tree["events"], key=lambda e: (e["t"], 0 if e["type"] == "sample" else 1))
    return tree


def _normalised(p):
    p = np.maximum(np.asarray(p, float), 0.0)
    return (p / p.sum()).tolist()


def dense_root(tree):
    """Full-vector exact log-likelihood and the deme distribution of the root lineage.
    Same arithmetic as the engine's esco_dense.loglik(method='auto'); it also keeps the final state vector."""
    from scipy.linalg import expm
    mdl = esco_dense.Model(tree["M"], tree["Ne"])
    d = mdl.d
    P = np.ones(())
    ids, t, logL = [], None, 0.0
    D, E = esco_dense._counts_D(0, mdl)
    for ev in tree["events"]:
        if t is not None and ev["t"] > t and ids:
            dt, n = ev["t"] - t, len(ids)
            if n == 1:
                P = expm(mdl.Q.T * dt) @ P
                P = P / P.sum()
            elif d ** n <= 4096:
                P, ls = esco_dense.propagate_expm(P, dt, mdl, D)
                logL += ls
            else:
                P, ls = esco_dense.propagate_unif(P, dt, mdl, D, E)
                logL += ls
        t = ev["t"]
        if ev["type"] == "sample":
            e = np.zeros(d)
            e[ev["deme"]] = 1.0
            P = np.multiply.outer(P, e)
            ids.append(ev["id"])
        elif ev["type"] == "coal":
            i, j = ids.index(ev["a"]), ids.index(ev["b"])
            P = np.ascontiguousarray(np.diagonal(P, axis1=i, axis2=j) * mdl.inv_ne)
            ids = [x for x in ids if x not in (ev["a"], ev["b"])] + [ev["new"]]
            s = P.sum()
            if s <= 0:
                return -math.inf, None
            logL += math.log(s)
            P = P / s
        else:
            raise ValueError(ev["type"])
        D, E = esco_dense._counts_D(len(ids), mdl)
    return logL, (_normalised(P.ravel()) if P.ndim == 1 else None)


MASCOT_MAX_RHS = 200_000   # ODE right-hand-side calls per evaluation before giving up (stiff, extreme parameters)


def mascot_root(tree, rtol=1e-10, atol=1e-12):
    """MASCOT log-likelihood and MASCOT's deme probabilities of the root lineage.
    Same arithmetic as the engine's mascot.loglik; it also keeps the final lineage-state probabilities.
    Raises RuntimeError if the ODE solve fails, produces non-finite values or needs more than MASCOT_MAX_RHS
    right-hand-side evaluations (this happens only at extreme parameter values, e.g. Ne near 0)."""
    from scipy.integrate import solve_ivp
    calls = [0]

    def rhs(t, y, *args):
        calls[0] += 1
        if calls[0] > MASCOT_MAX_RHS or not np.all(np.isfinite(y)):
            raise RuntimeError("MASCOT ODE is too stiff or diverged at these parameter values")
        return mascot._rhs(t, y, *args)

    M = np.array(tree["M"], dtype=float)
    np.fill_diagonal(M, 0.0)
    d = M.shape[0]
    lam = 1.0 / np.array(tree["Ne"], dtype=float)
    out = M.sum(axis=1)
    ids, p, logL, t = [], np.zeros((0, d)), 0.0, None
    for ev in tree["events"]:
        if t is not None and ev["t"] > t and ids:
            n = len(ids)
            with np.errstate(over="ignore", invalid="ignore"):
                sol = solve_ivp(rhs, (t, ev["t"]), np.concatenate([p.ravel(), [0.0]]), method="DOP853",
                                rtol=rtol, atol=atol, args=(n, d, M.T, out, lam))
            if not sol.success:
                raise RuntimeError(f"MASCOT ODE failed on ({t}, {ev['t']}): {sol.message}")
            y = sol.y[:, -1]
            p = np.maximum(y[:-1].reshape(n, d), 0.0)
            p /= p.sum(axis=1, keepdims=True)
            logL += y[-1]
        t = ev["t"]
        if ev["type"] == "sample":
            row = np.zeros(d)
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
    return logL, (p[-1].tolist() if len(ids) == 1 else None)


def loglik(tree, migration=None, Ne=None, method="tt", eps=1e-6, paired=True, **tt_options):
    """Log-likelihood of a fixed time tree under the structured coalescent.

    tree: escott tree dict (see escott.io), a JSON string or a path to a JSON tree; build one from a BEAST
      NEXUS/Newick tree with escott.io.load_tree.
    migration: d x d backward migration rates, migration[a][b] = rate a -> b for one lineage (BEAST 'b_migration').
    Ne: length-d effective population sizes; two lineages in deme a coalesce at rate 1/Ne[a].
    method: 'tt' (exact model, tensor train), 'dense' (exact model, full d^k vector; small trees only),
      'auto' (dense if d^k <= DENSE_MAX_STATES else tt), or 'mascot' (MASCOT approximation, Mueller et al. 2018).
    eps: tensor-train rounding tolerance (tt only).
    paired: tt only. True (default) also runs the paired setting (eps/10, half time step) and sets
      error = max(paired_diff, error_bound_empirical); False runs once and error = error_bound_empirical.
      paired_diff = |logL(eps, h) - logL(eps/10, h/2)| (systematically too small on its own: covered the actual
      error on 1 of 18 small trees, review/AUDIT_CODE.md B2); error_bound_empirical = 0.81 * eps * N_round, N_round
      = number of TT roundings, from the fit |logL_TT - logL_exact| <= c*eps*N_round with |c| <= 0.81 on 96
      (tree, eps) pairs with an exact reference (bench/robustness.md).  Empirical, not a guaranteed bound.
    tt_options: passed to the tensor-train engine (e.g. rmax).  info['rmax_cuts'] > 0 means rmax, not eps, set some
      ranks; the error estimate then does not apply.
    info['root_dist'] holds the deme distribution of the root lineage for every method.
    """
    tree = _as_tree(tree, migration, Ne)
    if method == "auto":
        method = exact_method(tree)
    t0 = time.time()
    if method == "tt":
        r1 = esco_tt.loglik_detail(tree, eps=eps, **tt_options)
        bound = ERROR_C * r1["eps"] * r1["n_round"]
        root = _normalised(r1["root_dist"]) if r1.get("root_dist") is not None else None
        info = dict(eps=eps, max_rank=r1["max_rank"], runtime=r1["runtime"], error_bound_empirical=bound,
                    n_round=r1["n_round"], rmax_cuts=r1["rmax_cuts"], root_dist=root)
        if not paired:
            return Result(r1["loglik"], bound, "tt", info)
        r2 = esco_tt.loglik_detail(tree, eps=eps / 10, steps_mult=2, **tt_options)
        paired_diff = abs(r1["loglik"] - r2["loglik"])
        info.update(runtime=r1["runtime"] + r2["runtime"], paired_loglik=r2["loglik"], paired_setting="eps/10, h/2",
                    max_rank_paired=r2["max_rank"], paired_diff=paired_diff,
                    rmax_cuts=r1["rmax_cuts"] + r2["rmax_cuts"])
        return Result(r1["loglik"], max(paired_diff, bound), "tt", info)
    if method == "dense":
        ll, root = dense_root(tree)
        k = max_lineages(tree)
        return Result(ll, 0.0, "dense", {"max_lineages": k, "max_states": tree["demes"] ** k, "root_dist": root,
                                         "runtime": time.time() - t0})
    if method == "mascot":
        ll, root = mascot_root(tree)
        return Result(ll, None, "mascot", {"ode": "DOP853 rtol 1e-10", "root_dist": root, "runtime": time.time() - t0})
    raise ValueError(f"unknown method {method!r}")
