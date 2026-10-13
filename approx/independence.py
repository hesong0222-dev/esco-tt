"""SISCO and LISCO (Mueller, Rasmussen & Stadler 2017, MBE 34(11):2970), on the same footing as approx/mascot.py.

All three closures carry one normalised d-vector p_i per lineage and share the event rules
  interval:     dlog P(T)/dt = - sum_a lam_a sum_{i<j} p_i(a) p_j(a)
  coalescence:  P(T) *= sum_a lam_a p_i(a) p_j(a);  p_parent(a) proportional to lam_a p_i(a) p_j(a)
  sampling:     new lineage with p = delta(deme)
They differ only in how the no-coalescence conditioning enters the lineage ODE:
  dp_i(a)/dt = sum_b (m_ba p_i(b) - m_ab p_i(a))
               + kappa * [ p_i(a) sum_b lam_b p_i(b) o_i(b) - lam_a p_i(a) o_i(a) ],   o_i(a) = sum_{k!=i} p_k(a)
  kappa = 1   MASCO/MASCOT (approx/mascot.py; reproduced here to 1e-12, see bench/sisco.md)
  kappa = 1/2 LISCO  -- IndependentStructuredCoalescent.java + independent_ode_integrator.java of
                        github.com/nicfel/The-Structured-Coalescent @81b647f: unnormalised q_i decays as
                        dq_i(a) = mig(q_i) - (lam_a/2) q_i(a) o_i(a)  (o from normalised p), and at every
                        coalescence logL += sum_i log(sum_a q_i(a)) before renormalising.  Writing q_i = w_i p_i
                        gives exactly the kappa = 1/2 ODE above, and sum_i dlog w_i/dt equals the interval rate.
  kappa = 0   SISCO  -- paper eq. 4 ("only dependent on the migration rates"); ApproximateStructuredCoalescent.java
                        (same repo) propagates p by the migration matrix only.
lam_a = 1/Ne[a];  m_ab = M[a][b] (backward a->b), as in sim/, dense/, approx/mascot.py.
"""
import json
import math
import sys

import numpy as np
from scipy.integrate import solve_ivp

KAPPA = {"mascot": 1.0, "lisco": 0.5, "sisco": 0.0}


def _rhs(t, y, n, d, M, out, lam, kappa):
    p = y[:-1].reshape(n, d)
    S = p.sum(axis=0)
    others = S[None, :] - p
    mig = p @ M - p * out[None, :]          # sum_b p_i(b) m_ba - p_i(a) sum_b m_ab
    dp = mig
    if kappa:
        rate_ia = lam[None, :] * others
        lam_i = (p * rate_ia).sum(axis=1, keepdims=True)
        dp = mig + kappa * (p * lam_i - p * rate_ia)
    dlog = -0.5 * (lam * (S * S - (p * p).sum(axis=0))).sum()
    return np.concatenate([dp.ravel(), [dlog]])


def loglik(tree, method="sisco", rtol=1e-10, atol=1e-12, return_root=False):
    kappa = KAPPA[method] if isinstance(method, str) else float(method)
    if isinstance(tree, str):
        with open(tree) as f:
            tree = json.load(f)
    M = np.array(tree["M"], dtype=float)
    np.fill_diagonal(M, 0.0)
    d = M.shape[0]
    lam = 1.0 / np.array(tree["Ne"], dtype=float)
    out = M.sum(axis=1)
    ids, p = [], np.zeros((0, d))
    logL, t = 0.0, None
    for ev in tree["events"]:
        if t is not None and ev["t"] > t and len(ids) > 0:
            n = len(ids)
            y0 = np.concatenate([p.ravel(), [0.0]])
            sol = solve_ivp(_rhs, (t, ev["t"]), y0, method="DOP853", rtol=rtol, atol=atol,
                            args=(n, d, M, out, lam, kappa))
            if not sol.success:
                raise RuntimeError(f"ODE failed on ({t}, {ev['t']}): {sol.message}")
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
                return (-math.inf, None) if return_root else -math.inf
            logL += math.log(s)
            keep = [x for x in range(len(ids)) if x not in (i, j)]
            p = np.vstack([p[keep], w / s])
            ids = [ids[x] for x in keep] + [ev["new"]]
    return (logL, p[-1].copy()) if return_root else logL


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path, *(f"{m} {loglik(path, m):.10f}" for m in ("sisco", "lisco", "mascot")), sep="\t")
