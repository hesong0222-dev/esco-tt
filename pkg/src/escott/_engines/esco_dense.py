"""Exact structured-coalescent (ESCO) log-likelihood by the full joint location vector.

State between events: P[l_1, ..., l_n] = joint probability that the n live lineages sit in
demes l_1..l_n AND no unobserved coalescence happened, renormalized at every event (the
log normalizers are accumulated into log L).

Master equation (contract):
    dP/dt = sum_i (migration generator on axis i) P  -  D(l) P,
    D(l) = sum_{i<j} 1[l_i = l_j] / Ne[l_i] = sum_a C(n_a(l), 2) / Ne[a].
Convention: M[a][b] = backward rate a -> b for one lineage.

Two propagators (cross-checked against each other in dense/validate.py):
  * 'expm'    : explicit sparse generator + scipy.sparse.linalg.expm_multiply (small states)
  * 'unif'    : matrix-free uniformization (Poisson-weighted powers of the nonnegative
                matrix B = I + G/Lambda), substeps with renormalization; the tensor is never
                flattened into a matrix, so memory is ~6 state vectors.
'auto' uses expm for d^n <= 4096 and unif above.
"""
import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.stats import poisson


class Model:
    def __init__(self, M, Ne):
        self.M = np.array(M, dtype=float)
        self.d = self.M.shape[0]
        np.fill_diagonal(self.M, 0.0)
        self.Ne = np.array(Ne, dtype=float)
        self.inv_ne = 1.0 / self.Ne
        self.out = self.M.sum(axis=1)
        self.Q = self.M - np.diag(self.out)  # Q[a,b]: rate a->b, rows sum to 0
        self.R = self.M  # off-diagonal part


def _counts_D(n, mdl):
    """Return (D, E) on the n-axis grid: D = coalescence decay, E = total out-migration rate."""
    d = mdl.d
    shape = (d,) * n
    if n == 0:
        return np.zeros(()), np.zeros(())
    D = np.zeros(shape)
    E = np.zeros(shape)
    for a in range(d):
        cnt = np.zeros(shape, dtype=np.int16)
        for i in range(n):
            ind = np.zeros(d, dtype=np.int16)
            ind[a] = 1
            cnt += ind.reshape((1,) * i + (d,) + (1,) * (n - i - 1))
        D += cnt * (cnt - 1) / 2.0 * mdl.inv_ne[a]
        E += cnt * mdl.out[a]
    return D, E


def _sparse_generator(n, mdl, D):
    d = mdl.d
    N = d ** n
    G = sp.csr_matrix((N, N))
    QT = sp.csr_matrix(mdl.Q.T)
    for i in range(n):
        G = G + sp.kron(sp.kron(sp.identity(d ** i), QT), sp.identity(d ** (n - i - 1)), format="csr")
    G = G - sp.diags(D.ravel())
    return G.tocsr()


def _apply_offdiag(P, mdl, out, scale=1.0):
    """out += scale * sum_i (R^T on axis i) P   (R = off-diagonal migration rates)."""
    n = P.ndim
    d = mdl.d
    R = mdl.R * scale
    for i in range(n):
        A = d ** i
        B = d ** (n - i - 1)
        out.reshape(A, d, B)[...] += np.matmul(R.T, P.reshape(A, d, B))
    return out


def _poisson_weights(mu, tol):
    """Poisson(mu) pmf 0..m with remaining tail mass < tol (tail bounded geometrically)."""
    w = []
    m = 0
    while True:
        lp = m * math.log(mu) - mu - math.lgamma(m + 1) if mu > 0 else (0.0 if m == 0 else -math.inf)
        p = math.exp(lp)
        w.append(p)
        if m + 1 > mu:
            r = mu / (m + 2)  # ratio bound for p_{j+1}/p_j, j >= m+1; first tail term is p * mu/(m+1) (AUDIT B8)
            if p * mu / (m + 1) / (1 - r) < tol:
                break
        m += 1
    return np.array(w)


def propagate_unif(P, dt, mdl, D, E, rel_tol=1e-15):
    """Return (P_new normalized, log survival) after time dt, matrix-free uniformization."""
    if dt <= 0:
        return P, 0.0
    diag = -(E + D)
    Lam = float(-diag.min()) * 1.0001 + 1e-300
    Dmax = float(D.max())
    # substep h such that Lam*h <= 40 and Dmax*h <= 20
    nsub = max(1, int(math.ceil(max(Lam * dt / 40.0, Dmax * dt / 20.0))))
    h = dt / nsub
    w = 1.0 + diag / Lam  # >= 0
    logs = 0.0
    tmp = np.empty_like(P)
    for _ in range(nsub):
        mu = Lam * h
        # truncation: Poisson tail below rel_tol * (lower bound of survival exp(-Dmax h))
        tol = rel_tol * math.exp(-Dmax * h)
        weights = _poisson_weights(mu, tol)
        mmax = len(weights) - 1
        V = P.copy()
        acc = weights[0] * V
        for m in range(1, mmax + 1):
            np.multiply(V, w, out=tmp)
            _apply_offdiag(V, mdl, tmp, 1.0 / Lam)
            V, tmp = tmp, V
            acc += weights[m] * V
        s = acc.sum()
        logs += math.log(s)
        P = acc / s
    return P, logs


EXPM_MAX_DECAY = 600.0  # max Dmax*h per expm substep: survival >= exp(-600) ~ 1e-261 cannot underflow (AUDIT B3)


def propagate_expm(P, dt, mdl, D):
    if dt <= 0:
        return P, 0.0
    n = P.ndim
    if n == 1:
        v = expm(mdl.Q.T * dt) @ P  # n==1: no coalescence decay
        v = np.maximum(v, 0.0)
        s = v.sum()
        return v / s, math.log(s)
    G = _sparse_generator(n, mdl, D)
    # one expm_multiply call as before unless the interval could underflow; then substeps with renormalisation
    nsub = max(1, int(math.ceil(float(D.max()) * dt / EXPM_MAX_DECAY)))
    h = dt / nsub
    logs = 0.0
    for _ in range(nsub):
        v = expm_multiply(G * h, P.ravel()).reshape(P.shape)
        v = np.maximum(v, 0.0)
        s = v.sum()
        if not (math.isfinite(s) and s > 0):
            raise FloatingPointError(f"dense expm: survival mass became {s!r}")
        P = v / s
        logs += math.log(s)
    return P, logs


def loglik(tree, method="auto", return_info=False):
    if isinstance(tree, str):
        with open(tree) as f:
            tree = json.load(f)
    mdl = Model(tree["M"], tree["Ne"])
    d = mdl.d
    P = np.ones(())
    ids = []
    t = None
    logL = 0.0
    maxn = 0
    D, E = _counts_D(0, mdl)
    for ev in tree["events"]:
        if t is not None and ev["t"] > t and len(ids) > 0:
            dt = ev["t"] - t
            n = len(ids)
            use = method
            if method == "auto":
                use = "expm" if d ** n <= 4096 else "unif"
            if n == 1:
                P = expm(mdl.Q.T * dt) @ P
                P = P / P.sum()
            elif use == "expm":
                P, ls = propagate_expm(P, dt, mdl, D)
                logL += ls
            else:
                P, ls = propagate_unif(P, dt, mdl, D, E)
                logL += ls
        t = ev["t"]
        if ev["type"] == "sample":
            e = np.zeros(d)
            e[ev["deme"]] = 1.0
            P = np.multiply.outer(P, e)
            ids.append(ev["id"])
        elif ev["type"] == "coal":
            i = ids.index(ev["a"])
            j = ids.index(ev["b"])
            Pd = np.diagonal(P, axis1=i, axis2=j) * mdl.inv_ne  # merged axis last
            P = np.ascontiguousarray(Pd)
            ids = [x for x in ids if x not in (ev["a"], ev["b"])] + [ev["new"]]
            s = P.sum()
            if s <= 0:
                return -math.inf
            logL += math.log(s)
            P = P / s
        else:
            raise ValueError(ev["type"])
        maxn = max(maxn, len(ids))
        D, E = _counts_D(len(ids), mdl)
    if return_info:
        return logL, {"max_lineages": maxn, "max_states": d ** maxn}
    return logL


def max_lineages(tree):
    n = m = 0
    for ev in tree["events"]:
        n += 1 if ev["type"] == "sample" else -1
        m = max(m, n)
    return m


if __name__ == "__main__":
    for path in sys.argv[1:]:
        t0 = time.time()
        ll, info = loglik(path, return_info=True)
        print(f"{path}\t{ll:.12f}\t{info['max_lineages']}\t{time.time() - t0:.3f}s")
