"""Shared pieces of the MTT comparison (bench/mtt_compare.md).

Parameter vector theta (log scale): [log Ne_0..Ne_{d-1}, log M_ab for a != b in a-major order]
(the same order as MTT's off-diagonal rateMatrix of dimension d(d-1), see bench/mtt_compare.md).
Priors (identical in MTT, exact and MASCOT runs): Ne_a ~ LogNormal(NE_MU, NE_S), M_ab ~ LogNormal(M_MU, M_S),
meanlog/sdlog in log space (BEAST LogNormalDistributionModel with meanInRealSpace=false).
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ESCO = os.path.abspath(os.path.join(HERE, "..", ".."))
if ESCO not in sys.path:
    sys.path.insert(0, ESCO)

NE_MU, NE_S = 0.0, 1.0
M_MU, M_S = -1.0, 1.5
TREES = ["d2_k16_slow_r1", "d2_k16_medium_r0", "d3_k10_slow_r1", "d3_k10_medium_r2"]


def load_tree(name, data_dir=None):
    data_dir = data_dir or os.path.join(ESCO, "sim", "data")
    with open(os.path.join(data_dir, name + ".json")) as f:
        return json.load(f)


def offdiag(d):
    return [(a, b) for a in range(d) for b in range(d) if a != b]


def param_names(d):
    return [f"Ne{a}" for a in range(d)] + [f"M{a}{b}" for a, b in offdiag(d)]


def theta_true(tree):
    d = len(tree["Ne"])
    return np.log(np.array(list(tree["Ne"]) + [tree["M"][a][b] for a, b in offdiag(d)], float))


def with_params(tree, theta):
    d = len(tree["Ne"])
    x = np.exp(theta)
    M = [[0.0] * d for _ in range(d)]
    for (a, b), v in zip(offdiag(d), x[d:]):
        M[a][b] = float(v)
    return {"M": M, "Ne": [float(v) for v in x[:d]], "events": tree["events"]}


def log_prior(theta, d):
    """Density of theta = log(x) where x ~ LogNormal: i.e. Normal(mu, s) on theta (Jacobian included)."""
    mu = np.r_[np.full(d, NE_MU), np.full(d * (d - 1), M_MU)]
    s = np.r_[np.full(d, NE_S), np.full(d * (d - 1), M_S)]
    z = (theta - mu) / s
    return float(-0.5 * (z @ z) - np.log(s).sum() - 0.5 * len(theta) * math.log(2 * math.pi))


def make_loglik(tree, engine):
    if engine == "exact":
        from dense.esco_dense import loglik as f
    elif engine == "mascot":
        from approx.mascot import loglik as f
    else:
        raise ValueError(engine)

    def ll(theta):
        if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > 12):
            return -math.inf
        try:
            v = f(with_params(tree, theta))
        except (FloatingPointError, RuntimeError, ValueError, ZeroDivisionError):
            return -math.inf
        return v if math.isfinite(v) else -math.inf
    return ll


def adaptive_metropolis(logpost, theta0, n_iter=None, seconds=None, adapt_frac=0.3, seed=0, cov0=None,
                        thin=1, progress=None):
    """Haario-style adaptive random-walk Metropolis in theta space.
    Covariance adapts only during the first adapt_frac of the run (iterations, or wall time if seconds is set);
    samples after that come from a fixed kernel. Returns dict(samples, logpost, accept_after_adapt, n_iter, adapt_end, seconds)."""
    rng = np.random.default_rng(seed)
    p = len(theta0)
    sd = 2.38 ** 2 / p
    cov = np.diag(np.full(p, 0.1 ** 2)) if cov0 is None else np.array(cov0, float)
    x = np.array(theta0, float)
    lx = logpost(x)
    if not math.isfinite(lx):
        raise ValueError("non-finite start")
    out, lps = [], []
    mean, m2, nstat = x.copy(), np.zeros((p, p)), 1
    t0 = time.time()
    it = acc = acc_n = 0
    adapting = True
    adapt_end = None
    while True:
        el = time.time() - t0
        if n_iter is not None and it >= n_iter:
            break
        if seconds is not None and el >= seconds:
            break
        if adapting:
            done = (it >= adapt_frac * n_iter) if n_iter is not None else (el >= adapt_frac * seconds)
            if done:
                adapting = False
                adapt_end = it
        try:
            L = np.linalg.cholesky(sd * cov + 1e-10 * np.eye(p))
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(sd * np.diag(np.diag(cov)) + 1e-10 * np.eye(p))
        y = x + L @ rng.standard_normal(p)
        ly = logpost(y)
        a = math.log(rng.random()) < ly - lx
        if a:
            x, lx = y, ly
        if not adapting:
            acc += a
            acc_n += 1
        if adapting:
            nstat += 1
            dlt = x - mean
            mean += dlt / nstat
            m2 += np.outer(dlt, x - mean)
            if nstat > 2 * p and it % 10 == 0:
                cov = m2 / (nstat - 1)
        it += 1
        if it % thin == 0:
            out.append(x.copy())
            lps.append(lx)
        if progress and it % progress == 0:
            print(f"  it {it} lp {lx:.3f} el {time.time()-t0:.0f}s", flush=True)
    return {"samples": np.array(out), "logpost": np.array(lps), "accept_after_adapt": acc / max(acc_n, 1),
            "n_iter": it, "adapt_end": adapt_end if adapt_end is not None else it, "seconds": time.time() - t0,
            "thin": thin, "cov": cov}


def ess(x):
    """Geyer initial-monotone-sequence ESS of a 1-d chain."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or np.var(x) == 0:
        return float("nan")
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / (n * np.var(x))
    s, prev = 0.0, math.inf
    for k in range(0, n - 1, 2):
        g = ac[k] + ac[k + 1]
        if g <= 0:
            break
        g = min(g, prev)
        prev = g
        s += g
    return float(n / max(2 * s - 1, 1e-12))


def rhat(chains):
    """Split-R-hat over a list of 1-d chains (Gelman et al. BDA3)."""
    halves = []
    for c in chains:
        c = np.asarray(c, float)
        h = len(c) // 2
        halves += [c[:h], c[h:2 * h]]
    m = min(len(h) for h in halves)
    a = np.array([h[:m] for h in halves])
    W = a.var(axis=1, ddof=1).mean()
    B = m * a.mean(axis=1).var(ddof=1)
    return float(math.sqrt(((m - 1) / m * W + B / m) / W))


def wquantile(x, w, q):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w) / w.sum()
    return np.interp(q, c - 0.5 * w / w.sum(), x)


def ks_weighted(x, wx, y, wy=None):
    """Two-sample KS distance between weighted ECDFs."""
    wy = np.ones(len(y)) if wy is None else wy
    grid = np.sort(np.r_[x, y])
    def F(v, w):
        o = np.argsort(v)
        c = np.cumsum(np.asarray(w)[o]) / np.sum(w)
        return np.r_[0.0, c][np.searchsorted(np.asarray(v)[o], grid, side="right")]
    return float(np.max(np.abs(F(x, wx) - F(y, wy))))


def overlap(x, wx, y, wy=None, bins=60):
    """Overlap coefficient (sum of min of the two histogram densities) on a common grid."""
    wy = np.ones(len(y)) if wy is None else wy
    lo, hi = np.quantile(np.r_[x, y], [0.001, 0.999])
    e = np.linspace(lo, hi, bins + 1)
    hx, _ = np.histogram(np.clip(x, lo, hi), e, weights=wx)
    hy, _ = np.histogram(np.clip(y, lo, hi), e, weights=wy)
    return float(np.minimum(hx / hx.sum(), hy / hy.sum()).sum())


def summarize(samples, w=None):
    """Per column: mean, 2.5%, 97.5% (on the natural scale, exp(theta))."""
    X = np.exp(samples)
    w = np.ones(len(X)) if w is None else np.asarray(w)
    res = []
    for j in range(X.shape[1]):
        res.append((float(np.sum(w * X[:, j]) / w.sum()), float(wquantile(X[:, j], w, 0.025)),
                    float(wquantile(X[:, j], w, 0.975))))
    return res
