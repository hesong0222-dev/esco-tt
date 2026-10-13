"""Validation of dense/esco_dense.py.

(a) d=1: Kingman heterochronous coalescent closed form.
(b) k=2, d=2 symmetric: analytic 2-state (same/different deme) solution.
(c) Independent Monte Carlo over migration histories (Gillespie, no master equation).
(d) expm_multiply vs uniformization on the test set (two independent propagators).
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esco_dense as ed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "sim", "data")


def kingman_loglik(tree):
    Ne = tree["Ne"][0]
    n, t, ll = 0, None, 0.0
    for ev in tree["events"]:
        if t is not None:
            ll -= n * (n - 1) / 2 / Ne * (ev["t"] - t)
        t = ev["t"]
        if ev["type"] == "sample":
            n += 1
        else:
            ll += math.log(1 / Ne)
            n -= 1
    return ll


def test_a():
    sys.path.insert(0, os.path.join(HERE, "..", "sim"))
    import simulate
    rng = np.random.default_rng(1)
    worst = 0.0
    for k in (2, 5, 10, 20):
        times, demes = simulate.make_samples(k, 1, rng)
        ev = simulate.simulate_tree([[0.0]], [1.7], times, demes, rng)
        tree = {"demes": 1, "M": [[0.0]], "Ne": [1.7], "events": ev}
        for m in ("expm", "unif"):
            err = abs(ed.loglik(tree, method=m) - kingman_loglik(tree))
            worst = max(worst, err)
    # d=2 with zero migration and all samples in deme 0 must also equal Kingman with Ne[0]
    times, demes = simulate.make_samples(8, 1, rng)
    ev = simulate.simulate_tree([[0.0]], [1.3], times, demes, rng)
    tree2 = {"demes": 2, "M": [[0, 0], [0, 0]], "Ne": [1.3, 5.0], "events": ev}
    err = abs(ed.loglik(tree2) - kingman_loglik({"Ne": [1.3], "events": ev}))
    worst = max(worst, err)
    return worst


def test_b():
    worst = 0.0
    for (m, Ne, T, same) in [(0.3, 1.0, 0.7, True), (0.3, 1.0, 0.7, False), (5.0, 2.0, 1.9, False),
                             (0.01, 0.5, 3.0, True), (1.0, 1.0, 0.05, False)]:
        # states (same, diff): ds = -2m s + 2m f - s/Ne ; df = 2m s - 2m f ; L = s(T)/Ne
        A = np.array([[-2 * m - 1 / Ne, 2 * m], [2 * m, -2 * m]])
        x0 = np.array([1.0, 0.0]) if same else np.array([0.0, 1.0])
        # closed-form 2x2 exponential via eigen-decomposition (symmetric A)
        w, V = np.linalg.eigh(A)
        x = V @ (np.exp(w * T) * (V.T @ x0))
        exact = math.log(x[0] / Ne)
        tree = {"demes": 2, "M": [[0, m], [m, 0]], "Ne": [Ne, Ne], "events": [
            {"t": 0.0, "type": "sample", "id": 0, "deme": 0},
            {"t": 0.0, "type": "sample", "id": 1, "deme": 0 if same else 1},
            {"t": T, "type": "coal", "a": 0, "b": 1, "new": 2}]}
        for meth in ("expm", "unif"):
            worst = max(worst, abs(ed.loglik(tree, method=meth) - exact))
    # also an asymmetric, unequal-Ne d=3, k=2 case against a direct 9x9 expm
    M = np.array([[0, .4, .9], [.2, 0, .1], [1.3, .6, 0]])
    Ne = np.array([1.0, 2.0, 0.7])
    Q = M - np.diag(M.sum(1))
    G = np.kron(Q.T, np.eye(3)) + np.kron(np.eye(3), Q.T)
    Dg = np.zeros(9)
    for a in range(3):
        Dg[a * 3 + a] = 1 / Ne[a]
    G -= np.diag(Dg)
    x0 = np.zeros(9)
    x0[0 * 3 + 2] = 1
    T = 1.234
    x = expm(G * T) @ x0
    exact = math.log(sum(x[a * 3 + a] / Ne[a] for a in range(3)))
    tree = {"demes": 3, "M": M.tolist(), "Ne": Ne.tolist(), "events": [
        {"t": 0.0, "type": "sample", "id": 0, "deme": 0},
        {"t": 0.0, "type": "sample", "id": 1, "deme": 2},
        {"t": T, "type": "coal", "a": 0, "b": 1, "new": 2}]}
    worst = max(worst, abs(ed.loglik(tree) - exact))
    return worst


def mc_loglik(tree, nrep, rng):
    """Monte Carlo: simulate lineage migrations only (rates M, no coalescence) through the fixed
    tree; weight = prod exp(-int D dt) * prod_coal 1[same deme]/Ne. L = E[weight].
    Independent of the master equation / matrix exponentials."""
    M = np.array(tree["M"], float)
    np.fill_diagonal(M, 0)
    d = M.shape[0]
    out = M.sum(1)
    inv = 1 / np.array(tree["Ne"], float)
    events = tree["events"]
    W = np.empty(nrep)
    for r in range(nrep):
        loc = {}
        logw = 0.0
        t = 0.0
        dead = False
        for ev in events:
            # evolve lineages from t to ev.t: Gillespie on migrations, integrate D exactly
            while True:
                ids = list(loc)
                rates = np.array([out[loc[i]] for i in ids]) if ids else np.zeros(0)
                tot = rates.sum()
                dt = rng.exponential(1 / tot) if tot > 0 else np.inf
                cnt = np.bincount(list(loc.values()), minlength=d) if ids else np.zeros(d)
                Dv = float((cnt * (cnt - 1) / 2 * inv).sum())
                if t + dt >= ev["t"]:
                    logw -= Dv * (ev["t"] - t)
                    t = ev["t"]
                    break
                logw -= Dv * dt
                t += dt
                i = ids[rng.choice(len(ids), p=rates / tot)]
                a = loc[i]
                loc[i] = int(rng.choice(d, p=M[a] / out[a]))
            if ev["type"] == "sample":
                loc[ev["id"]] = ev["deme"]
            else:
                la, lb = loc.pop(ev["a"]), loc.pop(ev["b"])
                if la != lb:
                    dead = True
                    break
                logw += math.log(inv[la])
                loc[ev["new"]] = la
        W[r] = 0.0 if dead else math.exp(logw)
    mean = W.mean()
    se = W.std(ddof=1) / math.sqrt(nrep)
    return mean, se


def mc_loglik_vec(tree, N, rng, chunk=1_000_000):
    """Vectorized version of mc_loglik (same estimator, numpy over paths): migrations by
    thinning a Poisson clock of rate n*Lam_max per path, D integrated exactly between jumps."""
    M = np.array(tree["M"], float)
    np.fill_diagonal(M, 0)
    d = M.shape[0]
    out = M.sum(1)
    Lmax = out.max()
    cum = np.cumsum(M / np.where(out > 0, out, 1)[:, None], axis=1)
    inv = 1 / np.array(tree["Ne"], float)
    events = tree["events"]
    nid = max(max(e.get("id", -1), e.get("new", -1)) for e in events) + 1
    tot = []
    for c0 in range(0, N, chunk):
        n_ = min(chunk, N - c0)
        loc = np.full((n_, nid), -1, dtype=np.int64)
        logw = np.zeros(n_)
        alive = np.ones(n_, bool)
        live = []
        t = 0.0
        for ev in events:
            T = ev["t"]
            if T > t and live:
                L = np.array(live)
                k = len(L)
                tc = np.full(n_, t)
                act = alive.copy()
                while act.any():
                    idx = np.nonzero(act)[0]
                    sub = loc[np.ix_(idx, L)]
                    cnt = np.stack([(sub == a).sum(1) for a in range(d)], 1)
                    Dv = (cnt * (cnt - 1) / 2 * inv).sum(1)
                    dt = rng.exponential(1 / (k * Lmax), size=len(idx)) if Lmax > 0 else np.full(len(idx), np.inf)
                    fin = tc[idx] + dt >= T
                    step = np.where(fin, T - tc[idx], dt)
                    logw[idx] -= Dv * step
                    tc[idx] += step
                    act[idx[fin]] = False
                    j = idx[~fin]
                    if len(j):
                        lin = L[rng.integers(0, k, size=len(j))]
                        cur = loc[j, lin]
                        acc = rng.uniform(size=len(j)) * Lmax < out[cur]
                        j, lin, cur = j[acc], lin[acc], cur[acc]
                        u = rng.uniform(size=len(j))
                        new = (u[:, None] > cum[cur]).sum(1)
                        loc[j, lin] = np.minimum(new, d - 1)
            t = T
            if ev["type"] == "sample":
                loc[:, ev["id"]] = ev["deme"]
                live.append(ev["id"])
            else:
                la, lb = loc[:, ev["a"]], loc[:, ev["b"]]
                alive &= la == lb
                logw += np.log(inv[la])
                loc[:, ev["new"]] = la
                live = [x for x in live if x not in (ev["a"], ev["b"])] + [ev["new"]]
        tot.append(np.where(alive, np.exp(logw), 0.0))
    W = np.concatenate(tot)
    return W.mean(), W.std(ddof=1) / math.sqrt(N)


def test_c(nrep=40000):
    rng = np.random.default_rng(7)
    rows = []
    sys.path.insert(0, os.path.join(HERE, "..", "sim"))
    files = ["d2_k04_slow_r0", "d2_k04_medium_r1", "d2_k04_fast_r2", "d3_k04_medium_r0", "d3_k04_fast_r1",
             "d2_k06_medium_r0", "d4_k04_medium_r2", "d3_k06_slow_r1"]
    for f in files:
        tree = json.load(open(os.path.join(DATA, f + ".json")))
        L = math.exp(ed.loglik(tree))
        m, se = mc_loglik(tree, nrep, rng)
        z = (m - L) / se if se > 0 else float("nan")
        rows.append((f, L, m, se, z))
    return rows


def test_d():
    worst = 0.0
    n = 0
    for f in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        tree = json.load(open(f))
        if tree["demes"] ** ed.max_lineages(tree) > 4096:
            continue
        a = ed.loglik(tree, method="expm")
        b = ed.loglik(tree, method="unif")
        worst = max(worst, abs(a - b))
        n += 1
    return worst, n


if __name__ == "__main__":
    t0 = time.time()
    print(f"(a) d=1 Kingman closed form: max |dlogL| = {test_a():.3e}")
    print(f"(b) k=2 analytic / direct expm: max |dlogL| = {test_b():.3e}")
    w, n = test_d()
    print(f"(d) expm_multiply vs uniformization on {n} test trees: max |dlogL| = {w:.3e}")
    print("(c) Monte Carlo over migration histories (40000 paths each):")
    print("    tree                  L_dense          L_MC            SE          z")
    for f, L, m, se, z in test_c():
        print(f"    {f:20s} {L:.6e}  {m:.6e}  {se:.2e}  {z:+.2f}")
    print(f"total {time.time() - t0:.1f}s")
