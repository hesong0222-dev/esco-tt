"""Checks of approx/independence.py (SISCO / LISCO).

1. kappa=1 reproduces approx/mascot.py on the 243 small trees.
2. LISCO kappa=1/2 form == literal transliteration of the upstream Java (unnormalised q, log masses at coalescence).
3. d=1: all three equal the Kingman closed form.
4. Fast-migration limit vs exact dense (all -> exact); 5. slow-migration scan (diagnostic, no exact limit).
Usage: .venv/bin/python bench/sisco_validate.py  -> prints a report (copied into bench/sisco.md)
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "dense"), os.path.join(ROOT, "sim")]
from approx import independence as ind  # noqa: E402
from approx.mascot import loglik as mascot  # noqa: E402
import esco_dense  # noqa: E402
import simulate  # noqa: E402
from validate import kingman_loglik  # noqa: E402


def lisco_java(tree):
    """IndependentStructuredCoalescent.calculateLogP + independent_ode_integrator, but with DOP853 instead of RK4."""
    M = np.array(tree["M"], float)
    np.fill_diagonal(M, 0.0)
    d = M.shape[0]
    half = 0.5 / np.array(tree["Ne"], float)   # coalescent_rates[k] = input/2

    def rhs(t, q, n):
        q = q.reshape(n, d)
        pn = q / q.sum(1, keepdims=True)
        S = pn.sum(0)
        mig = q @ M - q * M.sum(1)[None, :]
        return (mig - half[None, :] * q * (S[None, :] - pn)).ravel()

    ids, q, logP, t = [], np.zeros((0, d)), 0.0, None
    for ev in tree["events"]:
        if t is not None and ev["t"] > t and ids:
            n = len(ids)
            q = solve_ivp(rhs, (t, ev["t"]), q.ravel(), method="DOP853", rtol=1e-11, atol=1e-14,
                          args=(n,)).y[:, -1].reshape(n, d)
        t = ev["t"]
        if ev["type"] == "sample":
            q = np.vstack([q, np.eye(d)[ev["deme"]]])
            ids.append(ev["id"])
        else:
            mass = q.sum(1)
            logP += np.log(mass).sum()
            q = q / mass[:, None]
            i, j = ids.index(ev["a"]), ids.index(ev["b"])
            lam = 2 * half * q[i] * q[j]
            logP += math.log(lam.sum())
            keep = [x for x in range(len(ids)) if x not in (i, j)]
            q = np.vstack([q[keep], lam / lam.sum()])
            ids = [ids[x] for x in keep] + [ev["new"]]
    return logP


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "sim", "data", "d*_k*_*.json")))
    w1 = max(abs(ind.loglik(f, "mascot") - mascot(f)) for f in files)
    print(f"1. kappa=1 vs approx/mascot.py, {len(files)} trees: max |diff| = {w1:.1e}")
    w2 = max(abs(ind.loglik(f, "lisco") - lisco_java(json.load(open(f)))) for f in files[::4])
    print(f"2. LISCO kappa form vs Java-literal q form, {len(files[::4])} trees: max |diff| = {w2:.1e}")

    rng = np.random.default_rng(7)
    w3 = 0.0
    for k in (2, 5, 10, 20, 40):
        times, demes = simulate.make_samples(k, 1, rng)
        ev = simulate.simulate_tree([[0.0]], [1.7], times, demes, rng)
        tr = {"M": [[0.0]], "Ne": [1.7], "events": ev}
        for m in ("sisco", "lisco", "mascot"):
            w3 = max(w3, abs(ind.loglik(tr, m) - kingman_loglik(tr)))
    print(f"3. d=1 vs Kingman closed form (k=2..40): max |diff| = {w3:.1e}")

    # fast migration: scale M of small d=3 trees up; exact and all closures must converge (error ~ 1/s)
    print("4. fast-migration limit (M scaled x s on d=3 k=8 medium trees; exact dense):")
    for f in sorted(glob.glob(os.path.join(ROOT, "sim", "data", "d3_k08_medium_r*.json"))):
        tr = json.load(open(f))
        for s in (1, 10, 100, 1000):
            t2 = dict(tr, M=(np.array(tr["M"]) * s).tolist())
            ex = esco_dense.loglik(t2)
            print(f"   {os.path.basename(f)} x{s:<5d} exact {ex:10.5f}  "
                  + "  ".join(f"{m}-exact {ind.loglik(t2, m) - ex:+.2e}" for m in ("sisco", "lisco", "mascot")))
    print("5. slow-migration limit (M scaled x s, same trees):")
    for f in sorted(glob.glob(os.path.join(ROOT, "sim", "data", "d3_k08_medium_r*.json")))[:2]:
        tr = json.load(open(f))
        for s in (1, 0.1, 0.01, 0.001):
            t2 = dict(tr, M=(np.array(tr["M"]) * s).tolist())
            ex = esco_dense.loglik(t2)
            print(f"   {os.path.basename(f)} x{s:<6g} exact {ex:10.5f}  "
                  + "  ".join(f"{m}-exact {ind.loglik(t2, m) - ex:+.2e}" for m in ("sisco", "lisco", "mascot")))


if __name__ == "__main__":
    main()
