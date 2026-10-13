"""Structured coalescent simulator (backward in time), contract-format JSON output.

Convention (shared with dense/, approx/): M[a][b] (a != b) is the BACKWARD-time rate at
which a single lineage currently in deme a moves to deme b. Within deme a each pair of
lineages coalesces at rate 1/Ne[a]. Heterochronous sampling supported.

Usage:
    python sim/simulate.py            # (re)generate the test set in sim/data/
"""
import json
import os
import sys

import numpy as np


def simulate_tree(M, Ne, sample_times, sample_demes, rng):
    """Simulate a genealogy under the structured coalescent.

    Returns the contract-format event list (time-sorted, present -> past).
    Sample ids are 0..k-1 (in order of sample_times as given); internal ids k, k+1, ...
    """
    M = np.asarray(M, dtype=float)
    Ne = np.asarray(Ne, dtype=float)
    d = len(Ne)
    out_rate = M.sum(axis=1) - np.diag(M)
    k = len(sample_times)
    order = sorted(range(k), key=lambda i: (sample_times[i], i))
    pending = [(float(sample_times[i]), i, int(sample_demes[i])) for i in order]
    events = []
    active = []  # list of [id, deme]
    next_id = k
    t = 0.0
    pi = 0
    while True:
        # add samples at the current time
        while pi < len(pending) and pending[pi][0] <= t:
            ts, sid, sd = pending[pi]
            events.append({"t": ts, "type": "sample", "id": sid, "deme": sd})
            active.append([sid, sd])
            pi += 1
        if pi == len(pending) and len(active) == 1:
            break
        counts = np.bincount([a[1] for a in active], minlength=d) if active else np.zeros(d)
        coal_rates = counts * (counts - 1) / 2.0 / Ne
        mig_rates = np.array([out_rate[a[1]] for a in active]) if active else np.zeros(0)
        total = coal_rates.sum() + mig_rates.sum()
        t_next_sample = pending[pi][0] if pi < len(pending) else np.inf
        dt = rng.exponential(1.0 / total) if total > 0 else np.inf
        if t + dt >= t_next_sample:
            t = t_next_sample
            continue
        t += dt
        u = rng.uniform() * total
        if u < coal_rates.sum():
            a = rng.choice(d, p=coal_rates / coal_rates.sum())
            idx = [i for i, x in enumerate(active) if x[1] == a]
            i, j = rng.choice(len(idx), size=2, replace=False)
            li, lj = active[idx[i]], active[idx[j]]
            events.append({"t": t, "type": "coal", "a": li[0], "b": lj[0], "new": next_id})
            for x in sorted([idx[i], idx[j]], reverse=True):
                active.pop(x)
            active.append([next_id, a])
            next_id += 1
        else:
            i = rng.choice(len(active), p=mig_rates / mig_rates.sum())
            a = active[i][1]
            p = M[a].copy()
            p[a] = 0.0
            active[i][1] = int(rng.choice(d, p=p / p.sum()))
    return events


def make_params(d, regime, rng):
    """Ne heterogeneous; per-pair migration rate = factor * mean coalescence rate * U(0.5,1.5)."""
    factor = {"slow": 0.1, "medium": 1.0, "fast": 10.0}[regime]
    Ne = [1.0 + 0.5 * a for a in range(d)]
    cbar = float(np.mean(1.0 / np.array(Ne)))
    M = np.zeros((d, d))
    for a in range(d):
        for b in range(d):
            if a != b:
                M[a, b] = factor * cbar * rng.uniform(0.5, 1.5)
    return Ne, M.tolist()


def make_samples(k, d, rng, t_span=1.0):
    """About half the samples at t=0, the rest uniform in (0, t_span); demes uniform (each deme
    gets >=1 sample when k >= d)."""
    demes = list(rng.integers(0, d, size=k))
    if k >= d:
        demes[:d] = list(range(d))
        rng.shuffle(demes)
    times = [0.0 if rng.uniform() < 0.5 else float(rng.uniform(0, t_span)) for _ in range(k)]
    times[0] = 0.0
    return times, [int(x) for x in demes]


def generate(d, k, regime, rep, seed):
    rng = np.random.default_rng(seed)
    Ne, M = make_params(d, regime, rng)
    times, demes = make_samples(k, d, rng)
    events = simulate_tree(M, Ne, times, demes, rng)
    return {"demes": d, "M": M, "Ne": Ne, "events": events,
            "meta": {"d": d, "k": k, "regime": regime, "rep": rep, "seed": seed}}


def validate(tree):
    ev = tree["events"]
    ts = [e["t"] for e in ev]
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))
    alive = set()
    for e in ev:
        if e["type"] == "sample":
            assert e["id"] not in alive
            alive.add(e["id"])
        else:
            assert e["a"] in alive and e["b"] in alive
            alive -= {e["a"], e["b"]}
            alive.add(e["new"])
    assert len(alive) == 1


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    regimes = ["slow", "medium", "fast"]
    manifest = []
    seed_base = 20261003
    n = 0
    for d in (2, 3, 4):
        for k in range(4, 21, 2):
            for ri, regime in enumerate(regimes):
                for rep in range(3):
                    seed = seed_base + 100000 * d + 1000 * k + 10 * ri + rep
                    tree = generate(d, k, regime, rep, seed)
                    validate(tree)
                    name = f"d{d}_k{k:02d}_{regime}_r{rep}.json"
                    with open(os.path.join(outdir, name), "w") as f:
                        json.dump(tree, f)
                    manifest.append(name)
                    n += 1
    big = os.path.join(outdir, "large")
    os.makedirs(big, exist_ok=True)
    for k in (30, 40, 50, 60):
        for ri, regime in enumerate(regimes):
            for rep in range(2):
                seed = seed_base + 9000000 + 1000 * k + 10 * ri + rep
                tree = generate(4, k, regime, rep, seed)
                validate(tree)
                name = f"d4_k{k:02d}_{regime}_r{rep}.json"
                with open(os.path.join(big, name), "w") as f:
                    json.dump(tree, f)
                manifest.append("large/" + name)
    with open(os.path.join(outdir, "MANIFEST.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"wrote {len(manifest)} trees to {outdir}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "data"))
