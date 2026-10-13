"""Quick structured-coalescent tree simulator (backward Gillespie) writing CONTRACT.md JSON.  Stand-in until sim/ exists."""
import json
import numpy as np


def migration_matrix(d, m, rng=None, asym=0.0):
    """Symmetric all-to-all backward migration with per-lineage total rate m; asym>0 adds lognormal jitter."""
    M = np.full((d, d), m / max(d - 1, 1))
    if asym > 0 and rng is not None:
        M = M * rng.lognormal(0.0, asym, size=(d, d))
    np.fill_diagonal(M, 0.0)
    return M


def simulate(n, d, M, Ne, seed=0, sample_demes=None, sample_times=None):
    rng = np.random.default_rng(seed)
    M = np.asarray(M, float)
    Ne = np.asarray(Ne, float)
    if sample_demes is None:
        sample_demes = [i % d for i in range(n)]
    if sample_times is None:
        sample_times = [0.0] * n
    order = np.argsort(sample_times, kind="stable")
    pending = [(float(sample_times[i]), int(i), int(sample_demes[i])) for i in order]
    events = []
    alive = {}          # id -> deme
    t = 0.0
    next_id = n
    n_mig = 0
    while pending or len(alive) > 1:
        # rates
        rates = []
        for a in range(d):
            ids = [i for i, dm in alive.items() if dm == a]
            na = len(ids)
            if na >= 2:
                rates.append(("coal", a, na * (na - 1) / 2.0 / Ne[a]))
            for b in range(d):
                if b != a and na > 0 and M[a, b] > 0:
                    rates.append(("mig", (a, b), na * M[a, b]))
        tot = sum(r[2] for r in rates)
        dt = rng.exponential(1.0 / tot) if tot > 0 else np.inf
        if pending and (t + dt >= pending[0][0]):
            ts, i, dm = pending.pop(0)
            t = ts
            alive[i] = dm
            events.append({"t": t, "type": "sample", "id": i, "deme": dm})
            continue
        t += dt
        u = rng.uniform() * tot
        for kind, arg, r in rates:
            u -= r
            if u <= 0:
                break
        if kind == "coal":
            ids = [i for i, dm in alive.items() if dm == arg]
            a, b = rng.choice(ids, 2, replace=False)
            del alive[int(a)], alive[int(b)]
            alive[next_id] = arg
            events.append({"t": t, "type": "coal", "a": int(a), "b": int(b), "new": next_id})
            next_id += 1
        else:
            a, b = arg
            ids = [i for i, dm in alive.items() if dm == a]
            i = int(rng.choice(ids))
            alive[i] = b
            n_mig += 1
    return {"demes": d, "M": M.tolist(), "Ne": Ne.tolist(), "events": events,
            "meta": {"n": n, "n_mig": n_mig, "seed": seed, "tmrca": t}}


if __name__ == "__main__":
    import sys
    n, d, m = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    tr = simulate(n, d, migration_matrix(d, m), [1.0] * d, seed=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    json.dump(tr, sys.stdout)
