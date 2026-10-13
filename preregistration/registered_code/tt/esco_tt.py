"""Tensor-train exact structured coalescent (ESCO) likelihood.  Interface: loglik(tree_json) -> float.
See CONTRACT.md for the model.  Details in loglik_detail()."""
import json
import math
import time
import numpy as np
from scipy.linalg import expm
from . import ttcore as T


# ----------------------------------------------------------------------------- tree helpers
def planar_leaf_order(events):
    """Position of every node = index of the leftmost leaf of its subtree in a DFS planar order."""
    children = {e["new"]: (e["a"], e["b"]) for e in events if e["type"] == "coal"}
    leaves = [e["id"] for e in events if e["type"] == "sample"]
    is_child = set(c for ab in children.values() for c in ab)
    roots = [n for n in list(leaves) + list(children) if n not in is_child]
    pos = {}
    counter = [0]

    def dfs(v):
        if v in children:
            a, b = children[v]
            dfs(a)
            dfs(b)
            pos[v] = min(pos[a], pos[b])
        else:
            pos[v] = counter[0]
            counter[0] += 1

    import sys
    sys.setrecursionlimit(max(10000, 10 * len(events)))
    for r in roots:
        dfs(r)
    return pos


# ----------------------------------------------------------------------------- decay operator
def count_function_tt(k, d, a, f):
    """Diagonal TT of g(l_1..l_k) = f(n_a), n_a = #{i : l_i = a}.  Exact, rank <= k (before rounding)."""
    ind = np.zeros(d)
    ind[a] = 1.0
    cores = []
    for i in range(k):
        m_in = i + 1        # partial count before site i in 0..i
        m_out = i + 2
        if i == k - 1:
            W = np.zeros((m_in, d, 1))
            for m in range(m_in):
                for l in range(d):
                    W[m, l, 0] = f(m + int(ind[l]))
        else:
            W = np.zeros((m_in, d, m_out))
            for m in range(m_in):
                for l in range(d):
                    W[m, l, m + int(ind[l])] = 1.0
        cores.append(W)
    return cores


def decay_operator(k, d, Ne, h, eps_op=1e-14):
    """List of diagonal TTs whose Hadamard product equals exp(-h * sum_a n_a(n_a-1)/(2 Ne_a)).
    For d=2 a single count function suffices (n_1 = k - n_0).  Returns (list_of_TT, max_rank)."""
    c = [1.0 / (2.0 * Ne[a]) for a in range(d)]
    if d == 2:
        f = lambda n: math.exp(-h * (c[0] * n * (n - 1) + c[1] * (k - n) * (k - n - 1)))
        ops = [T.tt_round(count_function_tt(k, d, 0, f), eps_op)]
    else:
        ops = []
        for a in range(d):
            f = (lambda ca: (lambda n: math.exp(-h * ca * n * (n - 1))))(c[a])
            ops.append(T.tt_round(count_function_tt(k, d, a, f), eps_op))
        # try merging into one operator if it stays cheap
        merged = ops[0]
        ok = True
        for W in ops[1:]:
            merged = T.tt_round(T.hadamard_full(merged, W), eps_op)
            if T.max_rank(merged) ** 2 > sum(T.max_rank(o) ** 2 for o in ops):
                ok = False
                break
        if ok:
            ops = [merged]
    return ops, max(T.max_rank(o) for o in ops)


# ----------------------------------------------------------------------------- main
def loglik_detail(tree, eps=1e-8, steps_mult=1, s_max=0.25, nmin=2, rmax=None, zipup=True, order=4, verbose=False):
    """Returns dict with loglik, max_rank, runtime, n_steps, decay op ranks."""
    if isinstance(tree, str):
        tree = json.loads(tree)
    d = tree["demes"]
    M = np.array(tree["M"], dtype=float)
    Ne = np.array(tree["Ne"], dtype=float)
    Q = M.copy()
    np.fill_diagonal(Q, 0.0)
    Q -= np.diag(Q.sum(axis=1))
    qmax = float(np.max(-np.diag(Q))) if d > 1 else 0.0
    cmax = float(np.max(1.0 / (2 * Ne)))
    events = sorted(tree["events"], key=lambda e: e["t"])
    pos = planar_leaf_order(events)

    t0 = time.time()
    alive = []          # lineage ids ordered by planar position
    cores = []          # TT cores, one per alive lineage
    ll = 0.0
    t_prev = events[0]["t"] if events else 0.0
    maxr = 1
    nsteps_total = 0
    op_rank_max = 1

    def evolve(cores, dt):
        nonlocal maxr, nsteps_total
        k = len(cores)
        if k == 0 or dt <= 0:
            return cores
        if k == 1:
            return T.apply_site_matrix(cores, expm(dt * Q))
        h_max = min(s_max / (cmax * k * (k - 1)), s_max / qmax if qmax > 0 else np.inf)
        n = max(nmin, int(math.ceil(dt / h_max))) * steps_mult
        h = dt / n
        nsteps_total += n
        # composition of Strang steps S(tau) = D(tau/2) Mig(tau) D(tau/2); order 2 or 4 (Suzuki-Yoshida)
        if order == 4:
            p4 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
            taus = [p4, p4, 1.0 - 4.0 * p4, p4, p4]
        else:
            taus = [1.0]
        seq = []  # alternating decay/migration fractions of h, decays merged between substeps
        dec_acc = 0.0
        for tau in taus:
            dec_acc += tau / 2.0
            seq.append(("D", dec_acc)); seq.append(("M", tau)); dec_acc = tau / 2.0
        seq.append(("D", dec_acc))
        # full-step sequence: first D, then repeated [M, D(merged)], last D is the closing half
        Ecache, Dcache = {}, {}

        def getE(f):
            if f not in Ecache:
                Ecache[f] = expm(f * h * Q)
            return Ecache[f]

        def getD(f):
            nonlocal op_rank_max
            if f not in Dcache:
                ops, rk = decay_operator(k, d, Ne, f * h)
                op_rank_max = max(op_rank_max, rk)
                Dcache[f] = ops
            return Dcache[f]

        def decay(cores, ops):
            nonlocal maxr
            for W in ops:
                cores = T.hadamard_apply(cores, W, eps, rmax, full_limit=4000 if zipup else 10 ** 12)
                maxr = max(maxr, T.max_rank(cores))
            return cores

        inner = seq[1:-1]
        first, last = seq[0][1], seq[-1][1]
        for s in range(n):
            cores = decay(cores, getD(first if s == 0 else first + last))
            for kind, f in inner:
                cores = T.apply_site_matrix(cores, getE(f)) if kind == "M" else decay(cores, getD(f))
        cores = decay(cores, getD(last))
        return cores

    for e in events:
        cores = evolve(cores, e["t"] - t_prev)
        t_prev = e["t"]
        if e["type"] == "sample":
            p = pos[e["id"]]
            idx = sum(1 for x in alive if pos[x] < p)
            vec = np.zeros(d)
            vec[e["deme"]] = 1.0
            cores = T.insert_site(cores, idx, vec) if cores else [vec.reshape(1, d, 1)]
            alive.insert(idx, e["id"])
        elif e["type"] == "coal":
            ia, ib = alive.index(e["a"]), alive.index(e["b"])
            i, j = min(ia, ib), max(ia, ib)
            assert j == i + 1, "coalescing lineages must be adjacent in planar order"
            cores = T.merge_sites_diag(cores, i, 1.0 / Ne)
            alive[i:i + 2] = [e["new"]]
            s = T.tt_sum(cores)
            if not (s > 0):
                return dict(loglik=-np.inf, max_rank=maxr, runtime=time.time() - t0)
            cores[0] = cores[0] / s
            ll += math.log(s)
            if len(cores) > 1:
                cores = T.tt_round(cores, eps, rmax)
        else:
            raise ValueError(e["type"])
        if verbose:
            print(f"[{time.time() - t0:.0f}s] t={e['t']:.4g} {e['type']} k={len(cores)} maxrank={T.max_rank(cores)} ll={ll:.6f}", flush=True)
    return dict(loglik=ll, max_rank=maxr, runtime=time.time() - t0, n_steps=nsteps_total,
                op_rank=op_rank_max, root_dist=T.to_dense(cores).ravel().tolist() if len(cores) == 1 else None)


def loglik(tree, eps=1e-8, **kw):
    return loglik_detail(tree, eps=eps, **kw)["loglik"]


def loglik_with_error(tree, eps=1e-8, eps_factor=1e-2, **kw):
    """loglik plus error estimate: truncation (eps vs eps/100) and time step (h vs h/2, Richardson)."""
    r1 = loglik_detail(tree, eps=eps, **kw)
    r2 = loglik_detail(tree, eps=eps, steps_mult=2, **kw)
    r3 = loglik_detail(tree, eps=eps * eps_factor, **kw)
    fac = 15.0 if kw.get("order", 4) == 4 else 3.0
    rich = r2["loglik"] + (r2["loglik"] - r1["loglik"]) / fac
    return dict(loglik=rich, ll_h=r1["loglik"], ll_h2=r2["loglik"], ll_eps2=r3["loglik"],
                err_step=abs(r2["loglik"] - r1["loglik"]) / fac, err_trunc=abs(r3["loglik"] - r1["loglik"]),
                max_rank=r1["max_rank"], max_rank_eps2=r3["max_rank"], runtime=r1["runtime"])


if __name__ == "__main__":
    import sys
    tree = json.load(open(sys.argv[1]))
    print(loglik_detail(tree, verbose=True))
