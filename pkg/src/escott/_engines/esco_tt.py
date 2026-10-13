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


def _mass_hadamard(P, W):
    """sum of entries of P∘W (= <P, W>), by tensordot contractions (exact, no rounding)."""
    v = np.ones((1, 1))
    for cp, cw in zip(P, W):
        t = np.tensordot(v, cp, axes=([0], [0]))            # (b, l, c)
        v = np.tensordot(t, cw, axes=([0, 1], [0, 1]))      # (c, e)
    return float(v[0, 0])


def count_roundings(tree, steps_mult=1, s_max=0.25, nmin=2, order=4):
    """Number of TT roundings loglik_detail will perform (same step rule; no TT arithmetic on P)."""
    d = tree["demes"]
    M = np.array(tree["M"], dtype=float)
    Ne = np.array(tree["Ne"], dtype=float)
    qmax = float(np.max(M.sum(axis=1) - np.diag(M))) if d > 1 else 0.0
    cmax = float(np.max(1.0 / (2 * Ne)))
    events = sorted(tree["events"], key=lambda e: e["t"])
    k, t_prev, n_round = 0, events[0]["t"], 0
    for e in events:
        dt = e["t"] - t_prev
        t_prev = e["t"]
        if k >= 2 and dt > 0:
            h_max = min(s_max / (cmax * k * (k - 1)), s_max / qmax if qmax > 0 else np.inf)
            n = max(nmin, int(math.ceil(dt / h_max))) * steps_mult
            n_ops = 1 if d == 2 else len(decay_operator(k, d, Ne, dt / n)[0])
            n_round += ((5 if order == 4 else 1) * n + 1) * n_ops
        if e["type"] == "sample":
            k += 1
        else:
            k -= 1
            n_round += 1 if k > 1 else 0
    return n_round


# ----------------------------------------------------------------------------- main
RENORM_BELOW = 1e-30   # renormalise the TT between events once its mass drops below this (B1 underflow guard)


def loglik_detail(tree, eps=1e-8, steps_mult=1, s_max=0.25, nmin=2, rmax=None, zipup=True, order=4, verbose=False,
                  mass_fix=False, eps_total=None):
    """Returns dict with loglik, max_rank, runtime, n_steps, decay op ranks, n_round (number of TT roundings).
    mass_fix=True: after every rounding rescale the TT so that its total mass equals the exact mass of the
    unrounded tensor (sum of P∘W = <P, W> for a decay step; 1 after a coalescence).  This removes the
    first-order, one-signed mass loss of Frobenius truncation.  mass_fix=False (default) is the original
    algorithm used for all results before 2026-10-06.
    eps_total: if given, the per-rounding tolerance is eps = eps_total / N_round (N_round from count_roundings), so
    that the accumulated truncation, empirically |dlogL| ~ c * eps * N_round, is ~ c * eps_total (eps is ignored).
    Note (review/AUDIT_CODE.md B4): mass_fix did not reduce the error in the audit (sign flips, same or larger |error|
    on 3 of 4 trees; the large slow-tree error remained), so it is not recommended; kept only as an experiment.
    Underflow guard (AUDIT B1): between coalescences the survival mass is moved into loglik whenever it falls below
    RENORM_BELOW, so it can never underflow; results are bit-identical whenever that never happens.
    rmax_cuts (output): number of SVD truncations whose rank was set by rmax rather than eps (AUDIT B5); if > 0 the
    value carries a truncation error not controlled by eps."""
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
    if eps_total is not None:
        eps = eps_total / max(1, count_roundings(tree, steps_mult, s_max, nmin, order))

    t0 = time.time()
    alive = []          # lineage ids ordered by planar position
    cores = []          # TT cores, one per alive lineage
    ll = 0.0
    t_prev = events[0]["t"] if events else 0.0
    maxr = 1
    nsteps_total = 0
    op_rank_max = 1
    n_round = 0
    rmax_cuts0 = T.rmax_cuts

    def evolve(cores, dt):
        nonlocal maxr, nsteps_total, ll
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
            nonlocal maxr, n_round, ll
            for W in ops:
                target = _mass_hadamard(cores, W) if mass_fix else None
                cores = T.hadamard_apply(cores, W, eps, rmax, full_limit=4000 if zipup else 10 ** 12)
                n_round += 1
                if mass_fix:
                    cores[0] = cores[0] * (target / T.tt_sum(cores))
                maxr = max(maxr, T.max_rank(cores))
                m = T.tt_sum(cores)
                if not (math.isfinite(m) and m > 0):
                    raise FloatingPointError(f"TT mass became {m!r} during propagation")
                if m < RENORM_BELOW:          # move the survival factor into loglik before it can underflow
                    cores[0] = cores[0] / m
                    ll += math.log(m)
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
            if not math.isfinite(s):
                raise FloatingPointError(f"TT mass became {s!r} at a coalescence")
            if not (s > 0):
                return dict(loglik=-np.inf, max_rank=maxr, runtime=time.time() - t0,
                        rmax_cuts=T.rmax_cuts - rmax_cuts0)
            cores[0] = cores[0] / s
            ll += math.log(s)
            if len(cores) > 1:
                cores = T.tt_round(cores, eps, rmax)
                n_round += 1
                if mass_fix:
                    cores[0] = cores[0] / T.tt_sum(cores)
        else:
            raise ValueError(e["type"])
        if verbose:
            print(f"[{time.time() - t0:.0f}s] t={e['t']:.4g} {e['type']} k={len(cores)} maxrank={T.max_rank(cores)} ll={ll:.6f}", flush=True)
    return dict(loglik=ll, max_rank=maxr, runtime=time.time() - t0, n_steps=nsteps_total, n_round=n_round, eps=eps,
                op_rank=op_rank_max, rmax_cuts=T.rmax_cuts - rmax_cuts0, root_dist=T.to_dense(cores).ravel().tolist() if len(cores) == 1 else None)


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
