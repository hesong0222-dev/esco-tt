"""Brute-force ESCO likelihood on the full d^k vector (sparse generator + expm_multiply).  For validation only."""
import json
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from .esco_tt import planar_leaf_order


def generator(k, d, Q, Ne):
    """A such that dP/dt = A P for the flattened (d,)*k tensor (C order, site 0 slowest)."""
    N = d ** k
    I = sp.identity
    A = sp.csr_matrix((N, N))
    QT = sp.csr_matrix(Q.T)
    for i in range(k):
        A = A + sp.kron(sp.kron(I(d ** i), QT), I(d ** (k - i - 1)), format="csr")
    # decay
    idx = np.indices((d,) * k).reshape(k, -1)  # (k, N)
    dec = np.zeros(N)
    for a in range(d):
        na = (idx == a).sum(axis=0)
        dec += na * (na - 1) / (2.0 * Ne[a])
    A = A - sp.diags(dec)
    return A.tocsr()


def loglik(tree, return_root=False):
    if isinstance(tree, str):
        tree = json.loads(tree)
    d = tree["demes"]
    M = np.array(tree["M"], dtype=float)
    Ne = np.array(tree["Ne"], dtype=float)
    Q = M.copy()
    np.fill_diagonal(Q, 0.0)
    Q -= np.diag(Q.sum(axis=1))
    events = sorted(tree["events"], key=lambda e: e["t"])
    pos = planar_leaf_order(events)
    alive, P, ll = [], None, 0.0
    t_prev = events[0]["t"]
    for e in events:
        dt = e["t"] - t_prev
        t_prev = e["t"]
        k = len(alive)
        if k > 0 and dt > 0:
            A = generator(k, d, Q, Ne)
            P = expm_multiply(A * dt, P.ravel()).reshape((d,) * k)
        if e["type"] == "sample":
            p = pos[e["id"]]
            idx = sum(1 for x in alive if pos[x] < p)
            vec = np.zeros(d)
            vec[e["deme"]] = 1.0
            if P is None:
                P = vec.copy()
            else:
                P = np.expand_dims(P, idx) * vec.reshape((1,) * idx + (d,) + (1,) * (k - idx))
            alive.insert(idx, e["id"])
        else:
            ia, ib = alive.index(e["a"]), alive.index(e["b"])
            i, j = min(ia, ib), max(ia, ib)
            D = np.diagonal(P, axis1=i, axis2=j)          # diag axis moved to the end
            D = np.moveaxis(D, -1, i)
            D = D / Ne.reshape((1,) * i + (d,) + (1,) * (k - 2 - i))
            P = D
            alive[i:i + 2] = [e["new"]]
            s = P.sum()
            P = P / s
            ll += math.log(s)
    return (ll, P) if return_root else ll


if __name__ == "__main__":
    import sys
    print(loglik(json.load(open(sys.argv[1]))))
