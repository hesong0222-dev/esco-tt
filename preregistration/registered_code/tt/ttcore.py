"""Minimal tensor-train (TT) toolkit for ESCO. Cores: list of arrays (r_{i-1}, d, r_i), r_0 = r_k = 1."""
import numpy as np
import scipy.linalg


def _svd(A):
    """SVD with gesvd fallback (gesdd occasionally fails to converge)."""
    try:
        return np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(A, full_matrices=False, lapack_driver="gesvd")


def ranks(cores):
    return [c.shape[0] for c in cores[1:]]


def max_rank(cores):
    return max([1] + ranks(cores))


def to_dense(cores):
    v = cores[0]
    for c in cores[1:]:
        v = np.tensordot(v, c, axes=([v.ndim - 1], [0]))
    return v.reshape(v.shape[1:-1])


def from_dense(T, eps=0.0):
    """Exact TT-SVD of a dense tensor (tests only)."""
    k = T.ndim
    d = T.shape[0]
    cores = []
    r = 1
    A = T.reshape(1, -1)
    for i in range(k - 1):
        A = A.reshape(r * d, -1)
        U, s, Vt = _svd(A)
        keep = _keep(s, eps * np.linalg.norm(s) / np.sqrt(k - 1))
        cores.append(U[:, :keep].reshape(r, d, keep))
        A = (s[:keep, None] * Vt[:keep])
        r = keep
    cores.append(A.reshape(r, d, 1))
    return cores


def _keep(s, delta, rmax=None):
    """number of singular values to keep so that discarded 2-norm <= delta."""
    if s.size == 0:
        return 1
    tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]  # tail[j] = norm of s[j:]
    keep = int(np.searchsorted(-tail, -delta, side="right"))  # first j with tail[j] <= delta
    keep = max(1, min(keep, s.size))
    if rmax is not None:
        keep = min(keep, rmax)
    return keep


def right_canonicalize(cores):
    """RQ sweep right->left. Returns new cores, right-canonical except the first (carries norm)."""
    cores = [c.copy() for c in cores]
    for i in range(len(cores) - 1, 0, -1):
        r0, d, r1 = cores[i].shape
        A = cores[i].reshape(r0, d * r1)
        Q, R = np.linalg.qr(A.T)  # A.T = Q R  ->  A = R^T Q^T
        rr = Q.shape[1]
        cores[i] = Q.T.reshape(rr, d, r1)
        cores[i - 1] = np.tensordot(cores[i - 1], R.T, axes=([2], [0]))
    return cores


def norm(cores):
    c = right_canonicalize(cores)
    return np.linalg.norm(c[0])


def tt_round(cores, eps, rmax=None):
    """Oseledets rounding: left-to-right QR, then right-to-left SVD truncation.
    Result is right-canonical with norm in core 0. Relative Frobenius error <= eps (approximately)."""
    k = len(cores)
    if k == 1:
        return [cores[0].copy()]
    cores = [c.copy() for c in cores]
    # left-to-right QR
    for i in range(k - 1):
        r0, d, r1 = cores[i].shape
        Q, R = np.linalg.qr(cores[i].reshape(r0 * d, r1))
        cores[i] = Q.reshape(r0, d, Q.shape[1])
        cores[i + 1] = np.tensordot(R, cores[i + 1], axes=([1], [0]))
    nrm = np.linalg.norm(cores[-1])
    delta = eps * nrm / np.sqrt(k - 1)
    # right-to-left SVD
    for i in range(k - 1, 0, -1):
        r0, d, r1 = cores[i].shape
        U, s, Vt = _svd(cores[i].reshape(r0, d * r1))
        keep = _keep(s, delta, rmax)
        cores[i] = Vt[:keep].reshape(keep, d, r1)
        cores[i - 1] = np.tensordot(cores[i - 1], U[:, :keep] * s[:keep], axes=([2], [0]))
    return cores


def tt_sum(cores):
    """Sum of all entries."""
    v = np.ones(1)
    for c in cores:
        v = v @ c.sum(axis=1)
    return float(v[0])


def inner(a, b):
    v = np.ones((1, 1))
    for ca, cb in zip(a, b):
        v = np.einsum("ab,aic,bid->cd", v, ca, cb, optimize=True)
    return float(v[0, 0])


def apply_site_matrix(cores, E):
    """P_new(..l'..) = sum_l P(..l..) E[l,l'] on every site (exact, rank-preserving)."""
    return [np.tensordot(c, E, axes=([1], [0])).transpose(0, 2, 1) for c in cores]


def insert_site(cores, pos, vec):
    """Insert a new site at index pos with rank-1 content vec (length d)."""
    r = cores[pos].shape[0] if pos < len(cores) else cores[-1].shape[2]
    new = np.einsum("ab,l->alb", np.eye(r), vec)
    return cores[:pos] + [new] + cores[pos:]


def merge_sites_diag(cores, pos, w):
    """Contract sites pos,pos+1 along the diagonal l_pos = l_{pos+1} = a with weight w[a]."""
    A, B = cores[pos], cores[pos + 1]
    H = np.matmul(A.transpose(1, 0, 2), B.transpose(1, 0, 2)) * w[:, None, None]  # (l, a, c)
    H = H.transpose(1, 0, 2)
    return cores[:pos] + [H] + cores[pos + 2:]


def hadamard_full(P, W):
    """Exact Hadamard product of two TTs (rank product)."""
    out = []
    for a, b in zip(P, W):
        r0, d, r1 = a.shape
        s0, _, s1 = b.shape
        out.append((a[:, None, :, :, None] * b[None, :, :, None, :]).reshape(r0 * s0, d, r1 * s1))
    return out


def hadamard_zipup(P, W, eps, rmax=None):
    """Hadamard product P∘W with truncation during a left-to-right sweep (zip-up).
    P is right-canonicalized first so that the left-side truncation is near-optimal.
    A final tt_round(eps) should be applied by the caller for a norm-controlled result."""
    k = len(P)
    P = right_canonicalize(P)
    nrm = np.linalg.norm(P[0])
    delta = eps * nrm / np.sqrt(max(k - 1, 1))
    out = []
    C = np.ones((1, 1, 1))  # (x, alpha_p, alpha_w)
    for i in range(k):
        a, b = P[i], W[i]
        p, d, rb = a.shape
        q, _, rc = b.shape
        x = C.shape[0]
        # tmp[x,q,l,b] = sum_p C[x,p,q] a[p,l,b]
        tmp = np.tensordot(C, a, axes=([1], [0]))              # (x, q, l, b)
        tmp = tmp.transpose(2, 0, 3, 1).reshape(d, x * rb, q)   # (l, x*b, q)
        T = np.matmul(tmp, b.transpose(1, 0, 2))                # (l, x*b, c)
        T = T.reshape(d, x, rb, rc).transpose(1, 0, 2, 3)       # (x, l, b, c)
        if i == k - 1:
            out.append(T.reshape(x, d, rb * rc))
            break
        U, s, Vt = _svd(T.reshape(x * d, rb * rc))
        keep = _keep(s, delta, rmax)
        out.append(U[:, :keep].reshape(x, d, keep))
        C = (s[:keep, None] * Vt[:keep]).reshape(keep, rb, rc)
    return out


def hadamard_apply(P, W, eps, rmax=None, full_limit=4000):
    """P∘W rounded to eps: exact product + rounding when the product rank is small, zip-up otherwise."""
    rp, rw = max_rank(P), max_rank(W)
    if rp * rw * rp * rw <= full_limit:
        return tt_round(hadamard_full(P, W), eps, rmax)
    return tt_round(hadamard_zipup(P, W, eps * 0.1, rmax), eps, rmax)


def _diag_scale(W):
    """rough sup-norm estimate of a diagonal TT (product of per-core max abs), used only to scale tolerance."""
    return 1.0  # W used here is a decay factor with entries in (0,1]; keep scale 1.
