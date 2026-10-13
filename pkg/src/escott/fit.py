"""Maximum-likelihood fits of constant-rate structured-coalescent models on a fixed tree (escott fit / compare).

Procedure (the one that was needed on the avian-influenza tree, bench/realdata2_masco_refined.md):
  1. MASCOT multistart: L-BFGS-B from N random starts (cheap); distinct local optima = "peaks".
  2. Exact refinement of every peak within REFINE_WITHIN log units of the best: quasi-Newton steps on the exact
     log-likelihood from the MASCOT optimum, with the MASCOT Hessian as the initial curvature model (BFGS-updated
     with the exact gradients) and the exact gradient
     written as g_MASCOT + grad(l_exact - l_MASCOT) (the difference is flat, so a coarse step is accurate and swamps
     tensor-train rounding noise). Parameters at a bound stay there; a KKT probe moves each one inward afterwards.
  3. Cross-check: MASCOT is re-fitted from every refined exact optimum. A better MASCOT optimum found this way means
     the multistart missed one; it becomes a new peak and is refined exactly as well.
Each peak is reported with its status: 'converged' (predicted gain below tolerance and every KKT probe <= 0) or
'lower bound' (the true maximum of that basin may be higher).
Parameters are optimised in log space within [LO, HI] = [-12, 7], i.e. rates and Ne in [6.1e-6, 1097].
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from . import api

LO, HI = -12.0, 7.0
BOUND_MARGIN = 1.0           # a log-parameter within this of LO/HI counts as at the bound
REFINE_WITHIN = 10.0         # refine MASCOT peaks within this many log units of the best one
MAX_REFINE = 5               # ... but at most this many
SAME_LL, SAME_THETA = 0.01, 0.5
MASCOT_MAXITER = 100         # L-BFGS-B iterations per MASCOT start (a start that wanders is cut short)
H_MASCOT_GRAD, H_HESS = 1e-4, 1e-2
H_DELTA = {"tt": 0.1, "dense": 1e-3}
GAIN_TOL = {"tt": 0.01, "dense": 1e-4}
PROBE_LOW, PROBE_HIGH = math.log(0.02), HI - 2.0
MAX_STEP = 2.0               # largest change of one log-parameter per refinement step

# ----------------------------------------------------------------------------- parametrisation


class Model:
    """asym: d(d-1) backward rates + d Ne; sym: d(d-1)/2 symmetric rates + d Ne. theta = log parameters."""

    def __init__(self, kind, names):
        if kind not in ("sym", "asym"):
            raise ValueError("model must be 'sym' or 'asym'")
        self.kind, self.names_d = kind, list(names)
        d = self.d = len(names)
        if kind == "asym":
            self.pairs = [(a, b) for a in range(d) for b in range(d) if a != b]
            self.labels = [f"m.{names[a]}_to_{names[b]}" for a, b in self.pairs]
        else:
            self.pairs = [(a, b) for a in range(d) for b in range(a + 1, d)]
            self.labels = [f"m.{names[a]}<->{names[b]}" for a, b in self.pairs]
        self.labels += [f"Ne.{n}" for n in names]
        self.n_rates = len(self.pairs)
        self.npar = self.n_rates + d

    def unpack(self, theta):
        v = np.exp(np.asarray(theta, float))
        M = np.zeros((self.d, self.d))
        for k, (a, b) in enumerate(self.pairs):
            M[a, b] = v[k]
            if self.kind == "sym":
                M[b, a] = v[k]
        return M.tolist(), v[self.n_rates:].tolist()

    def pack(self, M, Ne):
        M = np.asarray(M, float)
        th = [math.log(max(M[a, b], math.exp(LO))) for a, b in self.pairs] + [math.log(x) for x in Ne]
        return np.clip(th, LO, HI)


# ----------------------------------------------------------------------------- worker processes

_TREE = None


def _init(tree):
    global _TREE
    _TREE = tree


def _evaluate(engine, kind, theta, eps):
    """(loglik, root distribution) at log-parameters theta. engine: 'mascot' or 'exact' (dense/TT, single run)."""
    m = Model(kind, _TREE["deme_names"])
    M, Ne = m.unpack(theta)
    tr = dict(_TREE, M=M, Ne=Ne)
    if engine == "mascot":
        try:
            ll, root = api.mascot_root(tr)
        except RuntimeError:          # the ODE solver gives up at extreme parameter values: treat as very unlikely
            return -1e10, None
    else:
        r = api.loglik(tr, method=api.exact_method(tr), eps=eps, paired=False)
        ll, root = r.loglik, r.info["root_dist"]
    return (ll if math.isfinite(ll) else -1e10), root


def _ev(args):
    return _evaluate(*args)


def _mascot_local(args):
    """Serial L-BFGS-B on the MASCOT log-likelihood from theta0 (runs inside one worker)."""
    kind, theta0, h = args
    n = len(theta0)

    def fg(z):
        f0 = -_evaluate("mascot", kind, z, None)[0]
        g = np.zeros(n)
        for k in range(n):
            zk = np.array(z, float)
            zk[k] += h
            g[k] = (-_evaluate("mascot", kind, zk, None)[0] - f0) / h
        return f0, g

    r = minimize(fg, np.clip(theta0, LO, HI), jac=True, method="L-BFGS-B", bounds=[(LO, HI)] * n,
                 options={"maxiter": MASCOT_MAXITER, "ftol": 1e-11, "gtol": 1e-5})
    ll, root = _evaluate("mascot", kind, r.x, None)
    return {"theta": r.x.tolist(), "loglik": ll, "root": root, "ok": bool(r.success), "msg": str(r.message)}


# ----------------------------------------------------------------------------- steps


def _same(p, q):
    if abs(p["loglik"] - q["loglik"]) >= SAME_LL:
        return False
    a, b = np.array(p["theta"]), np.array(q["theta"])
    inner = ~((a < LO + BOUND_MARGIN) & (b < LO + BOUND_MARGIN)) & ~((a > HI - BOUND_MARGIN) & (b > HI - BOUND_MARGIN))
    return bool(np.all(np.abs(a - b)[inner] < SAME_THETA))


def cluster(fits):
    """Distinct optima, best first, with the number of fits that reached each."""
    peaks = []
    for f in sorted(fits, key=lambda f: -f["loglik"]):
        for p in peaks:
            if _same(p, f):
                p["hits"] += 1
                break
        else:
            peaks.append(dict(f, hits=1))
    return peaks


def refine(pool, kind, theta0, eps, method, max_iter=12, progress=None, label=""):
    """Exact refinement from theta0 (see module docstring). Returns a dict with the refined point and its status."""
    th0 = np.array(theta0, float)
    n_all = len(th0)
    free = [i for i in range(n_all) if LO + BOUND_MARGIN < th0[i] < HI - BOUND_MARGIN]
    bound = [i for i in range(n_all) if i not in free]
    n = len(free)
    hd, tol = H_DELTA[method], GAIN_TOL[method]

    def at(z):
        th = th0.copy()
        th[free] = z
        return th

    def run(jobs):
        return list(pool.map(_ev, [(e, kind, list(t), eps) for e, t in jobs]))

    z = th0[free].copy()
    E = np.eye(n)
    hist, status = [], None
    if n:
        pts = [(i, j, si, sj) for i in range(n) for j in range(i, n) for si in (1, -1) for sj in (1, -1)]
        vals = run([("mascot", at(z + H_HESS * (si * E[i] + sj * E[j]))) for i, j, si, sj in pts])
        H = np.zeros((n, n))
        for (i, j, si, sj), v in zip(pts, vals):
            H[i, j] += si * sj * v[0] / (4 * H_HESS ** 2)
        H = np.triu(H) + np.triu(H, 1).T
        w, V = np.linalg.eigh(-H)
        w = np.maximum(w, 1e-3 * max(w.max(), 1.0))
        B = V @ np.diag(w) @ V.T
    best, known, prev = None, None, None
    for it in range(max_iter if n else 1):
        jobs = [] if known else [("exact", at(z))]
        jobs += [("exact", at(z + hd * E[k])) for k in range(n)]
        jobs += [("mascot", at(z))] + [("mascot", at(z + hd * E[k])) for k in range(n)]
        jobs += [("mascot", at(z + H_MASCOT_GRAD * E[k])) for k in range(n)]
        r = run(jobs)
        if known is None:
            known, r = r[0], r[1:]
        lt, root = known
        lm = r[n][0]
        rec = {"it": it, "theta": at(z).tolist(), "loglik": lt, "root": root, "mascot": lm}
        if best is None or lt > best["loglik"]:
            best = rec
        if not n:
            rec["pred_gain"] = 0.0
            hist.append(rec)
            break
        dT = np.array([r[k][0] for k in range(n)]) - lt
        dM = np.array([r[n + 1 + k][0] for k in range(n)]) - lm
        gM = (np.array([r[2 * n + 1 + k][0] for k in range(n)]) - lm) / H_MASCOT_GRAD
        g = gM + (dT - dM) / hd
        if prev is not None:                 # BFGS update of the curvature model of -log L
            sv, yv = z - prev[0], prev[1] - g
            if sv @ yv > 1e-10 * np.linalg.norm(sv) * np.linalg.norm(yv):
                Bs = B @ sv
                B = B - np.outer(Bs, Bs) / (sv @ Bs) + np.outer(yv, yv) / (yv @ sv)
        prev = (z.copy(), g)
        step = np.linalg.solve(B, g)
        pred = float(g @ step) / 2
        rec["pred_gain"], rec["grad_norm"] = pred, float(np.linalg.norm(g))
        hist.append(rec)
        if progress:
            progress(f"{label} exact refinement it {it}: log L {lt:.5f}, predicted gain {pred:.3g}")
        if pred < tol:
            status = "converged"
            break
        big = float(np.max(np.abs(step)))
        if big > MAX_STEP:                   # never jump more than a factor e^MAX_STEP in one parameter
            step = step * (MAX_STEP / big)
        ln = -math.inf
        for a in (1.0, 0.5, 0.25):
            zn = np.clip(z + a * step, LO, HI)
            ln, rn = run([("exact", at(zn))])[0]
            if ln > lt:
                break
        if ln <= lt:
            status = "step rejected"
            break
        z, known = zn, (ln, rn)
    else:
        status = "iteration cap"
        lt, root = known                                                  # the last accepted step
        if lt > best["loglik"]:
            best = {"theta": at(z).tolist(), "loglik": lt, "root": root, "mascot": run([("mascot", at(z))])[0][0]}
    if status is None:
        status = "converged"
    res = {"theta": best["theta"], "loglik": best["loglik"], "root": best["root"], "mascot_at_point": best["mascot"],
           "iterations": len(hist), "pred_gain": hist[-1].get("pred_gain"), "stop": status, "kkt": {},
           "start_loglik": hist[0]["loglik"]}
    if bound:
        probes = []
        for i in bound:
            th = np.array(best["theta"])
            th[i] = PROBE_LOW if th[i] < LO + BOUND_MARGIN else PROBE_HIGH
            probes.append(th)
        vals = run([("exact", t) for t in probes])
        res["kkt"] = {i: v[0] - best["loglik"] for i, v in zip(bound, vals)}
    return res


def paired_error(tree, kind, theta, eps):
    m = Model(kind, tree["deme_names"])
    M, Ne = m.unpack(theta)
    r = api.loglik(dict(tree, M=M, Ne=Ne), method="tt", eps=eps, paired=True)
    return r.error, r.loglik


# ----------------------------------------------------------------------------- driver


def fit(tree, kind, starts=10, engine="both", eps=1e-6, jobs=1, seed=1, max_iter=12, paired_check=True,
        extra_starts=(), progress=None):
    """Multistart MASCOT, exact refinement of each peak, MASCOT cross-check. Returns a result dict."""
    m = Model(kind, tree["deme_names"])
    method = api.exact_method(tree)
    rng = np.random.default_rng(seed)
    x0s = [rng.uniform(-3, 1.5, m.npar) for _ in range(starts)] + [np.asarray(x, float) for x in extra_starts]
    out = {"model": kind, "labels": m.labels, "npar": m.npar, "engine": engine, "exact_method": method, "eps": eps,
           "starts": len(x0s)}
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(tree,)) as pool:
        _init(tree)
        if progress:
            progress(f"{kind}: MASCOT multistart, {len(x0s)} starts")
        fits = list(pool.map(_mascot_local, [(kind, list(x), 1e-5) for x in x0s]))
        peaks = cluster(fits)
        out["mascot_peaks"] = peaks
        out["starts_not_converged"] = sum(not f["ok"] for f in fits)
        if progress:
            progress(f"{kind}: {len(peaks)} distinct MASCOT optim{'um' if len(peaks) == 1 else 'a'}; best {peaks[0]['loglik']:.4f} "
                     f"(reached by {peaks[0]['hits']}/{len(x0s)} starts)")
        if engine == "mascot":
            return out
        todo = [p for p in peaks if p["loglik"] >= peaks[0]["loglik"] - REFINE_WITHIN][:MAX_REFINE]
        refined = []

        def refine_peak(p, origin):
            lab = f"{kind} peak {len(refined) + 1}"
            r = refine(pool, kind, p["theta"], eps, method, max_iter, progress, lab)
            r.update(mascot_opt=p["loglik"], mascot_root=p["root"], mascot_theta=p["theta"],
                     exact_at_mascot=r["start_loglik"],
                     origin=origin, hits=p.get("hits"))
            refined.append(r)

        for p in todo:
            refine_peak(p, "multistart")
        if progress:
            progress(f"{kind}: cross-check, MASCOT re-fitted from {len(refined)} refined exact optima")
        back = list(pool.map(_mascot_local, [(kind, r["theta"], 1e-5) for r in refined]))
        out["cross_check"] = []
        best_m = peaks[0]["loglik"]
        new_peaks = []
        for r, b in zip(refined, back):
            better = bool(b["loglik"] > best_m + SAME_LL)
            out["cross_check"].append({"from_exact": r["loglik"], "mascot": b["loglik"], "better": better})
            if better and not any(_same(b, q) for q in peaks + new_peaks):
                new_peaks.append(dict(b, hits=0))
        for p in new_peaks:
            if progress:
                progress(f"{kind}: the multistart missed a MASCOT optimum ({p['loglik']:.4f}); refining it too")
            refine_peak(p, "cross-check")
        out["mascot_peaks"] = peaks + new_peaks
    for r in refined:
        if method == "tt" and paired_check:
            if progress:
                progress(f"{kind}: paired check (eps/10, h/2) at the optimum {r['loglik']:.4f}")
            r["error"], _ = paired_error(tree, kind, r["theta"], eps)
        else:
            r["error"] = 0.0 if method == "dense" else api.ERROR_C * eps * \
                api.esco_tt.count_roundings(dict(tree, **dict(zip(("M", "Ne"), m.unpack(r["theta"])))))
        tol = max(2 * r["error"], 1e-3)
        bad = {m.labels[i]: v for i, v in r["kkt"].items() if v > tol}
        if r["stop"] == "converged" and not bad:
            r["status"] = "converged"
        else:
            why = []
            if r["stop"] != "converged":
                why.append(r["stop"])
            why += [f"moving {k} off its bound raises log L by {v:.3g}" for k, v in bad.items()]
            r["status"] = "lower bound (" + "; ".join(why) + ")"
    refined.sort(key=lambda r: -r["loglik"])
    for i, r in enumerate(refined):
        r["same_as"] = next((j + 1 for j in range(i) if _same(refined[j], r)), None)
    out["peaks"] = refined
    return out


def lrt(ll_asym, ll_sym, d, err_asym=0.0, err_sym=0.0):
    """Likelihood-ratio test asym vs sym: (2 Delta, df, p, 2 Delta error, p range)."""
    df = d * (d - 1) // 2
    stat = 2 * (ll_asym - ll_sym)
    e = 2 * (err_asym + err_sym)
    p = float(chi2.sf(max(stat, 0.0), df))
    rng = (float(chi2.sf(max(stat + e, 0.0), df)), float(chi2.sf(max(stat - e, 0.0), df)))
    return {"stat": stat, "df": df, "p": p, "stat_error": e, "p_range": rng}


# ----------------------------------------------------------------------------- printing


def _root(r, names):
    if r is None:
        return "-"
    return "/".join(f"{x:.2f}" for x in r)


def format_fit(res, names):
    L = []
    m = Model(res["model"], names)
    L.append(f"model {res['model']}: {m.n_rates} migration rates + {m.d} Ne = {m.npar} parameters; log-space bounds "
             f"[{math.exp(LO):.2g}, {math.exp(HI):.4g}]")
    pk = res["mascot_peaks"]
    L.append(f"MASCOT multistart: {res['starts']} starts -> {len(pk)} distinct optim{'um' if len(pk) == 1 else 'a'}; best reached by "
             f"{pk[0]['hits']} start(s)" + (f"; {res['starts_not_converged']} start(s) stopped before converging"
                                           if res.get("starts_not_converged") else ""))
    if res["engine"] == "mascot":
        L.append(f"  {'#':>2s} {'MASCOT log L':>13s} {'starts':>6s}  root ({'/'.join(names)})")
        for i, p in enumerate(pk, 1):
            L.append(f"  {i:>2d} {p['loglik']:13.4f} {p['hits']:>6d}  {_root(p['root'], names)}")
        best = pk[0]
        L.append("")
        L.append("best MASCOT fit:")
        for lab, x in zip(m.labels, np.exp(best["theta"])):
            L.append(f"  {lab:<34s} {x:11.4g}")
        return "\n".join(L)
    L.append(f"exact engine: {res['exact_method']}" + (f", eps {res['eps']:g}" if res["exact_method"] == "tt" else ""))
    L.append("")
    L.append(f"peaks, sorted by exact log L ('exact at MASCOT opt.' = exact value at MASCOT's optimum; root = "
             f"probability of {' / '.join(names)}):")
    L.append(f"  {'#':>2s} {'origin':>11s} {'MASCOT opt.':>12s} {'exact at':>12s} {'exact max':>12s} {'error':>8s}  "
             f"{'exact root':>16s} {'MASCOT root':>16s}  status")
    L.append(f"  {'':>2s} {'':>11s} {'':>12s} {'MASCOT opt.':>12s}")
    for i, r in enumerate(res["peaks"], 1):
        L.append(f"  {i:>2d} {r['origin']:>11s} {r['mascot_opt']:12.4f} {r['exact_at_mascot']:12.4f} "
                 f"{r['loglik']:12.4f} {r['error']:8.1g}  {_root(r['root'], names):>16s} "
                 f"{_root(r['mascot_root'], names):>16s}  {r['status']}"
                 + (f"; same exact optimum as #{r['same_as']}" if r.get("same_as") else ""))
    cc = res.get("cross_check", [])
    if any(c["better"] for c in cc):
        L.append("cross-check: re-fitting MASCOT from the refined exact optima found a better MASCOT optimum than the "
                 "multistart did (added above with origin 'cross-check'); compare MASCOT and exact only at these "
                 "refined optima.")
    else:
        L.append("cross-check: re-fitting MASCOT from each refined exact optimum found no better MASCOT optimum.")
    best = res["peaks"][0]
    bm = max(res["mascot_peaks"], key=lambda p: p["loglik"])
    L.append("")
    L.append(f"best exact fit: log L = {best['loglik']:.4f} (error {best['error']:.1g}; {best['status']}); "
             f"MASCOT at the same parameters: {best['mascot_at_point']:.4f}")
    if res["engine"] == "both":
        L.append(f"best MASCOT fit: log L = {bm['loglik']:.4f}")
        L.append(f"  {'parameter':<34s} {'MASCOT MLE':>11s} {'exact MLE':>11s} {'change':>8s}")
        for lab, a, b in zip(m.labels, np.exp(bm["theta"]), np.exp(best["theta"])):
            ch = "at bound" if (min(a, b) <= math.exp(LO + BOUND_MARGIN) or max(a, b) >= math.exp(HI - BOUND_MARGIN)) \
                else f"{100 * (b / a - 1):+.1f}%"
            L.append(f"  {lab:<34s} {a:11.4g} {b:11.4g} {ch:>8s}")
    else:
        for lab, b in zip(m.labels, np.exp(best["theta"])):
            L.append(f"  {lab:<34s} {b:11.4g}")
    return "\n".join(L)


def format_compare(fs, fa, names):
    d = len(names)
    L = [f"likelihood-ratio test, asymmetric vs symmetric migration ({d} demes): df = {d * (d - 1) // 2}; rates at a "
         f"bound make the chi-square reference approximate"]
    bm = lambda f: max(f["mascot_peaks"], key=lambda p: p["loglik"])["loglik"]
    tm = lrt(bm(fa), bm(fs), d)
    ea, es = fa["peaks"][0], fs["peaks"][0]
    te = lrt(ea["loglik"], es["loglik"], d, ea["error"], es["error"])
    L.append(f"  {'engine':<8s} {'best asym':>12s} {'best sym':>12s} {'2 Delta':>16s} {'p':>22s}")
    L.append(f"  {'MASCOT':<8s} {bm(fa):12.4f} {bm(fs):12.4f} {tm['stat']:16.3f} {tm['p']:22.3g}")
    pr = f"{te['p']:.3g} ({te['p_range'][0]:.2g}-{te['p_range'][1]:.2g})" if te["stat_error"] > 0 else f"{te['p']:.3g}"
    st = f"{te['stat']:.3f} +- {te['stat_error']:.2g}" if te["stat_error"] > 0 else f"{te['stat']:.3f}"
    L.append(f"  {'exact':<8s} {ea['loglik']:12.4f} {es['loglik']:12.4f} {st:>16s} {pr:>22s}")
    notes = []
    if not ea["status"].startswith("converged"):
        notes.append("the exact asym optimum is a lower bound, so the exact 2 Delta may be larger")
    if not es["status"].startswith("converged"):
        notes.append("the exact sym optimum is a lower bound, so the exact 2 Delta may be smaller")
    if te["stat"] < -1e-3:
        notes.append("the exact asym optimum is below the sym one: the asym optimisation is incomplete")
    for n in notes:
        L.append("  note: " + n)
    v = lambda p: "rejects symmetry at 5%" if p < 0.05 else "does not reject symmetry at 5%"
    L.append(f"MASCOT {v(tm['p'])}; the exact likelihood {v(te['p'])}.")
    return "\n".join(L), tm, te
