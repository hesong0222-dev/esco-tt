"""Rough cost of an exact likelihood evaluation before running it (escott estimate).

Everything here is an empirical summary of the benchmark runs, not a guarantee:
  * TT rank: max TT rank at eps 1e-6 on simulated trees (bench/robustness.md section 3, 3 seeds per cell, plus the
    large d = 4 trees of section 4), indexed by d, migration regime and the largest number of lineages present at once.
  * TT time per evaluation: least-squares fit over 428 runs (342 simulated trees, 81 evaluations on the 133-tip
    avian-influenza tree, 5 H3N2 posterior trees; Apple M4 under load, and Kaggle CPUs)
      t = exp(-10.687) * N_round^0.055 * k^2.141 * rank^1.524 * d^3.265 seconds,
    90% of the runs within x0.34-x3.5 of it when the rank is known. The rank prediction adds more uncertainty.
  * Full-vector (dense) time: 6.5e-9 s per unit of dense_work() (sum over intervals of d^n * n * uniformisation
    terms), within x0.8-x1.4 on the H3N2 MCC and posterior trees (Apple M4, one thread); without parameters a cruder
    5e-6 s per state.
"""
import math

import numpy as np

from ._engines import esco_dense, esco_tt
from .api import DENSE_MAX_STATES, ERROR_C, max_lineages

# (d, regime) -> [(lineages alive at once, rank mean, rank min, rank max)] at eps 1e-6
RANK_TABLE = {
    (2, "slow"): [(7.3, 4, 4, 4), (11.7, 4, 4, 4), (16.0, 4, 3, 5), (20.7, 3.7, 3, 4)],
    (2, "medium"): [(7.7, 5.7, 5, 6), (14.0, 8.3, 7, 10), (14.7, 11.3, 8, 15), (20.0, 12.7, 11, 14)],
    (2, "fast"): [(7.7, 6.7, 6, 7), (12.7, 10.7, 9, 13), (14.7, 11, 10, 12), (20.0, 16.3, 15, 19)],
    (3, "slow"): [(9.0, 8, 7, 9), (12.0, 9, 8, 10), (16.7, 8.7, 7, 10), (22.0, 10.7, 7, 16), (28.0, 9, 9, 9)],
    (3, "medium"): [(7.3, 18.3, 17, 20), (12.3, 33.7, 28, 39), (18.3, 49.3, 36, 60), (21.0, 61, 49, 82)],
    (3, "fast"): [(7.0, 11, 10, 13), (12.3, 21.7, 21, 23), (17.7, 26, 23, 32), (23.7, 33.3, 31, 35)],
    (4, "slow"): [(9.3, 12.3, 10, 16), (13.3, 27, 19, 33), (18.0, 18.7, 16, 23), (19.0, 24, 24, 24),
                  (23.3, 24.7, 16, 31), (31.0, 16, 16, 16)],
    (4, "medium"): [(8.7, 38.3, 22, 53), (13.3, 78.7, 74, 87), (19.7, 114.3, 95, 133), (22.0, 125.7, 103, 163),
                    (30.0, 125, 125, 125)],
    (4, "fast"): [(8.3, 17.3, 17, 18), (12.7, 29.3, 28, 31), (19.0, 35.7, 33, 38), (23.0, 42, 38, 46)],
}
REGIMES = ("slow", "medium", "fast")
TIME_FIT = (-10.687, 0.055, 2.141, 1.524, 3.265)
TIME_SPREAD = (0.34, 3.5)   # 5% and 95% of measured / predicted time
DENSE_SEC_PER_STATE = 5e-6
DENSE_SEC_PER_WORK = 6.5e-9
PAIRED_COST = 2.0           # paired run (eps/10, h/2) relative to the main run: 2^0.055 (N_round) * 1.5^1.52 (rank)
UNEVEN_RATES = 100.0        # max/min migration rate above this: rates are uneven, rank between slow and the regime
TARGET_ERROR = 0.02         # recommended eps: largest of 1e-6, 1e-7 whose empirical bound is at most this


def regime(M, Ne):
    """Migration regime of the simulation design: mean migration rate / mean pairwise coalescence rate
    (simulations: 0.1 slow, 1 medium, 10 fast). Returns (name, ratio)."""
    M = np.array(M, float)
    d = len(Ne)
    off = M[~np.eye(d, dtype=bool)]
    ratio = float(off.mean() / np.mean(1.0 / np.array(Ne, float)))
    name = "slow" if ratio < 0.3 else ("medium" if ratio < 3.0 else "fast")
    return name, ratio


def rank_range(d, k, regimes=REGIMES):
    """(mean, low, high, extrapolated) of the max TT rank at eps 1e-6, or None without calibration data.
    With several regimes the mean is their geometric mean and the range spans all of them."""
    out = []
    extrap = False
    for rg in regimes:
        pts = RANK_TABLE.get((d, rg))
        if not pts:
            return None
        x = [p[0] for p in pts]
        if k > x[-1]:
            extrap = True
        mean = float(np.interp(k, x, [p[1] for p in pts]))
        lo = float(np.interp(k, x, [p[2] for p in pts]))
        hi = float(np.interp(k, x, [p[3] for p in pts])) * (1.5 if k > x[-1] else 1.0)
        out.append((mean, lo, hi))
    return (float(np.exp(np.mean(np.log([o[0] for o in out])))), min(o[1] for o in out), max(o[2] for o in out),
            extrap)


def tt_seconds(n_round, k, rank, d):
    a, bn, bk, br, bd = TIME_FIT
    return math.exp(a) * n_round ** bn * k ** bk * rank ** br * d ** bd


def dense_work(tree):
    """Work of one full-vector evaluation: per interval with n > 1 lineages and d^n > 4096 states (uniformisation),
    d^n * n * (Poisson terms + substeps). Smaller intervals use expm_multiply and are cheap."""
    mdl = esco_dense.Model(tree["M"], tree["Ne"])
    d = mdl.d
    n, t, work = 0, None, 0.0
    for ev in sorted(tree["events"], key=lambda e: (e["t"], 0 if e["type"] == "sample" else 1)):
        if t is not None and ev["t"] > t and n > 1 and d ** n > 4096:
            dt = ev["t"] - t
            D, E = esco_dense._counts_D(n, mdl)
            lam, dmax = float((E + D).max()), float(D.max())
            nsub = max(1, math.ceil(max(lam * dt / 40, dmax * dt / 20)))
            work += d ** n * n * (lam * dt + 6 * math.sqrt(lam * dt / nsub + 1) * nsub + nsub)
        t = ev["t"]
        n += 1 if ev["type"] == "sample" else -1
    return work


def estimate(tree):
    """Cost summary for a tree dict (parameters optional). Returns a dict; see format_estimate."""
    d = len(tree["deme_names"]) if "deme_names" in tree else tree["demes"]
    k = max_lineages(tree)
    n_tips = sum(e["type"] == "sample" for e in tree["events"])
    states = d ** k
    r = dict(d=d, k=k, n_tips=n_tips, states=states, dense=states <= DENSE_MAX_STATES,
             dense_mb=8 * states / 1e6, has_params="M" in tree and "Ne" in tree)
    r["dense_seconds"] = DENSE_SEC_PER_STATE * states
    if r["has_params"]:
        r["regime"], r["ratio"] = regime(tree["M"], tree["Ne"])
        tr = dict(tree, demes=d)
        r["n_round"] = esco_tt.count_roundings(tr)
        off = np.array(tree["M"], float)[~np.eye(d, dtype=bool)]
        r["uneven"] = bool(off.max() > UNEVEN_RATES * max(off.min(), 1e-300))
        if r["dense"]:
            r["dense_seconds"] = DENSE_SEC_PER_WORK * dense_work(tr)
        rgs = ("slow", r["regime"]) if r["uneven"] and r["regime"] != "slow" else (r["regime"],)
        rr = rank_range(d, k, rgs)
    else:
        rr = rank_range(d, k)
    r["rank"] = rr
    if rr is not None and r["has_params"]:
        mean, lo, hi, _ = rr
        r["tt_seconds"] = tuple(tt_seconds(r["n_round"], k, x, d) for x in (mean, lo, hi))
    if r["has_params"]:
        n = r["n_round"]
        r["bound"] = {eps: ERROR_C * eps * n for eps in (1e-6, 1e-7)}
        r["eps"] = 1e-6 if r["bound"][1e-6] <= TARGET_ERROR else 1e-7
    if d >= 6:
        r["feasibility"] = ("not feasible with the tensor train: ranks grow quickly with the number of demes and "
                            "escott was only calibrated for d <= 4")
    elif d == 5:
        r["feasibility"] = ("outside the calibrated range (d <= 4): the tensor train may work on small trees, but "
                            "expect large ranks and hours per evaluation")
    else:
        r["feasibility"] = "within the calibrated range (d <= 4)"
    return r


def _dur(s):
    if s < 10:
        return f"{s:.1f} s"
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def format_estimate(r, jobs=1):
    L = []
    L.append(f"state space: d^k = {r['d']}^{r['k']} = {r['states']:,} joint deme states")
    if r["d"] == 1:
        L.append("one deme: the structured coalescent is Kingman's coalescent; every engine is instant")
        return "\n".join(L)
    if r["dense"]:
        L.append(f"full-vector (dense) engine: feasible (limit {DENSE_MAX_STATES:,} states); about "
                 f"{r['dense_mb'] * 8:.0f} MB per process, about {_dur(r['dense_seconds'])} per evaluation "
                 f"({'range x0.8-x1.4' if r['has_params'] else 'rough; give parameters for a better figure'}); "
                 f"exact to floating point. 'auto' uses it.")
    else:
        L.append(f"full-vector (dense) engine: not feasible ({r['states']:.2e} states > {DENSE_MAX_STATES:,}); "
                 f"'auto' uses the tensor train")
    L.append(f"tensor train: {r['feasibility']}")
    if r["has_params"]:
        L.append(f"migration regime at these parameters: {r['regime']} (mean migration rate / mean coalescence "
                 f"rate = {r['ratio']:.3g}; simulations: 0.1 slow, 1 medium, 10 fast)")
        L.append(f"N_round (number of TT roundings per evaluation) = {r['n_round']:,}")
    if r["rank"] is None:
        L.append("expected TT rank: no calibration data for this number of demes")
    else:
        mean, lo, hi, ext = r["rank"]
        if not r["has_params"]:
            which = "all regimes (give parameters to narrow this)"
        elif r["uneven"] and r["regime"] != "slow":
            which = f"rates are uneven, so between the slow and the {r['regime']} regime"
        else:
            which = f"{r['regime']} regime"
        L.append(f"expected max TT rank at eps 1e-6: about {mean:.0f} (range {lo:.0f}-{hi:.0f}; simulated trees, "
                 f"d = {r['d']}, {which})" + ("; extrapolated beyond the largest simulated trees" if ext else ""))
    if "tt_seconds" in r:
        t, tlo, thi = r["tt_seconds"]
        L.append(f"expected TT time per evaluation at eps 1e-6: about {_dur(t)} (plausible range "
                 f"{_dur(tlo * TIME_SPREAD[0])}-{_dur(thi * TIME_SPREAD[1])}); the paired check (eps/10, h/2) adds about "
                 f"x{PAIRED_COST:.1f}")
    if r["has_params"]:
        b6, b7 = r["bound"][1e-6], r["bound"][1e-7]
        L.append(f"empirical TT error bound 0.81*eps*N_round: {b6:.1g} at eps 1e-6, {b7:.1g} at eps 1e-7")
        L.append(f"recommended eps: {r['eps']:g} (largest of 1e-6, 1e-7 with a bound <= {TARGET_ERROR}; "
                 f"do not use 1e-5 for inference)")
    else:
        L.append("N_round, TT time and the recommended eps depend on the parameters: add --params p.json or "
                 "--log beast.log")
    if r["has_params"]:
        per = r["dense_seconds"] if r["dense"] else r.get("tt_seconds", (None,))[0]
        if per is not None:
            chk = per if r["dense"] else per * (1 + PAIRED_COST)
            L.append(f"command time (rough, jobs = {jobs}): check about {_dur(chk)}; reweight about {_dur(per)} per "
                     f"posterior state / jobs; fit or compare about 30-60 exact evaluations per likelihood peak and "
                     f"model (about {_dur(45 * per / jobs)} per peak)")
    return "\n".join(L)
