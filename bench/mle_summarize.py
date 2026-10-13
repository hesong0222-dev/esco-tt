"""Summarize the MLE study (analysis rules of bench/mle_prereg.md).

Input : kaggle/out/mle/<kernel>/results.jsonl  (kernel dir name = <account>-<name>, e.g. <ACCOUNT>-d3k30s-a)
        local runs may be added as kaggle/out/mle/local-<name>/results.jsonl
Output: bench/mle_study.csv (one row per tree), bench/mle_study_summary.csv, markdown tables on stdout.

Rules: per tree and engine keep the highest-logL 'fit'; other stages only if they were built on that fit
(same source file, and base_fit_ll equal when present). Paired comparisons only on trees where both engines'
needed stages exist. l_hat = max logL over every stage of that engine on that fit (guards optimizer misses).
Primary metrics (per setting): Delta = log m_hat(MASCOT) - log m_hat(exact): median + bootstrap 95% CI (10k);
L_loss = logL_exact(theta_exact) - logL_exact(theta_MASCOT): median, fraction > 1.92.
Decision: |median Delta| < 0.05 and CI inside (-0.10, 0.10) -> no practical difference;
          CI excludes 0 and |median Delta| >= 0.10 -> systematic difference; else inconclusive.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mle_common as C
from sim import simulate as S

ROOT = os.path.dirname(HERE)
RNG = np.random.default_rng(20261006)
NBOOT = 10000
SETTINGS = [(3, 30, "slow"), (3, 50, "slow"), (4, 30, "slow"), (4, 50, "slow"), (3, 30, "medium")]


def true_root_deme(tree):
    """Re-run sim.simulate.simulate_tree with the same seed / call sequence and return the MRCA deme."""
    meta = tree["meta"]
    d, k = meta["d"], meta["k"]
    rng = np.random.default_rng(meta["seed"])
    m, Ne = C.true_params(d, meta["regime"])
    M = np.array(m * (1 - np.eye(d)))
    times, demes = S.make_samples(k, d, rng)
    Ne = np.asarray(Ne, float)
    out_rate = M.sum(axis=1) - np.diag(M)
    order = sorted(range(k), key=lambda i: (times[i], i))
    pending = [(float(times[i]), i, int(demes[i])) for i in order]
    active, next_id, t, pi, events = [], k, 0.0, 0, []
    while True:
        while pi < len(pending) and pending[pi][0] <= t:
            ts, sid, sd = pending[pi]
            events.append({"t": ts, "type": "sample", "id": sid, "deme": sd})
            active.append([sid, sd])
            pi += 1
        if pi == len(pending) and len(active) == 1:
            break
        counts = np.bincount([a[1] for a in active], minlength=d) if active else np.zeros(d)
        coal = counts * (counts - 1) / 2.0 / Ne
        mig = np.array([out_rate[a[1]] for a in active]) if active else np.zeros(0)
        total = coal.sum() + mig.sum()
        tns = pending[pi][0] if pi < len(pending) else np.inf
        dt = rng.exponential(1.0 / total) if total > 0 else np.inf
        if t + dt >= tns:
            t = tns
            continue
        t += dt
        u = rng.uniform() * total
        if u < coal.sum():
            a = rng.choice(d, p=coal / coal.sum())
            idx = [i for i, x in enumerate(active) if x[1] == a]
            i, j = rng.choice(len(idx), size=2, replace=False)
            li, lj = active[idx[i]], active[idx[j]]
            events.append({"t": t, "type": "coal", "a": li[0], "b": lj[0], "new": next_id})
            for x in sorted([idx[i], idx[j]], reverse=True):
                active.pop(x)
            active.append([next_id, a])
            next_id += 1
        else:
            i = rng.choice(len(active), p=mig / mig.sum())
            a = active[i][1]
            p = M[a].copy()
            p[a] = 0.0
            active[i][1] = int(rng.choice(d, p=p / p.sum()))
    assert events == tree["events"], "re-simulation does not reproduce the tree"
    return active[0][1]


def planned():
    """Planned trees per setting = union of job lists of all MLE kernels (kaggle/mle_k/*/run.py)."""
    P = defaultdict(set)
    for f in glob.glob(os.path.join(ROOT, "kaggle", "mle_k", "*", "run.py")):
        m = re.search(r'JOBS = "([^"]*)"', open(f).read())
        if m:
            for tag in m.group(1).split(","):
                P[setting_of(tag)].add(tag)
    return P


def setting_of(tag):
    m = re.match(r"d(\d)_k(\d+)_(\w+)_r\d+", tag)
    return int(m.group(1)), int(m.group(2)), m.group(3)


def load():
    recs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "kaggle", "out", "mle", "*", "results.jsonl"))):
        src = os.path.basename(os.path.dirname(f))
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r["_src"] = src
                recs[r["tree"]].append(r)
    return recs


def pick(rs, engine):
    """Best fit of this engine + the stages built on it. Returns dict stage -> record (or None)."""
    fits = [r for r in rs if r["engine"] == engine and r["stage"] == "fit"]
    if not fits:
        return None
    f = max(fits, key=lambda r: r["loglik"])
    out = {"fit": f, "n_fit_runs": len(fits)}
    for r in rs:
        if r["engine"] != engine or r["stage"] == "fit":
            continue
        if r["stage"] == "truth":
            out.setdefault("truth", r)
            continue
        same = (r["base_fit_ll"] == f["loglik"]) if "base_fit_ll" in r else (r["_src"] == f["_src"])
        if same:
            out[r["stage"]] = r
    return out


def per_tree(recs):
    rows = []
    for tag, rs in sorted(recs.items()):
        d, k, regime = setting_of(tag)
        tree = json.load(open(os.path.join(HERE, "mle", "trees", tag + ".json")))
        th0 = np.array(C.true_theta(d, regime))
        p = d + 1
        row = dict(tree=tag, d=d, k=k, regime=regime, rep=tree["meta"]["rep"], true_root=true_root_deme(tree))
        E = {e: pick(rs, e) for e in ("mascot", "tt")}
        row["complete"] = all(E[e] is not None for e in E)
        for e in ("mascot", "tt"):
            P = E[e]
            if P is None:
                continue
            fit = P["fit"]
            row[f"{e}_src"] = fit["_src"]
            row[f"{e}_n_fit_runs"] = P["n_fit_runs"]
            lls = [fit["loglik"]] + [P[s]["loglik"] for s in P if s.startswith("prof_") or s == "eqNe"]
            lhat = max(lls)
            row[f"{e}_ll"] = fit["loglik"]
            row[f"{e}_lhat_gap"] = lhat - fit["loglik"]   # >0 means a restricted fit beat the full fit
            th = np.array(fit["theta"])
            for j, nm in enumerate(C.names(d)):
                row[f"{e}_{nm}"] = th[j]
                row[f"{e}_atbound_{nm}"] = th[j] <= C_BOUNDS[0] + 1e-6 or th[j] >= C_BOUNDS[1] - 1e-6
                pr = P.get(f"prof_{j}")
                row[f"{e}_cov_{nm}"] = (2 * (lhat - pr["loglik"]) <= stats.chi2.ppf(0.95, 1)) if pr else None
            tr = P.get("truth")
            row[f"{e}_cov_joint"] = (2 * (lhat - tr["loglik"]) <= stats.chi2.ppf(0.95, p)) if tr else None
            root = np.array(fit["root"])
            row[f"{e}_root"] = int(np.argmax(root))
            row[f"{e}_root_ptrue"] = root[row["true_root"]]
            row[f"{e}_rootvec"] = root.tolist()
            eq = P.get("eqNe")
            row[f"{e}_rej_eqNe"] = (2 * (lhat - eq["loglik"]) > stats.chi2.ppf(0.95, d - 1)) if eq else None
            row[f"{e}_msg"] = fit.get("msg", "")
            row[f"{e}_nev"] = fit.get("nev")
            row[f"{e}_starts"] = fit.get("start_lls")
            row[f"{e}_used_best_seen"] = fit.get("used_best_seen")
            if e == "mascot":
                cx = P.get("cross")
                row["tt_ll_at_mascot_mle"] = cx["loglik"] if cx else None
            if e == "tt":
                for s in ("fit_flat", "eps7"):
                    if s in P:
                        row[f"val_{s}"] = P[s]
        # B1 (AUDIT_CODE.md, TT underflow) suspect flags: extreme TT optimum, TT >> MASCOT at same params, or any
        # TT stage logL < -700 (the underflow can only occur when one interval's integrated hazard exceeds ~708)
        # The flags target the pre-fix TT copy (mle_fit v1). Records from v2+ use the B1-fixed tt and are not flagged
        # (post-hoc rule, 2026-10-08: the only flagged v2 tree, d3_k50_slow_r21, was checked at eps 1e-7 and h/2 and
        # its TT truth logL moved by <= 2.1e-4; the 8.5 gap to MASCOT is real). INCLUDE_SUSPECT=0 keeps v1 behaviour.
        sus = []
        if E["tt"] is not None and E["tt"]["fit"].get("version", 1) < 2:
            th = np.array(E["tt"]["fit"]["theta"])
            if np.any(np.abs(th - th0) > np.log(20)):
                sus.append("extreme_param")
            if any(r["engine"] == "tt" and r.get("loglik") is not None and r["loglik"] < -700 for r in rs):
                sus.append("tt_ll_below_-700")
            cx = E["tt"].get("cross")
            if cx and E["tt"]["fit"]["loglik"] - cx["loglik"] > 5:
                sus.append("tt_minus_mascot_gt5_at_tt_mle")
            if E["mascot"] is not None:
                cm = E["mascot"].get("cross")
                if cm and cm["loglik"] - E["mascot"]["fit"]["loglik"] > 5:
                    sus.append("tt_minus_mascot_gt5_at_mascot_mle")
                tt_t, m_t = E["tt"].get("truth"), E["mascot"].get("truth")
                if tt_t and m_t and tt_t["loglik"] - m_t["loglik"] > 5:
                    sus.append("tt_minus_mascot_gt5_at_truth")
        row["suspect"] = ";".join(sus)
        if os.environ.get("INCLUDE_SUSPECT", "0") != "1" and sus:
            row["complete"] = False
        if row["complete"]:
            row["L_loss"] = (row["tt_ll"] - row["tt_ll_at_mascot_mle"]) if row.get("tt_ll_at_mascot_mle") is not None else None
            row["Delta_m"] = row["mascot_m"] - row["tt_m"]
            row["tt_move_from_start"] = float(np.linalg.norm(np.array(E["tt"]["fit"]["theta"]) - np.array(E["mascot"]["fit"]["theta"])))
            row["root_tv"] = 0.5 * float(np.abs(np.array(row["mascot_rootvec"]) - np.array(row["tt_rootvec"])).sum())
        rows.append(row)
    return rows


C_BOUNDS = (-9.0, 6.0)


def boot_median(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(np.median(x)), np.nan, np.nan
    b = np.median(x[RNG.integers(0, len(x), (NBOOT, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_rmse(x):
    x = np.asarray(x, float)
    b = np.sqrt(np.mean(x[RNG.integers(0, len(x), (NBOOT, len(x)))] ** 2, axis=1))
    return float(np.sqrt(np.mean(x ** 2))), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return ph, c - h, c + h


def decide(med, lo, hi):
    if abs(med) < 0.05 and lo > -0.10 and hi < 0.10:
        return "no practical difference"
    if (lo > 0 or hi < 0) and abs(med) >= 0.10:
        return "systematic difference"
    return "inconclusive"


def fmt(t, nd=3):
    return f"{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]"


def summarize(rows, plan):
    out = []
    G = defaultdict(list)
    for r in rows:
        G[(r["d"], r["k"], r["regime"])].append(r)
    print("## Primary metrics (pre-registered)\n")
    print("| setting | planned | both engines done | Delta = log m̂_MASCOT − log m̂_exact, median [95% boot CI] | decision | L_loss median | L_loss > 1.92 |")
    print("|---|---|---|---|---|---|---|")
    for s in SETTINGS:
        A = [r for r in G.get(s, []) if r["complete"]]
        npl = len(plan.get(s, []))
        if not A:
            print(f"| d{s[0]} k{s[1]} {s[2]} | {npl} | 0 | – | – | – | – |")
            continue
        dm = boot_median([r["Delta_m"] for r in A])
        L = [r["L_loss"] for r in A if r["L_loss"] is not None]
        print(f"| d{s[0]} k{s[1]} {s[2]} | {npl} | {len(A)} | {fmt(dm)} | {decide(*dm)} | {np.median(L):.3f} | "
              f"{sum(x > 1.92 for x in L)}/{len(L)} |")
        out.append(dict(setting=f"d{s[0]}k{s[1]}{s[2]}", metric="Delta_log_m", n=len(A), median=dm[0], lo=dm[1], hi=dm[2],
                        decision=decide(*dm)))
        out.append(dict(setting=f"d{s[0]}k{s[1]}{s[2]}", metric="L_loss", n=len(L), median=float(np.median(L)),
                        lo=None, hi=None, decision=f"{sum(x > 1.92 for x in L)}/{len(L)} > 1.92"))
    print("\n## Secondary (descriptive): error vs truth on the log scale, paired trees only\n")
    for s in SETTINGS:
        A = [r for r in G.get(s, []) if r["complete"]]
        if not A:
            continue
        d = s[0]
        th0 = C.true_theta(d, s[2])
        print(f"\n### d={d} k={s[1]} {s[2]}  (n={len(A)}; true m={np.exp(th0[0]):.4f}, Ne={np.round(np.exp(th0[1:]), 2).tolist()})\n")
        print("| param | MASCOT median bias [CI] | MASCOT RMSE [CI] | MASCOT LR cov [Wilson] | exact median bias [CI] | exact RMSE [CI] | exact LR cov [Wilson] | n cov pairs | paired MASCOT−exact median [CI] |")
        print("|---|---|---|---|---|---|---|---|---|")
        for j, nm in enumerate(C.names(d)):
            line = [f"log {nm}"]
            pairs = [r for r in A if r[f"mascot_cov_{nm}"] is not None and r[f"tt_cov_{nm}"] is not None]
            for e in ("mascot", "tt"):
                err = np.array([r[f"{e}_{nm}"] for r in A]) - th0[j]
                mb, rm = boot_median(err), boot_rmse(err)
                kc = sum(bool(r[f"{e}_cov_{nm}"]) for r in pairs)
                w = wilson(kc, len(pairs))
                line += [fmt(mb), f"{rm[0]:.3f} [{rm[1]:.3f}, {rm[2]:.3f}]",
                         f"{w[0]:.2f} [{w[1]:.2f}, {w[2]:.2f}]" if pairs else "–"]
                out.append(dict(setting=f"d{s[0]}k{s[1]}{s[2]}", metric=f"{e}_bias_{nm}", n=len(A), median=mb[0],
                                lo=mb[1], hi=mb[2], decision=f"rmse {rm[0]:.3f}; cov {w[0]:.2f} (n={len(pairs)})"))
            dd = boot_median([r[f"mascot_{nm}"] - r[f"tt_{nm}"] for r in A])
            line += [str(len(pairs)), fmt(dd)]
            print("| " + " | ".join(line) + " |")
        jp = [r for r in A if r["mascot_cov_joint"] is not None and r["tt_cov_joint"] is not None]
        for e in ("mascot", "tt"):
            w = wilson(sum(bool(r[f"{e}_cov_joint"]) for r in jp), len(jp))
            rr = wilson(sum(r[f"{e}_root"] == r["true_root"] for r in A), len(A))
            eqp = [r for r in A if r["mascot_rej_eqNe"] is not None and r["tt_rej_eqNe"] is not None]
            rej = wilson(sum(bool(r[f"{e}_rej_eqNe"]) for r in eqp), len(eqp))
            nb = sum(any(r[f"{e}_atbound_{nm}"] for nm in C.names(d)) for r in A)
            print(f"- {e}: joint LR coverage {w[0]:.2f} [{w[1]:.2f}, {w[2]:.2f}] (n={len(jp)}); most-likely root = true root "
                  f"{rr[0]:.2f} [{rr[1]:.2f}, {rr[2]:.2f}]; mean P(true root) {np.mean([r[f'{e}_root_ptrue'] for r in A]):.3f}; "
                  f"equal-Ne LRT rejects {rej[0]:.2f} (n={len(eqp)}); trees with a parameter at a bound: {nb}")
        eqp = [r for r in A if r["mascot_rej_eqNe"] is not None and r["tt_rej_eqNe"] is not None]
        print(f"- agreement: most-likely root {np.mean([r['mascot_root'] == r['tt_root'] for r in A]):.2f}; "
              f"root TV distance median {np.median([r['root_tv'] for r in A]):.3f} (max {np.max([r['root_tv'] for r in A]):.3f}); "
              f"equal-Ne LRT decision {np.mean([r['mascot_rej_eqNe'] == r['tt_rej_eqNe'] for r in eqp]) if eqp else float('nan'):.2f} (n={len(eqp)})")
        # optimizer diagnostics
        ab = sum("ABNORMAL" in (r["tt_msg"] or "") for r in A)
        mi = sum("ITERATIONS REACHED LIMIT" in (r["tt_msg"] or "").upper() for r in A)
        rs = sum(len(r["tt_starts"] or []) > 1 for r in A)
        ub = sum(bool(r.get("tt_used_best_seen")) for r in A)
        ms = [r["mascot_starts"] for r in A if r.get("mascot_starts") and len(r["mascot_starts"]) == 2]
        msd = max(abs(a - b) for a, b in ms) if ms else float("nan")
        gaps = [r[f"{e}_lhat_gap"] for r in A for e in ("mascot", "tt")]
        print(f"- optimizer: TT abnormal line-search termination {ab}/{len(A)}, iteration limit {mi}/{len(A)}, "
              f"truth restarts {rs}/{len(A)}, best-seen point used {ub}/{len(A)}; median TT nev {np.median([r['tt_nev'] for r in A]):.0f}; "
              f"TT moved from MASCOT start median {np.median([r['tt_move_from_start'] for r in A]):.3f}; MASCOT truth-vs-flat start "
              f"max |Δll| {msd:.1e} over {len(ms)} trees; restricted fit beat full fit: {sum(g > 1e-9 for g in gaps)} "
              f"(max {max(gaps):.3g})")
        V = [r for r in A if "val_fit_flat" in r or "val_eps7" in r]
        if V:
            for r in V:
                ff, e7 = r.get("val_fit_flat"), r.get("val_eps7")
                s1 = (f"flat-start TT fit: Δll {ff['loglik'] - r['tt_ll']:+.4f}, |Δθ|max "
                      f"{np.max(np.abs(np.array(ff['theta']) - np.array([r[f'tt_{n}'] for n in C.names(d)]))):.3f}") if ff else ""
                s2 = (f"eps1e-7: L_loss {e7['loglik'] - e7['ll_at_mascot_mle']:.4f} vs eps1e-6 {r['L_loss']:.4f}, "
                      f"ll(1e-7)−ll(1e-6) at TT MLE {e7['loglik'] - e7['ll6_at_tt_mle']:+.4f}") if e7 else ""
                print(f"  - validation {r['tree']}: {s1}; {s2}")
    return out


def main():
    rows = per_tree(load())
    plan = planned()
    if not rows:
        print("no results yet")
        return
    keys = sorted({k for r in rows for k in r if not k.startswith("val_") and not k.endswith("rootvec")})
    with open(os.path.join(HERE, "mle_study.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("## Completion and sources\n")
    for s in SETTINGS:
        A = [r for r in rows if (r["d"], r["k"], r["regime"]) == s]
        srcs = defaultdict(int)
        for r in A:
            if r["complete"]:
                srcs[r["tt_src"].split("-")[0]] += 1
        print(f"- d{s[0]} k{s[1]} {s[2]}: B1-suspect trees (excluded) {[r['tree'] + ':' + r['suspect'] for r in A if r['suspect']]}")
        print(f"- d{s[0]} k{s[1]} {s[2]}: planned {len(plan.get(s, []))}, with results {len(A)}, both engines fitted "
              f"{sum(r['complete'] for r in A)}; TT fits by account {dict(srcs)}")
    print()
    out = summarize(rows, plan)
    with open(os.path.join(HERE, "mle_study_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "metric", "n", "median", "lo", "hi", "decision"])
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
