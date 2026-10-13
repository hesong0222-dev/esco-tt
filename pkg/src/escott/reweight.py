"""Importance reweighting of a BEAST 2 MASCOT posterior sample to the exact structured-coalescent posterior.

The chain's states (tree + parameters) are a sample from the density the chain targeted, whose tree part is the MASCOT
density BEAST logged (e.g. the 'Mascot' column). Priors and the sequence likelihood are the same under both models,
so the exact posterior is the same sample with weights
    w_i = exp(l_exact,i - l_logged,i).
If the log has no such column (or with denominator='converged'), escott's converged MASCOT likelihood is used as the
denominator instead. Per-state results are appended to a CSV as they finish, so an interrupted run can be resumed.
"""
import csv
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import api
from .io import (_parse_newick, events_from_root, mascot_columns, params_from_row, read_beast_log, read_trees,
                 sample_column)

ESS_UNRELIABLE = 0.1    # weight ESS / n below this (or ESS < ESS_MIN): reweighting is not trustworthy
ESS_MIN = 50
ESS_CAUTION = 0.5
EXTRA_COLUMNS = re.compile(r"height|clock", re.I)   # non-model log columns that are also summarised


# ----------------------------------------------------------------------------- input


def load_states(trees_path, log_path, locations=None, location_key=None, label_field=-1, sep="|", burnin=0.1,
                thin=1):
    """Pair log rows with trees STATE_<Sample>. Returns (deme names, states, info); each state is a dict with
    'sample', 'tree' (escott tree dict with that row's parameters), 'row' (the log row)."""
    hdr, rows = read_beast_log(log_path)
    names, ne_cols, mig_cols = mascot_columns(hdr, log_path)
    scol = sample_column(hdr, log_path)
    lcol = logged_density_column(hdr)
    trans, trees = read_trees(trees_path)
    by_name = {}
    for name, nwk in trees:
        m = re.match(r"STATE_(\d+)$", name)
        by_name[m.group(1) if m else name] = nwk
    if not 0 <= burnin < 1:
        raise ValueError("--burnin is a fraction of the logged states, 0 <= burnin < 1")
    if thin < 1:
        raise ValueError("--thin must be >= 1")
    n_burn = int(math.floor(burnin * len(rows)))
    kept = rows[n_burn:][::thin]
    missing = [r[scol] for r in kept if r[scol] not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} logged state(s) have no tree STATE_<n> in {trees_path} (e.g. state "
                         f"{missing[0]}); the .trees and .log files must come from the same run")
    states, source = [], None
    for r in kept:
        root = _parse_newick(by_name[r[scol]].strip())
        ev, _, _, source = events_from_root(root, trans, locations, location_key, label_field, sep, names)
        M, Ne = params_from_row(r, names, ne_cols, mig_cols)
        states.append({"sample": r[scol], "row": r, "logged": r[lcol] if lcol else "",
                       "tree": {"demes": len(names), "deme_names": names, "M": M, "Ne": Ne, "events": ev}})
    info = {"n_log": len(rows), "n_burnin": n_burn, "n_kept": len(states), "location_source": source,
            "header": hdr, "ne_cols": ne_cols, "mig_cols": mig_cols, "logged_col": lcol}
    return names, states, info


def logged_density_column(header):
    """The column holding the MASCOT tree density BEAST computed during the run (the chain's target), e.g. 'Mascot';
    None if the log has none."""
    for c in header:
        if c.lower().startswith("mascot") and "." not in c:
            return c
    return None


# ----------------------------------------------------------------------------- per-state evaluation


def fields(names):
    f = ["sample", "peak_lineages", "engine", "eps", "exact", "exact_error", "mascot_converged", "mascot_logged"]
    f += [f"root_exact_{a}" for a in names] + [f"root_mascot_{a}" for a in names]
    return f + ["seconds"]


def evaluate_state(sample, tree, eps=1e-6, paired=True):
    """Exact and converged-MASCOT log-likelihood and root distributions of one posterior state (a CSV row dict)."""
    names = tree["deme_names"]
    t0 = time.time()
    method = api.exact_method(tree)
    ex = api.loglik(tree, method=method, eps=eps, paired=paired)
    ma = api.loglik(tree, method="mascot")
    row = {"sample": sample, "peak_lineages": api.max_lineages(tree), "engine": method,
           "eps": eps if method == "tt" else "", "exact": repr(float(ex.loglik)),
           "exact_error": repr(float(ex.error)), "mascot_converged": repr(float(ma.loglik)), "seconds": round(time.time() - t0, 2)}
    for a, x, y in zip(names, ex.info["root_dist"] or [math.nan] * len(names),
                       ma.info["root_dist"] or [math.nan] * len(names)):
        row[f"root_exact_{a}"], row[f"root_mascot_{a}"] = repr(float(x)), repr(float(y))
    return row


def _worker(args):
    return evaluate_state(*args)


def run(states, names, out_csv, eps=1e-6, paired=True, jobs=1, resume=False, progress=None):
    """Evaluate all states (skipping those already in out_csv when resume=True); returns the list of CSV rows."""
    flds = fields(names)
    done = {}
    if os.path.exists(out_csv):
        if not resume:
            raise FileExistsError(f"{out_csv} exists: pass --resume to continue it, or remove it / choose --out")
        with open(out_csv) as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != flds:
                raise ValueError(f"{out_csv} was written for other demes or by another version (columns differ)")
            for r in rd:
                done[r["sample"]] = r
        eps_old = {r["eps"] for r in done.values() if r["eps"]}
        if eps_old and eps_old != {repr(eps)} and eps_old != {str(eps)}:
            raise ValueError(f"{out_csv} holds tensor-train results at eps {sorted(eps_old)}, not {eps}")
    todo = [s for s in states if s["sample"] not in done]
    if progress:
        progress(f"{len(states)} states to evaluate, {len(states) - len(todo)} already in {out_csv}, "
                 f"{len(todo)} to go ({jobs} process{'es' if jobs > 1 else ''})")
    new = not os.path.exists(out_csv)
    rows = dict(done)
    t0 = time.time()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        if new:
            w.writeheader()
            f.flush()
        if todo:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                logged = {s["sample"]: s["logged"] for s in todo}
                futs = [ex.submit(_worker, (s["sample"], s["tree"], eps, paired)) for s in todo]
                for i, fu in enumerate(as_completed(futs), 1):
                    r = fu.result()
                    r["mascot_logged"] = logged[r["sample"]]
                    w.writerow(r)
                    f.flush()
                    rows[r["sample"]] = r
                    if progress:
                        el = time.time() - t0
                        eta = el / i * (len(todo) - i)
                        progress(f"[{i}/{len(todo)}] state {r['sample']}: {r['peak_lineages']} lineages, "
                                 f"{r['engine']}, {r['seconds']} s, exact {float(r['exact']):.4f}, MASCOT - exact "
                                 f"{float(r['mascot_converged']) - float(r['exact']):+.4f}; about {eta / 60:.0f} min left")
    return [rows[s["sample"]] for s in states]


# ----------------------------------------------------------------------------- summary


def kish(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def hpd(x, w, mass=0.95):
    """Shortest interval holding `mass` of the weighted sample."""
    o = np.argsort(x)
    xs, ws = np.asarray(x, float)[o], np.asarray(w, float)[o] / np.sum(w)
    c = np.concatenate([[0], np.cumsum(ws)])
    best = (xs[0], xs[-1])
    j = 0
    for i in range(len(xs)):
        while j < len(xs) and c[j + 1] - c[i] < mass:
            j += 1
        if j >= len(xs):
            break
        if xs[j] - xs[i] < best[1] - best[0]:
            best = (xs[i], xs[j])
    return float(best[0]), float(best[1])


def summarize(l_exact, l_denom, quantities, roots_exact, roots_mascot, names, errors=None):
    """Weights exp(l_exact - l_denom), ESS and posterior summaries. l_denom: the density the chain sampled from.
    quantities: {name: values per state}; roots_*: n x d arrays."""
    lE, lM = np.asarray(l_exact, float), np.asarray(l_denom, float)
    lw = lE - lM
    w = np.exp(lw - lw.max())
    n = len(w)
    ess = kish(w)
    s = {"n": n, "ess": ess, "ess_fraction": ess / n, "max_weight": float(w.max() / w.sum()),
         "logw_median": float(np.median(lw)), "logw_min": float(lw.min()), "logw_max": float(lw.max()),
         "logw_sd": float(np.std(lw)), "max_error": float(np.nanmax(errors)) if errors is not None else None}
    one = np.ones(n)
    s["params"] = []
    for q, x in quantities.items():
        x = np.asarray(x, float)
        mM, mE = float(x.mean()), float((w * x).sum() / w.sum())
        sdM = float(x.std())
        s["params"].append({"name": q, "mascot_mean": mM, "mascot_hpd": hpd(x, one), "exact_mean": mE,
                            "exact_hpd": hpd(x, w), "shift": mE / mM - 1 if mM != 0 else math.nan,
                            "shift_sd": (mE - mM) / sdM if sdM > 0 else math.nan})
    RE, RM = np.asarray(roots_exact, float), np.asarray(roots_mascot, float)
    wn = w / w.sum()
    s["root"] = {"mascot": RM.mean(axis=0).tolist(), "exact_weights_mascot_root": (wn @ RM).tolist(),
                 "exact": (wn @ RE).tolist(), "names": list(names)}
    if s["ess_fraction"] < ESS_UNRELIABLE:
        s["verdict"] = (f"WARNING: reweighting is not reliable: the weight ESS is only {100 * s['ess_fraction']:.0f}% "
                        f"of the states. Run an exact-likelihood MCMC instead of trusting these reweighted summaries.")
    elif ess < ESS_MIN:
        s["verdict"] = (f"WARNING: too few effective states (ESS {ess:.0f} < {ESS_MIN}) for reliable summaries. Use "
                        f"more states (less thinning); if the ESS stays low, run an exact-likelihood MCMC.")
    elif s["ess_fraction"] < ESS_CAUTION or s["max_weight"] > 0.1:
        s["verdict"] = ("Caution: the weights are uneven; reweighted summaries rest on few states. Treat them as "
                        "indicative and consider an exact-likelihood MCMC.")
    else:
        s["verdict"] = ("The weights are even: the exact posterior is well represented by the BEAST sample, and the "
                        "reweighted summaries can be trusted within the support caveat below.")
    return s


def summarize_rows(rows, states, info, names, denominator="logged"):
    """denominator 'logged': the log's MASCOT density column (the chain's target); 'converged': escott's converged
    MASCOT likelihood. 'logged' falls back to 'converged', with a warning, when the log has no such column."""
    def col(r, k):
        return float(r[k])
    lE = [col(r, "exact") for r in rows]
    note = None
    if denominator == "logged" and info.get("logged_col") and all(r.get("mascot_logged") for r in rows):
        lM = [col(r, "mascot_logged") for r in rows]
        how = f"weights = exact likelihood / the chain's target density (log column '{info['logged_col']}')"
    else:
        lM = [col(r, "mascot_converged") for r in rows]
        how = "weights = exact likelihood / escott's converged MASCOT likelihood"
        if denominator == "logged":
            denominator = "converged"
            note = ("WARNING: the log has no MASCOT density column, so the weights use escott's converged MASCOT "
                    "likelihood instead: an approximate reweighting that may differ from the chain's target.")
        else:
            how += " (--denominator converged)"
    err = [col(r, "exact_error") for r in rows]
    q = {}
    for a in names:
        q[f"Ne.{a}"] = [float(s["row"][info["ne_cols"][a]]) for s in states]
    for (a, b), c in info["mig_cols"].items():
        q[f"m.{a}_to_{b}"] = [float(s["row"][c]) for s in states]
    for c in info["header"]:
        if EXTRA_COLUMNS.search(c) and c not in q:
            try:
                q[c] = [float(s["row"][c]) for s in states]
            except ValueError:
                pass
    RE = [[col(r, f"root_exact_{a}") for a in names] for r in rows]
    RM = [[col(r, f"root_mascot_{a}") for a in names] for r in rows]
    s = summarize(lE, lM, q, RE, RM, names, err)
    s["denominator"], s["denominator_text"], s["denominator_warning"] = denominator, how, note
    s["engines"] = {e: sum(r["engine"] == e for r in rows) for e in ("dense", "tt")}
    return s


def format_summary(s, info, thin):
    L = []
    L.append(f"states: {info['n_log']} in the log; burn-in {info['n_burnin']} dropped; thin {thin} -> {s['n']} "
             f"used (exact engine: {s['engines']['dense']} dense, {s['engines']['tt']} tensor train)")
    L.append(s["denominator_text"])
    if s["denominator_warning"]:
        L.append(s["denominator_warning"])
    L.append(f"log-weight l_exact - l_denominator: median {s['logw_median']:+.3f}, range {s['logw_min']:+.3f} .. "
             f"{s['logw_max']:+.3f}, SD {s['logw_sd']:.3f}")
    L.append(f"weight ESS: {s['ess']:.1f} of {s['n']} states ({100 * s['ess_fraction']:.0f}%); largest normalised "
             f"weight {s['max_weight']:.3f}")
    if s["max_error"]:
        L.append(f"largest per-state error estimate of the exact value: {s['max_error']:.1g} log units "
                 f"(moves a weight by about {100 * s['max_error']:.2g}% at most)")
    L.append("")
    L.append(f"{'parameter':<34s} {'sample mean':>11s} {'sample 95% HPD':>21s} {'exact mean':>11s} "
             f"{'exact 95% HPD':>21s} {'shift':>7s} {'shift/SD':>8s}")
    for p in s["params"]:
        h1 = f"{p['mascot_hpd'][0]:.3g} - {p['mascot_hpd'][1]:.3g}"
        h2 = f"{p['exact_hpd'][0]:.3g} - {p['exact_hpd'][1]:.3g}"
        L.append(f"{p['name']:<34s} {p['mascot_mean']:11.4g} {h1:>21s} {p['exact_mean']:11.4g} {h2:>21s} "
                 f"{100 * p['shift']:+6.1f}% {p['shift_sd']:+8.3f}")
    L.append("")
    r = s["root"]
    w = max(12, max(len(a) for a in r["names"]) + 1)
    L.append("root deme probability (posterior mean of the per-state root distribution):")
    L.append(f"  {'':<38s}" + "".join(f"{a:>{w}s}" for a in r["names"]))
    for lab, key in (("BEAST sample, MASCOT root", "mascot"),
                     ("exact weights, MASCOT root", "exact_weights_mascot_root"),
                     ("exact posterior, exact root", "exact")):
        L.append(f"  {lab:<38s}" + "".join(f"{x:>{w}.3f}" for x in r[key]))
    L.append("")
    L.append(s["verdict"])
    L.append("Support caveat: reweighting only moves weight among the states the MASCOT chain visited; it cannot "
             "find regions the MASCOT posterior misses. The ESS above counts weight unevenness only, not chain "
             "autocorrelation.")
    return "\n".join(L)
