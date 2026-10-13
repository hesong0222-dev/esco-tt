"""escott: exact structured-coalescent likelihood of a fixed time tree, and checks of MASCOT.

commands:
  estimate  cost of an exact evaluation before running one (state space, TT rank, time, recommended eps)
  check     exact vs MASCOT log-likelihood of one tree at given parameters
  reweight  reweight a BEAST 2 MASCOT posterior sample (.trees + .log) to the exact posterior
  fit       maximum-likelihood fit (MASCOT multistart, exact refinement of every peak, cross-check)
  compare   likelihood-ratio test of symmetric vs asymmetric migration under both engines
"""
import argparse
import json
import os
import sys
import time

from . import __version__
from ._env import Progress, accelerate_warning
from .api import loglik, max_lineages
from .io import load_tree, params_from_beast_log, params_from_json, read_locations_csv


def risk_note(diff, err):
    """Plain-language reading of diff = MASCOT - exact (log units). err: the exact method's error estimate
    (None for the full-vector engine, which is exact up to floating point)."""
    a = abs(diff)
    err_txt = "floating-point level" if err is None else f"{err:.1g}"
    err = 1e-8 if err is None else err
    side = "higher" if diff > 0 else "lower"
    if a <= 3 * err:
        return (f"MASCOT and the exact likelihood agree to within the exact method's estimated numerical error "
                f"({err_txt}). No sign of the risky regime at these parameters.")
    head = (f"MASCOT is {a:.3g} log units {side} than the exact likelihood here; this is larger than the exact "
            f"method's estimated numerical error ({err_txt}), so the gap is due to MASCOT's approximation.")
    if a < 0.1:
        tail = "The gap is small; MASCOT is very likely adequate for this tree."
    elif a < 1.0:
        tail = ("The gap is moderate. It is unlikely to change conclusions on its own, but it matters for "
                "comparisons decided by fractions of a log unit.")
    else:
        tail = ("The gap is large: a likelihood-ratio test at 5% with 1 degree of freedom turns on a 1.92 log-unit "
                "difference, so a gap of this size can change model comparisons. Re-fit with the exact likelihood "
                "before relying on MASCOT here.")
    return (head + " " + tail + " Note: this compares the two likelihoods at one parameter value; it does not by "
            "itself show that MASCOT's parameter estimates are biased.")


# ----------------------------------------------------------------------------- shared options


def _tree_options(c, tree_help="NEXUS or Newick time tree (or escott JSON tree)"):
    c.add_argument("tree", help=tree_help)
    c.add_argument("--tree-name", default="0", help="tree name or index in a NEXUS file (default: first)")
    _location_options(c)


def _location_options(c):
    c.add_argument("--locations", help="CSV/TSV: tip label, deme name")
    c.add_argument("--location-key", help="tip annotation holding the deme, e.g. 'max' in a MASCOT summary tree "
                                          "(default: 'type' or 'location' if every tip has one)")
    c.add_argument("--label-field", type=int, default=-1,
                   help="otherwise: field of the tip label holding the deme (default -1 = last)")
    c.add_argument("--sep", default="|", help="tip label field separator (default '|')")


def _param_options(c, required):
    c.add_argument("--params", help='JSON: {"demes": [...], "Ne": {...}, "M": {"A_to_B": rate, ...}}')
    c.add_argument("--log", help="BEAST 2 MASCOT trace log (constant Ne and migration)" +
                   ("" if required else "; parameters from one row"))
    c.add_argument("--log-row", default="-1", help="row of the log to use (index; default -1 = last)")


def _eps_options(c):
    c.add_argument("--eps", type=float, default=1e-6, help="tensor-train rounding tolerance (default 1e-6)")
    c.add_argument("--paired-check", action=argparse.BooleanOptionalAction, default=True,
                   help="for tensor-train values, repeat the run at eps/10 and half the time step and report the "
                        "difference as an error estimate (default: on)")


def _load(a, params=None):
    locations = read_locations_csv(a.locations) if a.locations else None
    kw = {} if a.tree.endswith(".json") else dict(tree=a.tree_name, locations=locations, location_key=a.location_key,
                                                  label_field=a.label_field, sep=a.sep)
    return load_tree(a.tree, params, **kw)


def _params(a, required=True):
    if a.params:
        return params_from_json(a.params), None
    if a.log:
        names, M, Ne, sample = params_from_beast_log(a.log, a.log_row)
        return (names, M, Ne), sample
    if required:
        raise SystemExit(f"escott {a.cmd}: give --params params.json or --log beast_mascot.log")
    return None, None


def _header(a, tree):
    names = tree.get("deme_names") or [str(i) for i in range(len(tree["Ne"]))]
    n_tips = sum(e["type"] == "sample" for e in tree["events"])
    print(f"tree: {a.tree}  ({n_tips} tips, {len(names)} demes, at most {max_lineages(tree)} lineages at once)")
    print("demes: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)) +
          (f"   (tip locations from the {tree['location_source']})" if tree.get("location_source") else ""), flush=True)
    return names


# ----------------------------------------------------------------------------- commands


def _check(a):
    params, sample = _params(a)
    tree = _load(a, params)
    names = _header(a, tree)
    if sample is not None:
        print(f"parameters: {a.log} row Sample={sample}")
    print("Ne: " + "  ".join(f"{x:.4g}" for x in tree["Ne"]))
    print("M (backward rate row->col):")
    for n, row in zip(names, tree["M"]):
        print(f"  {n:>14s}: " + "  ".join(f"{x:.4g}" for x in row))
    t0 = time.time()
    ex = loglik(tree, method=a.method, eps=a.eps, paired=a.paired_check)
    t1 = time.time()
    ma = loglik(tree, method="mascot")
    t2 = time.time()
    how = "full vector, floating-point accuracy" if ex.method == "dense" else \
        f"tensor train, eps {a.eps:g}, max rank {ex.info['max_rank']}"
    err = ex.error if ex.method == "tt" else None
    est = "" if err is None else f"; est. error {err:.1g}"
    print(f"exact  log L = {ex.loglik:.6f}   ({how}{est}; {t1 - t0:.1f} s)")
    if ex.method == "tt":
        if a.paired_check:
            print(f"         error estimated from the paired run (eps/10, h/2): {ex.info['paired_diff']:.1g}; "
                  f"empirical bound 0.81*eps*N_round: {ex.info['error_bound_empirical']:.1g} "
                  f"(N_round {ex.info['n_round']:,}); the larger is used")
        else:
            print(f"         error = empirical bound 0.81*eps*N_round (N_round {ex.info['n_round']:,}); "
                  f"no paired run")
    print(f"MASCOT log L = {ma.loglik:.6f}   (MASCOT ODE solved to rtol 1e-10; {t2 - t1:.1f} s)")
    print(f"MASCOT - exact = {ma.loglik - ex.loglik:+.6f}")
    print()
    print(risk_note(ma.loglik - ex.loglik, err))


def _estimate(a):
    from .estimate import estimate, format_estimate
    params, sample = _params(a, required=False)
    tree = _load(a, params)
    _header(a, tree)
    if sample is not None:
        print(f"parameters: {a.log} row Sample={sample}")
    print(format_estimate(estimate(tree), a.jobs))


def _reweight(a):
    from . import reweight as rw
    prog = Progress(a.quiet)
    locations = read_locations_csv(a.locations) if a.locations else None
    names, states, info = rw.load_states(a.trees, a.log, locations, a.location_key, a.label_field, a.sep, a.burnin,
                                         a.thin)
    if not states:
        raise SystemExit("escott reweight: no states left after burn-in and thinning")
    out = a.out or os.path.splitext(a.log)[0] + ".escott-reweight.csv"
    print(f"trees: {a.trees}; log: {a.log}; demes: {', '.join(names)} (tip locations from the "
          f"{info['location_source']})")
    print(f"per-state results: {out}")
    rows = rw.run(states, names, out, a.eps, a.paired_check, a.jobs, a.resume, prog)
    s = rw.summarize_rows(rows, states, info, names, a.denominator)
    print(rw.format_summary(s, info, a.thin))


def _fit(a):
    from .fit import fit, format_fit
    tree = _load(a)
    names = _header(a, tree)
    prog = Progress(a.quiet)
    res = fit(tree, a.model, a.starts, a.engine, a.eps, a.jobs, a.seed, a.max_iter, a.paired_check, progress=prog)
    print(format_fit(res, names))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)


def _compare(a):
    from .fit import Model, fit, format_compare, format_fit
    tree = _load(a)
    names = _header(a, tree)
    prog = Progress(a.quiet)
    fs = fit(tree, "sym", a.starts, "both", a.eps, a.jobs, a.seed, a.max_iter, a.paired_check, progress=prog)
    best_sym = max(fs["mascot_peaks"], key=lambda p: p["loglik"])
    ms, ma = Model("sym", names), Model("asym", names)
    start = ma.pack(*ms.unpack(best_sym["theta"]))
    fa = fit(tree, "asym", a.starts, "both", a.eps, a.jobs, a.seed, a.max_iter, a.paired_check,
             extra_starts=[start], progress=prog)
    print(format_fit(fs, names))
    print()
    print(format_fit(fa, names))
    print()
    txt, tm, te = format_compare(fs, fa, names)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"sym": fs, "asym": fa, "lrt_mascot": tm, "lrt_exact": te}, f, indent=1, default=str)


def main(argv=None):
    p = argparse.ArgumentParser(prog="escott", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--version", action="version", version=f"escott {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="exact vs MASCOT log-likelihood of one tree at given parameters")
    _tree_options(c)
    _param_options(c, required=True)
    c.add_argument("--method", default="auto", choices=["auto", "dense", "tt"], help="exact engine (default auto)")
    _eps_options(c)

    c = sub.add_parser("estimate", help="cost of an exact evaluation, before running one")
    _tree_options(c)
    _param_options(c, required=False)
    c.add_argument("--jobs", type=int, default=1, help="processes you plan to use (for the time estimates)")

    c = sub.add_parser("reweight", help="reweight a BEAST 2 MASCOT posterior sample to the exact posterior")
    c.add_argument("trees", help="BEAST 2 .trees file (trees STATE_<n>)")
    c.add_argument("log", help="BEAST 2 MASCOT .log file of the same run (columns Sample, NeConstant.*, "
                               "b_migrationConstant.*_to_*)")
    _location_options(c)
    c.add_argument("--burnin", type=float, default=0.1, help="fraction of logged states to drop (default 0.1)")
    c.add_argument("--thin", type=int, default=1, help="use every N-th state after burn-in (default 1)")
    _eps_options(c)
    c.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
    c.add_argument("--resume", action="store_true", help="continue an interrupted run from its CSV")
    c.add_argument("--out", help="per-state CSV (default: <log>.escott-reweight.csv)")
    c.add_argument("--denominator", choices=("logged", "converged"), default="logged",
                   help="density in the weight denominator: 'logged' = the log's MASCOT density column, the chain's "
                        "target (default); 'converged' = escott's converged MASCOT likelihood")
    c.add_argument("--quiet", action="store_true", help="no progress messages")

    for name, hlp in (("fit", "maximum-likelihood fit with MASCOT and the exact likelihood"),
                      ("compare", "LRT of symmetric vs asymmetric migration under MASCOT and the exact likelihood")):
        c = sub.add_parser(name, help=hlp)
        _tree_options(c)
        if name == "fit":
            c.add_argument("--model", required=True, choices=["sym", "asym"],
                           help="sym: one rate per pair of demes; asym: one rate per direction")
            c.add_argument("--engine", default="both", choices=["exact", "mascot", "both"],
                           help="mascot: MASCOT multistart only; exact: plus exact refinement of every peak and "
                                "the MASCOT cross-check; both (default): exact plus MASCOT-vs-exact parameter table")
        c.add_argument("--starts", type=int, default=10, help="random starts of the MASCOT multistart (default 10)")
        c.add_argument("--max-iter", type=int, default=12, help="exact refinement iterations per peak (default 12)")
        c.add_argument("--seed", type=int, default=1, help="random seed of the starts (default 1)")
        _eps_options(c)
        c.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
        c.add_argument("--out", help="write all results as JSON")
        c.add_argument("--quiet", action="store_true", help="no progress messages")

    a = p.parse_args(argv)
    w = accelerate_warning()
    if w:
        print(w, file=sys.stderr)
    try:
        {"check": _check, "estimate": _estimate, "reweight": _reweight, "fit": _fit, "compare": _compare}[a.cmd](a)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        raise SystemExit(f"escott {a.cmd}: error: {e}")


if __name__ == "__main__":
    main()
