"""Exact ESCO log-likelihood at every logged state of the Taming-the-BEAST H3N2 MASCOT run (tree + parameters).

For each of the 401 states of data/raw/H3N2.log (tree STATE_<n> of data/raw/H3N2.trees):
  exact  : dense (peak lineages <= DENSE_MAX) or TT eps 1e-6 (otherwise), with root-deme distribution
  mascot : approx/mascot.py (converged DOP853), with root-deme distribution
  sisco, lisco : approx/independence.py
Appends one row per state to bench/h3n2_posterior_eval.csv (resumable: states already in the CSV are skipped).
Usage: OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python bench/h3n2_posterior_eval.py [stride]
Analysis: bench/h3n2_posterior_reweight.py
"""
import csv
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "data")]
from convert import parse_newick, read_nexus, to_events  # noqa: E402
from beast_io import load_log, params, DEMES  # noqa: E402
import realdata_fit as rf  # noqa: E402
from approx import independence as ind  # noqa: E402
from dense import esco_dense  # noqa: E402
from tt import esco_tt  # noqa: E402

DENSE_MAX = 15
OUT = os.path.join(HERE, "h3n2_posterior_eval.csv")
FIELDS = ["sample", "peak", "engine", "exact", "mascot", "sisco", "lisco", "beast_logged",
          "root_exact_HK", "root_exact_NY", "root_exact_NZ", "root_mascot_HK", "root_mascot_NY", "root_mascot_NZ",
          "t_exact", "tt_rank"] + [f"Ne_{a}" for a in DEMES] + [f"m_{a}_to_{b}" for a in DEMES for b in DEMES if a != b]


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    trans, trees = read_nexus(os.path.join(ROOT, "data", "raw", "H3N2.trees"))
    trees = dict(trees)
    log = load_log(os.path.join(ROOT, "data", "raw", "H3N2.log"))
    done = set()
    if os.path.exists(OUT):
        done = {r["sample"] for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        # cheapest first so a partial run already covers most states
        todo = []
        for row in log[::stride]:
            if row["Sample"] in done:
                continue
            ev, _, _ = to_events(parse_newick(trees[f"STATE_{row['Sample']}"]), trans)
            M, Ne = params(row)
            tree = {"demes": 3, "M": M, "Ne": Ne, "events": ev}
            todo.append((esco_dense.max_lineages(tree), row, tree))
        todo.sort(key=lambda x: x[0])
        for peak, row, tree in todo:
            t0 = time.time()
            rank = ""
            if peak <= DENSE_MAX:
                ex, root = rf.dense_root(tree)
                eng = "dense"
            else:
                r = esco_tt.loglik_detail(tree, eps=1e-6)
                ex, root, eng, rank = r["loglik"], r["root_dist"], "tt1e-6", r["max_rank"]
                s = sum(root)
                root = [x / s for x in root]
            te = time.time() - t0
            ma, mroot = rf.mascot_root(tree)
            out = {"sample": row["Sample"], "peak": peak, "engine": eng, "exact": ex, "mascot": ma,
                   "sisco": ind.loglik(tree, "sisco"), "lisco": ind.loglik(tree, "lisco"),
                   "beast_logged": float(row["Mascot"]), "t_exact": round(te, 2), "tt_rank": rank}
            for a, x, y in zip(("HK", "NY", "NZ"), root, mroot):
                out[f"root_exact_{a}"], out[f"root_mascot_{a}"] = x, y
            for a in DEMES:
                out[f"Ne_{a}"] = row[f"NeConstant.{a}"]
                for b in DEMES:
                    if a != b:
                        out[f"m_{a}_to_{b}"] = row[f"b_migrationConstant.{a}_to_{b}"]
            w.writerow(out)
            f.flush()
            print(f"{row['Sample']:>9s} peak {peak} {eng} {te:6.1f}s exact {ex:.5f} mascot-exact {ma - ex:+.4f}", flush=True)


if __name__ == "__main__":
    main()
