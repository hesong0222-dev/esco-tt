"""SISCO / LISCO / MASCOT vs exact on every tree with an exact reference value.

Sets
  small  : 239 of the 243 sim trees, exact = dense (bench/dense_vs_mascot_openblas.csv; 11 peak-13/14 trees from
           kaggle/dense_big.csv; 4 d4 k20 trees with peak 15-16 have no dense value and are skipped)
  large  : 53 sim trees, exact = TT eps 1e-6 (bench/large_sweep.csv, column tt; paired error err_h2)
  h3n2   : fit points of bench/realdata.csv (MCC + 4 posterior trees), exact = dense, recomputed here
  aiv    : AIV host-order (3 demes, masco MCC tree) points of bench/realdata2.csv, exact = TT (from the CSV)
MASCOT is recomputed with approx/mascot.py everywhere and checked against the stored value.
Output: bench/sisco_eval.csv.  Usage: OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python bench/sisco_eval.py
"""
import csv
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
from approx import independence as ind  # noqa: E402
from approx.mascot import loglik as mascot  # noqa: E402
from dense import esco_dense  # noqa: E402

OFF = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


def flt(x):
    return float(x) if x not in ("", None) else None


def point_tree(base, r, model):
    M = [[0.0] * 3 for _ in range(3)]
    for a, b in OFF:
        v = flt(r.get(f"m{a}{b}"))
        if model == "sym" and v is None:
            v = flt(r.get(f"m{b}{a}"))
        M[a][b] = v
    Ne = [flt(r[f"Ne{i}"]) for i in range(3)]
    return dict(base, M=M, Ne=Ne)


def rows():
    big = {l.split(",")[0]: float(l.split(",")[2]) for l in open(os.path.join(ROOT, "kaggle", "dense_big.csv"))}
    for r in csv.DictReader(open(os.path.join(HERE, "dense_vs_mascot_openblas.csv"))):
        f = os.path.join(ROOT, "sim", "data", r["file"])
        ex = flt(r["ll_dense"]) if r["ll_dense"] else big.get(r["file"])
        if ex is None:  # 4 trees (d4 k20, peak 15-16) have no dense value anywhere
            continue
        yield dict(set="small", tree=r["file"].replace(".json", ""), d=int(r["d"]), k=int(r["k"]), regime=r["regime"],
                   exact=ex, mascot_stored=flt(r["ll_mascot"]), exact_err=""), json.load(open(f))
    for r in csv.DictReader(open(os.path.join(HERE, "large_sweep.csv"))):
        if not r["tt"]:
            continue
        sub = "large3" if r["d"] == "3" else ("large4x" if r["tree"].endswith(("r2", "r3")) else "large")
        f = os.path.join(ROOT, "sim", "data", sub, r["tree"] + ".json")
        yield dict(set="large", tree=r["tree"], d=int(r["d"]), k=int(r["k"]), regime=r["regime"], exact=float(r["tt"]),
                   mascot_stored=float(r["mascot"]), exact_err=r["err_h2"]), json.load(open(f))
    seen = set()
    for r in csv.DictReader(open(os.path.join(HERE, "realdata.csv"))):
        if r["kind"] != "fit":
            continue
        key = (r["tree"], r["engine"], r["model"], r["loglik"])
        if key in seen:
            continue
        seen.add(key)
        name = "h3n2_mcc" if r["tree"] == "mcc" else f"h3n2_{r['tree'].replace('post', 'post')}"
        base = json.load(open(os.path.join(ROOT, "data", name + ".json")))
        tr = point_tree(base, r, r["model"])
        yield dict(set="h3n2", tree=f"{r['tree']}:{r['engine']}-MLE-{r['model']}", d=3, k=24, regime="real",
                   exact=None, mascot_stored=float(r["loglik"]) if r["engine"] == "mascot" else None,
                   exact_stored=float(r["loglik"]) if r["engine"] == "dense" else None, exact_err=""), tr
    base = json.load(open(os.path.join(ROOT, "data", "aivorder_masco_mcc.json")))
    seen = set()
    for r in csv.DictReader(open(os.path.join(HERE, "realdata2.csv"))):
        if not r["Ne0"] or not r["exact_logL"] or r["point"] in seen:
            continue
        seen.add(r["point"])
        yield dict(set="aiv", tree="aivorder:" + r["point"].replace(" ", "_"), d=3, k=133, regime="real",
                   exact=float(r["exact_logL"]), mascot_stored=float(r["mascot_logL"]), exact_err=""), \
            point_tree(base, r, r["model"])


def main():
    out = []
    for info, tr in rows():
        if info["exact"] is None:
            info["exact"] = esco_dense.loglik(tr)
            if info.get("exact_stored") is not None:
                assert abs(info["exact"] - info["exact_stored"]) < 1e-3, info
        info.pop("exact_stored", None)
        ma = mascot(tr)
        if info["mascot_stored"] is not None:
            assert abs(ma - info["mascot_stored"]) < 1e-3, (info, ma)
        info.pop("mascot_stored")
        info["mascot"] = ma
        info["lisco"] = ind.loglik(tr, "lisco")
        info["sisco"] = ind.loglik(tr, "sisco")
        for m in ("mascot", "lisco", "sisco"):
            info[f"{m}_minus_exact"] = info[m] - info["exact"]
        out.append(info)
        print(info["set"], info["tree"], *(f"{m} {info[m + '_minus_exact']:+.3f}" for m in ("mascot", "lisco", "sisco")),
              flush=True)
    with open(os.path.join(HERE, "sisco_eval.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
