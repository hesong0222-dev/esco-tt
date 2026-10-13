"""Run dense (exact) and MASCOT on the sim/data test set; write bench/dense_vs_mascot.csv."""
import csv
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dense"))
sys.path.insert(0, os.path.join(HERE, "..", "approx"))
import esco_dense as ed  # noqa: E402
import mascot  # noqa: E402

CAP = int(os.environ.get("DENSE_CAP", 4 ** 12))
OUT = os.path.join(HERE, "dense_vs_mascot.csv")
FIELDS = ["file", "d", "k", "regime", "rep", "max_lineages", "max_states",
          "ll_dense", "t_dense_s", "ll_mascot", "t_mascot_s", "mascot_minus_dense"]


def run(path):
    tree = json.load(open(path))
    m = tree["meta"]
    n = ed.max_lineages(tree)
    S = tree["demes"] ** n
    t0 = time.time()
    lm = mascot.loglik(tree)
    tm = time.time() - t0
    row = {"file": os.path.relpath(path, os.path.join(HERE, "..", "sim", "data")), "d": m["d"], "k": m["k"],
           "regime": m["regime"], "rep": m["rep"], "max_lineages": n, "max_states": S,
           "ll_mascot": f"{lm:.10f}", "t_mascot_s": f"{tm:.3f}"}
    if S <= CAP:
        t0 = time.time()
        ld = ed.loglik(tree)
        row.update(ll_dense=f"{ld:.10f}", t_dense_s=f"{time.time() - t0:.3f}",
                   mascot_minus_dense=f"{lm - ld:.8f}")
    else:
        row.update(ll_dense="", t_dense_s="", mascot_minus_dense="")
    print(row["file"], n, S, row["ll_dense"], row["t_dense_s"], flush=True)
    return row


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(HERE, "..", "sim", "data", "*.json")))
    size = {f: (lambda t: t["demes"] ** ed.max_lineages(t))(json.load(open(f))) for f in files}
    files.sort(key=lambda f: size[f])
    small = [f for f in files if size[f] < 4 ** 11]
    big = [f for f in files if size[f] >= 4 ** 11]
    with Pool(6) as pool:
        rows = pool.map(run, small)
    rows += [run(f) for f in big]
    rows.sort(key=lambda r: r["file"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)
