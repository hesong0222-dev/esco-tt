"""Robustness runs (REVIEW.md required fixes 2-3): eps ladder, step ladder, mass_fix on/off.
Usage: python bench/robustness.py out.csv TREE.json,eps|Teps_total,steps_mult,mass_fix[,s_max=0.25[,order=4]] ...   (NPROC parallel, appends rows)
TREE paths are relative to the esco root (or absolute)."""
import sys, os, csv
from multiprocessing import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from tt import esco_tt

FIELDS = ["tree", "eps", "eps_total", "steps_mult", "s_max", "order", "mass_fix", "loglik", "max_rank", "n_round", "n_steps", "runtime", "err"]


def run(job):
    path, eps, sm, mf, smax, order = job
    et = float(eps[1:]) if eps.startswith("T") else None   # "T1e-3" = eps_total mode
    eps = 0.0 if et else float(eps)
    row = dict(tree=os.path.basename(path), eps=eps, eps_total=et or "", steps_mult=sm, s_max=smax, order=order, mass_fix=mf)
    try:
        p = path if os.path.isabs(path) else os.path.join(ROOT, path)
        r = esco_tt.loglik_detail(open(p).read(), eps=eps, steps_mult=sm, s_max=smax, order=order, mass_fix=bool(mf), eps_total=et,
                                  verbose=os.environ.get("VERBOSE") == "1")
        row.update(eps=r["eps"], loglik=repr(r["loglik"]), max_rank=r["max_rank"], n_round=r["n_round"], n_steps=r["n_steps"],
                   runtime=round(r["runtime"], 1))
    except Exception as e:  # noqa
        row.update(err=str(e)[:100])
    return row


if __name__ == "__main__":
    out = sys.argv[1]
    jobs = []
    for a in sys.argv[2:]:
        f = a.split(",")
        jobs.append((f[0], f[1], int(f[2]), int(f[3]), float(f[4]) if len(f) > 4 else 0.25, int(f[5]) if len(f) > 5 else 4))
    new = not os.path.exists(out)
    with open(out, "a") as fh, Pool(int(os.environ.get("NPROC", "2"))) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for row in pool.imap_unordered(run, jobs):
            w.writerow(row)
            fh.flush()
            print(row, flush=True)
