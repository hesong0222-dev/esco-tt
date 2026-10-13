"""Cross-check tt/ against dense/ on sim/data.  Usage: python bench/crosscheck_tt_dense.py out.csv [max_states] [eps]"""
import sys, os, glob, json, csv, time
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tt import esco_tt
import importlib.util
_spec = importlib.util.spec_from_file_location('esco_dense', os.path.join(os.path.dirname(__file__), '..', 'dense', 'esco_dense.py'))
esco_dense = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(esco_dense)


def run(args):
    path, eps = args
    tree = json.load(open(path))
    kmax = esco_dense.max_lineages(tree)
    t0 = time.time()
    ld = esco_dense.loglik(tree)
    td = time.time() - t0
    r = esco_tt.loglik_with_error(tree, eps=eps)
    return dict(file=os.path.basename(path), d=tree["demes"], kmax=kmax, ll_dense=ld, ll_tt=r["loglik"], ll_tt_h=r["ll_h"],
                diff=r["loglik"] - ld, diff_h=r["ll_h"] - ld, err_step=r["err_step"], err_trunc=r["err_trunc"],
                max_rank=r["max_rank"], t_dense=round(td, 2), t_tt=round(r["runtime"], 2))


if __name__ == "__main__":
    out = sys.argv[1]
    max_states = float(sys.argv[2]) if len(sys.argv) > 2 else 2e6
    eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
    root = os.path.join(os.path.dirname(__file__), "..", "sim", "data")
    files = []
    for p in sorted(glob.glob(os.path.join(root, "*.json"))):
        t = json.load(open(p))
        if t["demes"] ** esco_dense.max_lineages(t) <= max_states:
            files.append((p, eps))
    fields = ["file", "d", "kmax", "ll_dense", "ll_tt", "ll_tt_h", "diff", "diff_h", "err_step", "err_trunc", "max_rank", "t_dense", "t_tt"]
    done = set()
    if os.path.exists(out):
        done = {r["file"] for r in csv.DictReader(open(out))}
        files = [x for x in files if os.path.basename(x[0]) not in done]
    print(len(files), "trees", flush=True)
    with open(out, "a") as f, Pool(int(os.environ.get("NPROC", "6"))) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if not done:
            w.writeheader()
        for row in pool.imap_unordered(run, files):
            w.writerow(row)
            f.flush()
            print(f"{row['file']:28s} k={row['kmax']:2d} dense={row['ll_dense']:.8f} diff={row['diff']:+.2e} (h:{row['diff_h']:+.2e}) est_step={row['err_step']:.1e} est_trunc={row['err_trunc']:.1e} rank={row['max_rank']} t={row['t_dense']}/{row['t_tt']}s", flush=True)
