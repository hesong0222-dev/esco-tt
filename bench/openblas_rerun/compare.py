"""옛(Accelerate) vs 새(OpenBLAS) CSV 비교: 열마다 max|old-new|, 1e-9 넘는 행 목록."""
import csv, glob, os, sys
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(H, "..")
TH = 1e-9
def load(p, key):
    return {key(r): r for r in csv.DictReader(open(p)) if not r.get("err")}
def cmp(name, old, new, cols, intcols=()):
    print(f"## {name}: old {len(old)}행, new {len(new)}행, 공통 {len(set(old)&set(new))}")
    miss = sorted(set(old) - set(new))
    if miss: print("  new에 없음:", miss)
    for c in cols:
        ds = []
        for k in sorted(set(old) & set(new)):
            a, b = old[k][c], new[k][c]
            if a == "" or b == "":
                if a != b: ds.append((k, a, b, float("inf")))
                continue
            ds.append((k, a, b, abs(float(a) - float(b))))
        mx = max((d[3] for d in ds), default=0)
        big = [d for d in ds if d[3] > TH]
        print(f"  {c}: max|old-new| = {mx:.3g}, >1e-9: {len(big)}행")
        for k, a, b, d in sorted(big, key=lambda x: -x[3]):
            print(f"    {k}: old {a}  new {b}  |Δ| {d:.3g}")
    for c in intcols:
        diff = [(k, old[k][c], new[k][c]) for k in sorted(set(old) & set(new)) if old[k][c] != new[k][c]]
        print(f"  {c}: 다른 행 {len(diff)}", diff)
which = sys.argv[1:] or ["dense", "tt", "rank"]
if "dense" in which and os.path.exists(f"{B}/dense_vs_mascot_openblas.csv"):
    f = lambda r: r["file"]
    cmp("dense_vs_mascot", load(f"{B}/dense_vs_mascot.csv", f), load(f"{B}/dense_vs_mascot_openblas.csv", f),
        ["ll_dense", "ll_mascot", "mascot_minus_dense"])
if "tt" in which and os.path.exists(f"{B}/tt_vs_dense_openblas.csv"):
    f = lambda r: r["file"]
    cmp("tt_vs_dense", load(f"{B}/tt_vs_dense.csv", f), load(f"{B}/tt_vs_dense_openblas.csv", f),
        ["ll_dense", "ll_tt", "ll_tt_h", "diff", "err_step", "err_trunc"], ["max_rank"])
if "rank" in which:
    f = lambda r: (int(r["d"]), int(r["k"]), float(r["m"]), int(r["seed"]), float(r["eps"]))
    new = {}
    for p in glob.glob(f"{H}/rank/*.csv"):
        new.update(load(p, f))
    cmp("rank_study", load(f"{B}/rank_study.csv", f), new, ["loglik"], ["max_rank", "n_steps", "n_mig"])
