"""crosscheck_tt_dense.run을 그대로 쓰되, 옛 CSV에 있던 181그루를 먼저, 나머지(옛날 미완 24그루)는 뒤에. 같은 CSV에 이어 씀."""
import csv, glob, json, os, sys
from multiprocessing import Pool
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(H, "..")
sys.path.insert(0, B)
import crosscheck_tt_dense as cc  # noqa: E402
OUT = os.path.join(B, "tt_vs_dense_openblas.csv")
FIELDS = ["file", "d", "kmax", "ll_dense", "ll_tt", "ll_tt_h", "diff", "diff_h", "err_step", "err_trunc", "max_rank", "t_dense", "t_tt"]
if __name__ == "__main__":
    old = {r["file"] for r in csv.DictReader(open(os.path.join(B, "tt_vs_dense.csv")))}
    done = {r["file"] for r in csv.DictReader(open(OUT))}
    files = [p for p in sorted(glob.glob(os.path.join(B, "..", "sim", "data", "*.json")))
             if (lambda t: t["demes"] ** cc.esco_dense.max_lineages(t))(json.load(open(p))) <= 2e6 and os.path.basename(p) not in done]
    files.sort(key=lambda p: os.path.basename(p) not in old)  # 옛 181그루 먼저
    print(len(files), "trees", flush=True)
    with open(OUT, "a") as f, Pool(int(os.environ.get("NPROC", "3"))) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        for row in pool.imap_unordered(cc.run, [(p, 1e-8) for p in files]):
            w.writerow(row); f.flush()
            print(row["file"], row["kmax"], f"{row['diff']:+.2e}", row["max_rank"], row["t_tt"], flush=True)
    print("TT_DONE", flush=True)
