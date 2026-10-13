"""OpenBLAS 재계산: run_dense_vs_mascot.run()을 그대로 쓰고 출력만 새 파일로. 큰 나무도 병렬(3)."""
import csv, glob, json, os, sys
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import run_dense_vs_mascot as rdm  # noqa: E402
OUT = os.path.join(HERE, "..", "dense_vs_mascot_openblas.csv")
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(HERE, "..", "..", "sim", "data", "*.json")))
    size = {f: json.load(open(f))["demes"] ** rdm.ed.max_lineages(json.load(open(f))) for f in files}
    small = [f for f in files if size[f] < 4 ** 11]
    big = sorted([f for f in files if size[f] >= 4 ** 11], key=lambda f: -size[f])
    with Pool(3) as pool:
        rows = pool.map(rdm.run, small) + pool.map(rdm.run, big, chunksize=1)
    rows.sort(key=lambda r: r["file"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rdm.FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)
