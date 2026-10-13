"""Merge rank-study CSVs into bench/rank_study.csv and print markdown tables.  Usage: make_rank_tables.py in1.csv [in2.csv ...]"""
import sys, csv, os
rows = {}
for p in sys.argv[1:]:
    for r in csv.DictReader(open(p)):
        if r.get("err"):
            continue
        key = (int(r["d"]), int(r["k"]), float(r["m"]), int(r["seed"]), float(r["eps"]))
        rows[key] = r
out = os.path.join(os.path.dirname(__file__), "rank_study.csv")
fields = ["d", "k", "m", "seed", "eps", "rmax", "loglik", "max_rank", "runtime", "n_steps", "n_mig", "tmrca"]
with open(out, "w") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for key in sorted(rows):
        w.writerow(rows[key])
print(f"{len(rows)} rows -> {out}\n")
for eps in (1e-8, 1e-6, 1e-4):
    sel = {k: v for k, v in rows.items() if k[4] == eps}
    if not sel:
        continue
    ks = sorted({k[1] for k in sel})
    print(f"### eps = {eps:g}  (cell: max TT rank / runtime s)\n")
    print("| d | m | " + " | ".join(f"k={k}" for k in ks) + " |")
    print("|---|---|" + "---|" * len(ks))
    for d in (2, 3, 4, 6):
        for m in (0.1, 1.0, 10.0):
            cells = []
            for k in ks:
                r = sel.get((d, k, m, 1, eps))
                cells.append(f"{r['max_rank']} / {float(r['runtime']):.0f}" + ("*" if r.get("rmax") else "") if r else "")
            if any(cells):
                print(f"| {d} | {m:g} | " + " | ".join(cells) + " |")
    print()
