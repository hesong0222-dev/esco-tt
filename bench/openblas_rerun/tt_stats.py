"""tt_vs_dense CSV 요약: rank_study.md 1절 표와 같은 열 + 오차추정 커버리지. 인자: csv [파일목록 제한용 csv]"""
import csv, sys, math, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
if len(sys.argv) > 2:
    keep = {r["file"] for r in csv.DictReader(open(sys.argv[2]))}
    rows = [r for r in rows if r["file"] in keep]
print(len(rows), "trees")
for d in "234":
    R = [r for r in rows if r["d"] == d]
    print(f"| {d} | {len(R)} | {max(int(r['kmax']) for r in R)} | {max(abs(float(r['diff'])) for r in R):.1e} | {max(abs(float(r['diff_h'])) for r in R):.1e} | {max(int(r['max_rank']) for r in R)} |")
a = [abs(float(r["diff"])) for r in rows]
print("median |diff|", f"{st.median(a):.2g}", "max", f"{max(a):.2g}", max(rows, key=lambda r: abs(float(r['diff'])))["file"])
for m in (1, 3):
    cov = sum(1 for r in rows if (float(r["err_step"]) + float(r["err_trunc"])) * m >= abs(float(r["diff"])))
    print(f"coverage x{m}: {cov}/{len(rows)} = {100*cov/len(rows):.1f}%")
worse = sum(1 for r in rows if abs(float(r["diff"])) > abs(float(r["diff_h"])))
print(f"Richardson worse than plain h: {worse}/{len(rows)} = {100*worse/len(rows):.1f}%")
inf = [r["file"] for r in rows if not math.isfinite(float(r["err_trunc"]))]
print("non-finite err_trunc:", inf)
