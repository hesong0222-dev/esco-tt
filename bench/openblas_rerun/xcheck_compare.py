"""kaggle/xcheck.csv(Mac Accelerate) vs xcheck_openblas.csv, dense(Kaggle) 대비 오차표 재생성."""
import csv, os, re
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "..")
def load(p):
    d = {}
    for l in open(p):
        f = l.strip().split(",")
        if len(f) < 4: continue
        m = re.search(r"-?[0-9.]+(e-?\d+)?", f[2].replace("np.float64(", ""))
        d[(f[0], float(f[1]))] = float(f[2].replace("np.float64(", "").rstrip(")")) if "TIMEOUT" not in f[2] and "ERROR" not in f[2] else None
    return d
old, new = load(f"{R}/kaggle/xcheck.csv"), load(f"{H}/xcheck_mascot_openblas.csv")
new.update({k: v for k, v in load(f"{H}/xcheck_openblas.csv").items() if v is not None or k not in new})
dense = {k[0]: v for k, v in load(f"{R}/kaggle/dense_big.csv").items()}
print("old-new 비교 (dense 행 제외):")
mx = 0
for k in sorted(set(old) & set(new)):
    if k[1] == 13.0: continue
    a, b = old[k], new[k]
    dd = abs(a - b) if (a is not None and b is not None) else float("inf")
    mx = max(mx, dd if dd != float("inf") else 0)
    flag = "  <-- >1e-9" if dd > 1e-9 else ""
    print(f"  {k[0]:24s} {k[1]:g}: old {a!r} new {b!r} |Δ| {dd:.3g}{flag}")
print("missing in new:", sorted(k for k in old if k not in new and k[1] != 13.0))
print("max finite |Δ|", mx)
print("\n| 계통수 | dense | TT 1e-6 오차 | TT 1e-7 오차 | MASCOT−dense |")
for t in sorted(dense):
    g = lambda e: new.get((t, e))
    c6 = f"{abs(g(1e-6)-dense[t]):.1e}" if g(1e-6) is not None else "—"
    c7 = f"{abs(g(1e-7)-dense[t]):.1e}" if g(1e-7) is not None else "—"
    print(f"| {t[:-5]} | {dense[t]:.6f} | {c6} | {c7} | {g(0.0)-dense[t]:+.3f} |")
for t in ("d4_k50_medium_r0.json", "d4_k60_medium_r0.json"):
    print(t, "MASCOT new", new.get((t, 0.0)), "old", old.get((t, 0.0)))
