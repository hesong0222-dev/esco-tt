"""B1(TT 질량 언더플로, review/AUDIT_CODE.md) 영향 전수 점검: MLE 연구의 v1 TT 기록 (2026-10-13, 심사 대응 레인).

기준(review/audit_code/fix_scan_b1.py와 같음): 모수점마다
  H = max over 합류~합류 구간 ∫ C(k,2) · max_a(1/Ne_a) dt   (모든 계통이 Ne가 가장 작은 지역에 있다고 놓은 가장 빠른 감쇠)
TT 질량은 합류 직전에 ≥ e^{−H}이므로 H < 300(질량 > 1e-130, 노름 제곱 > 1e-260)이면 언더플로가 불가능하다.
같은 나무에서 H는 1/min(Ne)에 비례하므로 H(θ) = H_true · (min Ne_true / min Ne_θ) (참 min Ne = 1).
기울기 중앙차분은 로그 공간 ±0.05라서 각 반복점 주변 점은 H가 최대 e^{0.05} = 1.051배 → H·1.051 < 300도 같이 본다.

대상: kaggle/out/mle/*/results.jsonl에서 TT fit이 v1(버전 표시 없음)인 그루. 그 그루의 TT가 평가한 모든 기록점:
  truth(참값), fit(최적점), cross(= MASCOT 기록 'cross', TT를 MASCOT 최적점에서 평가), prof_j, eqNe.
또 그루 로그(<tree>.log)의 '[tt] f=' 줄(최적화 중 중심점 logL)로 기록 안 된 경로 점의 logL 범위를 본다.
출력: bench/mle_b1_check.csv, 표준 출력.
실행: OPENBLAS_NUM_THREADS=1 .venv/bin/python bench/mle_b1_check.py
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def H_of(tree, Ne):
    lam = max(1.0 / x for x in Ne)
    ev = sorted(tree["events"], key=lambda e: (e["t"], 0 if e["type"] == "sample" else 1))
    k, t, acc, best = 0, ev[0]["t"], 0.0, 0.0
    for e in ev:
        acc += k * (k - 1) / 2 * lam * (e["t"] - t)
        t = e["t"]
        if e["type"] == "sample":
            k += 1
        else:
            best = max(best, acc)
            acc = 0.0
            k -= 1
    return best


def main():
    recs = defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(ROOT, "kaggle", "out", "mle", "*", "results.jsonl"))):
        src = os.path.basename(os.path.dirname(fn))
        for l in open(fn):
            if l.strip():
                r = json.loads(l)
                r["_src"] = src
                recs[r["tree"]].append(r)
    rows, trees = [], []
    for tag, rs in sorted(recs.items()):
        v1 = [r for r in rs if r["engine"] == "tt" and r["stage"] == "fit" and r.get("version", 1) == 1]
        if not v1:
            continue
        trees.append(tag)
        tree = json.load(open(os.path.join(HERE, "mle", "trees", tag + ".json")))
        Ne_true = tree["Ne"]
        Ht = H_of(tree, Ne_true)
        srcs = {r["_src"] for r in v1}
        pts = []
        for r in rs:
            if r["_src"] not in srcs or r.get("version", 1) != 1:
                continue
            if r["engine"] == "tt" and r["stage"] == "truth":
                pts.append(("tt truth", Ne_true, r["loglik"]))
            elif r["engine"] == "tt" and "theta" in r:
                pts.append((f"tt {r['stage']}", list(np.exp(r["theta"][1:])), r["loglik"]))
            elif r["engine"] == "mascot" and r["stage"] == "fit":
                # TT fit의 시작점이자 TT cross(=mascot 'cross') 평가점
                pts.append(("tt at MASCOT MLE (start, cross)", list(np.exp(r["theta"][1:])), None))
        for lab, Ne, ll in pts:
            h = H_of(tree, Ne)
            rows.append(dict(tree=tag, point=lab, minNe=min(Ne), minNe_over_true=min(Ne) / min(Ne_true), H=h,
                             H_stencil=h * np.exp(0.05), H_true=Ht, tt_loglik=ll, safe=h * np.exp(0.05) < 300))
        # 로그의 경로 점 (중심점 logL)
        fs = []
        for src in srcs:
            lg = os.path.join(ROOT, "kaggle", "out", "mle", src, tag + ".log")
            if os.path.exists(lg):
                fs += [float(x) for x in re.findall(r"\[tt[^\]]*\] f=([-\d.e+]+|nan|inf)", open(lg).read())]
        rows.append(dict(tree=tag, point=f"log: {len(fs)} TT 경로 중심점, logL 최소", minNe=None, minNe_over_true=None,
                         H=None, H_stencil=None, H_true=Ht, tt_loglik=(-max(fs) if fs else None), safe=None))
    with open(os.path.join(HERE, "mle_b1_check.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    P = [r for r in rows if r["H"] is not None]
    L = [r for r in rows if r["H"] is None]
    print(f"v1 TT 그루 {len(trees)}개, 기록점 {len(P)}개")
    print(f"H_true 최대 {max(r['H_true'] for r in rows):.1f}")
    print(f"H(기록점) 최대 {max(r['H'] for r in P):.1f}, 스텐실 포함 최대 {max(r['H_stencil'] for r in P):.1f}; "
          f"H·1.051 ≥ 300 인 점 {sum(not r['safe'] for r in P)}개")
    print(f"min Ne / 참 min Ne 최소 {min(r['minNe_over_true'] for r in P):.3f}; 0.25 미만 점 "
          f"{sum(r['minNe_over_true'] < 0.25 for r in P)}개")
    by = defaultdict(list)
    for r in P:
        by[r["point"].split()[1] if r["point"].startswith("tt ") else r["point"]].append(r)
    for k, v in sorted(by.items()):
        print(f"  {k:12s} n={len(v):4d}  H 최대 {max(x['H'] for x in v):7.1f}  minNe/참 최소 {min(x['minNe_over_true'] for x in v):.3f}")
    worst = sorted(P, key=lambda r: -r["H"])[:8]
    print("H 상위 8점:")
    for r in worst:
        print(f"  {r['tree']} {r['point']} H={r['H']:.1f} minNe={r['minNe']:.3f} ll={r['tt_loglik']}")
    lls = [r["tt_loglik"] for r in L if r["tt_loglik"] is not None]
    print(f"로그 경로 중심점: 그루 {len(lls)}개, 그루별 최소 logL 중 가장 낮은 값 {min(lls):.2f}, "
          f"−700 미만 그루 {sum(x < -700 for x in lls)}개, 기록 없는 그루 {len(L) - len(lls)}개")
    ss = [r for r in rows if r["point"].startswith("tt ") and r["tt_loglik"] is not None]
    print(f"기록된 TT logL 최소 {min(r['tt_loglik'] for r in ss):.2f}")


if __name__ == "__main__":
    main()
