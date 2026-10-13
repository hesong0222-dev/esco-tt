"""MLE 판정 감도 분석 (2026-10-13, 심사 대응 레인). 사전등록 판정(bench/mle_prereg.md)은 바꾸지 않는다.

입력: bench/mle_study.csv (mle_summarize.py 출력), kaggle/out/mle/*/results.jsonl (TT 시간·평가 수).
출력: bench/mle_sensitivity.csv (설정별 모든 CI·검정·판정), bench/mle_sensitivity_missing.csv (그루별 결측·난이도),
      표준 출력(마크다운 표, bench/mle_sensitivity.md에 옮김).
실행: OPENBLAS_NUM_THREADS=1 .venv/bin/python bench/mle_sensitivity.py

(a) 대칭 규칙: TOST(두 단측 검정) 여유폭 0.05·0.10 — 90% CI(부트스트랩 백분위, BCa, 정확 순서통계량)가 (−δ, δ) 안이면
    동등. '차이 있음'도 CI 끝값 기준으로: 95% CI 전체가 ±0.10 밖(|0에 가까운 CI 끝| ≥ 0.10)이면 '여유폭 넘는 차이'.
(b) 다섯 설정 다중 비교: 부호 검정(중앙값 = 0) 정확 p, Wilcoxon 부호순위 p, Holm 보정. Bonferroni 수준 CI = 99% CI
    (1 − 0.05/5)로 사전등록 규칙을 다시 적용.
(c) 작은 n용 CI: BCa 부트스트랩(scipy.stats.bootstrap, 1만 회), 순서통계량 기반 정확 중앙값 CI(이항분포, 분포 무가정).
(d) 2차 지표 결측: 설정별 결측 수(설계상 안 돌린 것과 시간 초과로 빠진 것 구분), 결측 그루와 완료 그루의 난이도 비교
    (TT 참값 1회 평가 시간, TT fit 평가 수, 1차 실행본 v1에서 끝났는지) — Mann–Whitney.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SETTINGS = ["d3_k30_slow", "d3_k50_slow", "d4_k30_slow", "d4_k50_slow", "d3_k30_medium"]
NBOOT = 10000
SEED = 20261013


def load_rows():
    rows = list(csv.DictReader(open(os.path.join(HERE, "mle_study.csv"))))
    for r in rows:
        r["set"] = r["tree"].rsplit("_r", 1)[0]
    return rows


def f(x):
    return None if x in ("", None) else float(x)


def pct_ci(x, level, rng):
    b = np.median(x[rng.integers(0, len(x), (NBOOT, len(x)))], axis=1)
    a = (1 - level) / 2
    return float(np.percentile(b, 100 * a)), float(np.percentile(b, 100 * (1 - a)))


def bca_ci(x, level):
    r = stats.bootstrap((x,), np.median, n_resamples=NBOOT, confidence_level=level, method="BCa",
                        random_state=np.random.default_rng(SEED))
    return float(r.confidence_interval.low), float(r.confidence_interval.high)


def exact_median_ci(x, level):
    """분포 무가정 중앙값 CI: 순서통계량 x_(l), x_(n+1-l), 포함확률 = 1 − 2·P(Bin(n,1/2) ≤ l−1) ≥ level 중 가장 좁은 것."""
    x = np.sort(x)
    n = len(x)
    best = None
    for l in range(1, n // 2 + 1):
        cov = 1 - 2 * stats.binom.cdf(l - 1, n, 0.5)
        if cov >= level:
            best = (l, cov)
    l, cov = best
    return float(x[l - 1]), float(x[n - l]), float(cov)


def decide_prereg(med, lo, hi):
    if abs(med) < 0.05 and lo > -0.10 and hi < 0.10:
        return "실질적 차이 없음"
    if (lo > 0 or hi < 0) and abs(med) >= 0.10:
        return "체계적 차이"
    return "유보"


def decide_sym(lo90_by_delta, lo95, hi95):
    """대칭 규칙. lo90_by_delta: {δ: (lo90, hi90)}. 반환: (TOST 0.05, TOST 0.10, CI끝 기준 차이)."""
    out = []
    for dlt in (0.05, 0.10):
        lo, hi = lo90_by_delta
        out.append("동등" if (lo > -dlt and hi < dlt) else "동등 아님")
    if lo95 >= 0.10 or hi95 <= -0.10:
        diff = "여유폭 넘는 차이"
    elif lo95 > 0 or hi95 < 0:
        diff = "0과 다름(여유폭 이하 가능)"
    else:
        diff = "0과 구별 안 됨"
    return out[0], out[1], diff


def sign_test_p(x):
    pos, neg = int(np.sum(x > 0)), int(np.sum(x < 0))
    return float(stats.binomtest(pos, pos + neg, 0.5).pvalue)


def holm(ps):
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def tost_sign_p(x, dlt):
    """정확 부호 검정 TOST: H0a 중앙값 ≤ −δ, H0b 중앙값 ≥ δ. p = max(두 단측 p)."""
    n = len(x)
    pa = stats.binomtest(int(np.sum(x > -dlt)), n, 0.5, alternative="greater").pvalue
    pb = stats.binomtest(int(np.sum(x < dlt)), n, 0.5, alternative="greater").pvalue
    return float(max(pa, pb))


def primary(rows):
    rng = np.random.default_rng(SEED)
    out = []
    for s in SETTINGS:
        x = np.array([f(r["Delta_m"]) for r in rows if r["set"] == s and r["complete"] == "True"])
        n = len(x)
        med = float(np.median(x))
        p95 = pct_ci(x, 0.95, rng)
        p90 = pct_ci(x, 0.90, rng)
        p99 = pct_ci(x, 0.99, rng)
        b95, b90, b99 = bca_ci(x, 0.95), bca_ci(x, 0.90), bca_ci(x, 0.99)
        e95, e90, e99 = exact_median_ci(x, 0.95), exact_median_ci(x, 0.90), exact_median_ci(x, 0.99)
        rec = dict(setting=s, n=n, median=med,
                   pct95_lo=p95[0], pct95_hi=p95[1], pct90_lo=p90[0], pct90_hi=p90[1], pct99_lo=p99[0], pct99_hi=p99[1],
                   bca95_lo=b95[0], bca95_hi=b95[1], bca90_lo=b90[0], bca90_hi=b90[1], bca99_lo=b99[0], bca99_hi=b99[1],
                   ex95_lo=e95[0], ex95_hi=e95[1], ex95_cov=e95[2], ex90_lo=e90[0], ex90_hi=e90[1], ex90_cov=e90[2],
                   ex99_lo=e99[0], ex99_hi=e99[1], ex99_cov=e99[2],
                   p_sign=sign_test_p(x), p_wilcoxon=float(stats.wilcoxon(x).pvalue),
                   n_neg=int(np.sum(x < 0)),
                   tost05_sign_p=tost_sign_p(x, 0.05), tost10_sign_p=tost_sign_p(x, 0.10))
        rec["prereg"] = decide_prereg(med, *p95)
        for nm, (lo, hi) in (("pct95", p95), ("bca95", b95), ("ex95", e95[:2])):
            rec[f"prereg_{nm}"] = decide_prereg(med, lo, hi)
        for nm, (lo, hi) in (("pct99", p99), ("bca99", b99), ("ex99", e99[:2])):
            rec[f"bonf_{nm}"] = decide_prereg(med, lo, hi)
        for nm, c90, c95 in (("pct", p90, p95), ("bca", b90, b95), ("ex", e90[:2], e95[:2])):
            t05, t10, dif = decide_sym(c90, *c95)
            rec[f"tost05_{nm}"], rec[f"tost10_{nm}"], rec[f"symdiff_{nm}"] = t05, t10, dif
        out.append(rec)
    for key in ("p_sign", "p_wilcoxon"):
        adj = holm(np.array([r[key] for r in out]))
        for r, a in zip(out, adj):
            r[key + "_holm"] = float(a)
    return out


# ----------------------------------------------------------------------------- (d) missingness
def tt_records():
    R = defaultdict(list)
    for fn in glob.glob(os.path.join(ROOT, "kaggle", "out", "mle", "*", "results.jsonl")):
        for l in open(fn):
            if l.strip():
                r = json.loads(l)
                if r["engine"] == "tt":
                    R[r["tree"]].append(r)
    return R


def prof_all_design():
    """그루별로 PROF_ALL=1 커널에 한 번이라도 배정됐는지 (= TT Ne 프로파일이 설계에 있었는지)."""
    D = defaultdict(bool)
    for fn in glob.glob(os.path.join(ROOT, "kaggle", "mle_k", "*", "run.py")):
        s = open(fn).read()
        import re
        m = re.search(r'JOBS = (?:\[j.split\(\':\'\) for j in )?"([^"]*)"', s)
        pa = re.search(r'PROF_ALL = "(\d)"', s)
        if not m or not pa:
            continue
        for tag in m.group(1).split(","):
            D[tag.split(":")[0]] |= pa.group(1) == "1"
    return D


def missing(rows):
    R = tt_records()
    D = prof_all_design()
    per = []
    for r in rows:
        if r["complete"] != "True":
            continue
        t = r["tree"]
        tr = [x for x in R[t] if x["stage"] == "truth"]
        fits = [x for x in R[t] if x["stage"] == "fit"]
        fb = max(fits, key=lambda x: x["loglik"])
        d = int(r["d"])
        per.append(dict(tree=t, set=r["set"], d=d,
                        miss_prof_m=r["tt_cov_m"] == "", miss_eqNe=r["tt_rej_eqNe"] == "",
                        miss_prof_Ne_any=any(r.get(f"tt_cov_Ne{a}", "") == "" for a in range(d)),
                        prof_Ne_designed=D.get(t, False),
                        tt_truth_sec=min(x["sec"] for x in tr) if tr else None,
                        tt_fit_nev=fb.get("nev"), tt_fit_version=fb.get("version", 1),
                        tt_fit_sec=fb.get("sec"), Delta_m=f(r["Delta_m"]), L_loss=f(r["L_loss"])))
    return per


def mw(a, b):
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 1 or len(b) < 1:
        return None
    return float(stats.mannwhitneyu(a, b).pvalue)


def main():
    rows = load_rows()
    P = primary(rows)
    keys = list(P[0].keys())
    with open(os.path.join(HERE, "mle_sensitivity.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        w.writerows(P)
    g = lambda v: f"{v:+.3f}"
    print("## (a)(c) CI 방법별 Δ 구간과 판정\n")
    print("| 설정 | n | 중앙값 | 95% 백분위(사전등록) | 95% BCa | 95% 정확(순서통계량, 실제 포함률) | 사전등록 판정 | BCa로 | 정확 CI로 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in P:
        print(f"| {r['setting']} | {r['n']} | {g(r['median'])} | [{g(r['pct95_lo'])}, {g(r['pct95_hi'])}] | "
              f"[{g(r['bca95_lo'])}, {g(r['bca95_hi'])}] | [{g(r['ex95_lo'])}, {g(r['ex95_hi'])}] ({r['ex95_cov']:.3f}) | "
              f"{r['prereg']} | {r['prereg_bca95']} | {r['prereg_ex95']} |")
    print("\n## (a) 대칭 규칙: TOST(90% CI) 여유폭 0.05 / 0.10, CI 끝값 기준 '차이'(95% CI)\n")
    print("| 설정 | 90% 백분위 | 90% BCa | 90% 정확 | TOST ±0.05 (백분위/BCa/정확) | TOST ±0.10 (백분위/BCa/정확) | 부호검정 TOST p (0.05 / 0.10) | CI 끝 기준 (백분위/BCa/정확) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in P:
        print(f"| {r['setting']} | [{g(r['pct90_lo'])}, {g(r['pct90_hi'])}] | [{g(r['bca90_lo'])}, {g(r['bca90_hi'])}] | "
              f"[{g(r['ex90_lo'])}, {g(r['ex90_hi'])}] | {r['tost05_pct']}/{r['tost05_bca']}/{r['tost05_ex']} | "
              f"{r['tost10_pct']}/{r['tost10_bca']}/{r['tost10_ex']} | {r['tost05_sign_p']:.3g} / {r['tost10_sign_p']:.3g} | "
              f"{r['symdiff_pct']} / {r['symdiff_bca']} / {r['symdiff_ex']} |")
    print("\n## (b) 다중 비교: 0 검정 p(Holm 보정)와 Bonferroni 수준(99%) CI로 다시 적용한 사전등록 규칙\n")
    print("| 설정 | Δ<0 그루 | 부호검정 p | Holm | Wilcoxon p | Holm | 99% 백분위 | 99% BCa | 99% 정확 (포함률) | 판정(백분위/BCa/정확) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in P:
        print(f"| {r['setting']} | {r['n_neg']}/{r['n']} | {r['p_sign']:.2g} | {r['p_sign_holm']:.2g} | {r['p_wilcoxon']:.2g} | "
              f"{r['p_wilcoxon_holm']:.2g} | [{g(r['pct99_lo'])}, {g(r['pct99_hi'])}] | [{g(r['bca99_lo'])}, {g(r['bca99_hi'])}] | "
              f"[{g(r['ex99_lo'])}, {g(r['ex99_hi'])}] ({r['ex99_cov']:.3f}) | {r['bonf_pct99']}/{r['bonf_bca99']}/{r['bonf_ex99']} |")

    M = missing(rows)
    with open(os.path.join(HERE, "mle_sensitivity_missing.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(M[0].keys()))
        w.writeheader()
        w.writerows(M)
    print("\n## (d) TT 2차 지표 결측\n")
    print("| 설정 | n | m 프로파일 결측 | eqNe 결측 | Ne 프로파일: 설계에 있음 / 그중 결측 | 결측(m 또는 eqNe) 그루의 TT 참값 1회 시간 중앙값 vs 완료 그루 (s) | MW p | fit 평가 수 중앙값 결측 vs 완료 | MW p | 결측 그루 중 v1(1차) 완료 fit | 결측 vs 완료 Δ 중앙값 | MW p |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SETTINGS:
        A = [x for x in M if x["set"] == s]
        mis = [x for x in A if x["miss_prof_m"] or x["miss_eqNe"]]
        ok = [x for x in A if not (x["miss_prof_m"] or x["miss_eqNe"])]
        des = [x for x in A if x["prof_Ne_designed"]]
        md = lambda L, k: f"{np.median([x[k] for x in L if x[k] is not None]):.0f}" if L else "–"
        mdd = lambda L, k: f"{np.median([x[k] for x in L]):+.3f}" if L else "–"
        p1, p2, p3 = mw([x["tt_truth_sec"] for x in mis], [x["tt_truth_sec"] for x in ok]), \
            mw([x["tt_fit_nev"] for x in mis], [x["tt_fit_nev"] for x in ok]), \
            mw([x["Delta_m"] for x in mis], [x["Delta_m"] for x in ok])
        ps = lambda p: "–" if p is None else f"{p:.2g}"
        print(f"| {s} | {len(A)} | {sum(x['miss_prof_m'] for x in A)} | {sum(x['miss_eqNe'] for x in A)} | "
              f"{len(des)} / {sum(x['miss_prof_Ne_any'] for x in des)} | {md(mis, 'tt_truth_sec')} vs {md(ok, 'tt_truth_sec')} | {ps(p1)} | "
              f"{md(mis, 'tt_fit_nev')} vs {md(ok, 'tt_fit_nev')} | {ps(p2)} | {sum(x['tt_fit_version'] == 1 for x in mis)}/{len(mis)} | "
              f"{mdd(mis, 'Delta_m')} vs {mdd(ok, 'Delta_m')} | {ps(p3)} |")
    allm = [x for x in M if x["miss_prof_m"] or x["miss_eqNe"]]
    allo = [x for x in M if not (x["miss_prof_m"] or x["miss_eqNe"])]
    # 설정 안에서 순위로 바꾼 뒤 합쳐서 비교 (설정 간 크기 차이 제거)
    rk = {}
    for s in SETTINGS:
        A = [x for x in M if x["set"] == s]
        v = stats.rankdata([x["tt_truth_sec"] for x in A]) / len(A)
        for x, q in zip(A, v):
            rk[x["tree"]] = q
    print(f"\n- 전체: 결측 {len(allm)}그루 / 완료 {len(allo)}그루. 설정 안 순위(0–1)로 본 TT 참값 1회 시간: 결측 중앙 "
          f"{np.median([rk[x['tree']] for x in allm]):.2f} vs 완료 {np.median([rk[x['tree']] for x in allo]):.2f}, "
          f"MW p {mw([rk[x['tree']] for x in allm], [rk[x['tree']] for x in allo]):.2g}")


if __name__ == "__main__":
    main()
