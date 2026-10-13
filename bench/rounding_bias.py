"""반올림(TT 라운딩) 편향의 모수 의존성 (2026-10-13, 심사 대응 레인) — (a) N_round 계산 부분.

오차 모형(bench/robustness.md): logL_TT − logL_exact ≈ c · eps · N_round(θ), c는 나무마다 거의 고정, |c| ≤ 0.81.
N_round는 시간 간격 규칙 h ≤ 0.25 / max(q_max, c_max·k(k−1))(esco_tt.count_roundings) 때문에 θ에 따라 달라진다.
→ 가능도 표면에 기울기 c·eps·∇N이 더해진다. 여기서 크기를 잰다.

(a1) MLE 143그루 전부: 참값·MASCOT 최적점·정확 최적점·eqNe 정확 최적점에서 N_round, 정확 최적점에서 ∂N/∂θ_j(로그 공간
     중앙차분 ±0.1), MASCOT 헤시안(정확 최적점, 중앙차분 1e-3)으로 최적점 이동 δθ = −H⁻¹·c·eps·∇N.
     L_loss 편향 = c·eps·(N(θ̂_exact) − N(θ̂_MASCOT)), LRT(eqNe) 편향 = 2·c·eps·(N(θ̂) − N(θ̂_eq)).
(a2) AIV 봉우리(bench/aiv_eps7/points + masco gal 뿌리): 봉우리 사이 N_round 차이 → 봉우리 순서·LRT 편향.
출력: bench/rounding_bias_nround.csv, bench/rounding_bias_aiv.csv, 표준 출력.
실행: OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python bench/rounding_bias.py
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import mle_common as C
from tt import esco_tt

EPS = 1e-6
CMAX = 0.81


def nr(tree, theta):
    return esco_tt.count_roundings(C.with_theta(tree, theta))


def mascot_hess(tree, th, h=1e-3):
    p = len(th)
    f = lambda x: C.evaluate("mascot", tree, x)[0]
    H = np.zeros((p, p))
    f0 = f(th)
    for i in range(p):
        for j in range(i, p):
            ei, ej = np.eye(p)[i] * h, np.eye(p)[j] * h
            if i == j:
                H[i, i] = (f(th + ei) - 2 * f0 + f(th - ei)) / h ** 2
            else:
                H[i, j] = H[j, i] = (f(th + ei + ej) - f(th + ei - ej) - f(th - ei + ej) + f(th - ei - ej)) / (4 * h * h)
    return H


def best(rs, eng, stage):
    c = [r for r in rs if r["engine"] == eng and r["stage"] == stage and "theta" in r]
    return max(c, key=lambda r: r["loglik"]) if c else None


def part_mle():
    rows = {r["tree"]: r for r in csv.DictReader(open(os.path.join(HERE, "mle_study.csv")))}
    R = defaultdict(list)
    for fn in glob.glob(os.path.join(ROOT, "kaggle", "out", "mle", "*", "results.jsonl")):
        for l in open(fn):
            if l.strip():
                r = json.loads(l)
                R[r["tree"]].append(r)
    out = []
    for tag in sorted(rows):
        s = rows[tag]
        if s["complete"] != "True":
            continue
        tree = json.load(open(os.path.join(HERE, "mle", "trees", tag + ".json")))
        d = tree["demes"]
        th0 = np.array(C.true_theta(d, tree["meta"]["regime"]))
        thm = np.array([float(s[f"mascot_{n}"]) for n in C.names(d)])
        tht = np.array([float(s[f"tt_{n}"]) for n in C.names(d)])
        eq = best(R[tag], "tt", "eqNe")
        n0, nm, nt = nr(tree, th0), nr(tree, thm), nr(tree, tht)
        neq = nr(tree, np.array(eq["theta"])) if eq else None
        g = np.array([(nr(tree, tht + 0.1 * e) - nr(tree, tht - 0.1 * e)) / 0.2 for e in np.eye(d + 1)])
        H = mascot_hess(tree, tht)
        try:
            dth = -np.linalg.solve(H, CMAX * EPS * g)   # c = 0.81 (최악), 부호는 c 부호에 따라 뒤집힘
        except np.linalg.LinAlgError:
            dth = np.full(d + 1, np.nan)
        rec = dict(tree=tag, set=tag.rsplit("_r", 1)[0], N_true=n0, N_mascot=nm, N_exact=nt, N_eqNe=neq,
                   dN_exact_minus_mascot=nt - nm, dNdlogm=g[0], dNdlogNe_max=float(np.max(np.abs(g[1:]))),
                   Lloss_bias_cmax=CMAX * EPS * abs(nt - nm), LRT_bias_cmax=(2 * CMAX * EPS * abs(nt - neq)) if neq else None,
                   shift_logm_cmax=float(abs(dth[0])), shift_max_cmax=float(np.max(np.abs(dth))),
                   Delta_m=float(s["Delta_m"]), L_loss=float(s["L_loss"]),
                   H_logm=float(H[0, 0]))
        out.append(rec)
        print(f"{tag} N {n0}/{nm}/{nt} dN {nt - nm} dN/dlogm {g[0]:.0f} shift_m {abs(dth[0]):.2e} "
              f"Lloss_bias {rec['Lloss_bias_cmax']:.2e} Delta {rec['Delta_m']:+.3f}", flush=True)
    with open(os.path.join(HERE, "rounding_bias_nround.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


def part_aiv():
    import realdata_fit as rf
    P = {k: v for k, v in json.load(open(os.path.join(HERE, "aiv_eps7", "points_index.json"))).items()}
    pts = {k: (v["tree"], os.path.join(HERE, "aiv_eps7", "points", k + ".json")) for k, v in P.items()}
    pts["masco_asym_gal"] = ("masco", os.path.join(HERE, "realdata2", "aivorder_refine_asym_best_fit.json"))
    out = []
    for k, (t, path) in sorted(pts.items()):
        rf._init(os.path.join(ROOT, "data", f"aivorder_{t}_mcc.json"))
        f = json.load(open(path))
        M, Ne = rf.unpack(f["model"], f["theta"])
        tr = dict(rf.TREE, M=M.tolist(), Ne=Ne.tolist())
        n = esco_tt.count_roundings(tr)
        out.append(dict(point=k, tree=t, model=f["model"], N_round=n, tt_1e6=f["loglik"], err_cmax=CMAX * EPS * n))
        print(k, n, flush=True)
    with open(os.path.join(HERE, "rounding_bias_aiv.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("aiv", "all"):
        part_aiv()
    if which in ("mle", "all"):
        part_mle()
