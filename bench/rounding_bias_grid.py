"""반올림 편향 (b): 작은 dense 나무에서 TT(eps 1e-6) 우도 표면과 정확(dense) 표면을 m 축 1차원 격자로 비교 (2026-10-13).

나무: sim/data의 d3·d4 slow 2그루씩(오차 모형 Set A에서 |c|가 큰 쪽 포함). 이동 행렬 전체를 배율 exp(s)로 바꾸고 Ne는 참값.
s 격자: dense 최대점 s* 주변 ±0.4, 17점. 두 표면에 4차 다항식을 맞춰 최대점 위치를 구하고 차이(TT − dense)를 잰다.
같은 격자에서 N_round(s)와 TT − dense 오차를 기록해 c·eps·N_round 모형의 기울기도 본다.
출력: bench/rounding_bias_grid.csv, 표준 출력.
실행: OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python bench/rounding_bias_grid.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from dense import esco_dense
from tt import esco_tt

TREES = ["d3_k12_slow_r1", "d3_k16_slow_r0", "d4_k08_slow_r0", "d4_k08_slow_r2"]
EPS = 1e-6


def scaled(tree, s):
    return dict(tree, M=(np.array(tree["M"], float) * np.exp(s)).tolist())


def main():
    out = []
    summ = []
    for tag in TREES:
        tree = json.load(open(os.path.join(ROOT, "sim", "data", tag + ".json")))
        t0 = time.time()
        r = minimize_scalar(lambda s: -esco_dense.loglik(scaled(tree, s)), bounds=(-3, 3), method="bounded",
                            options={"xatol": 1e-3})
        s_star = float(r.x)
        grid = s_star + np.linspace(-0.4, 0.4, 17)
        D, T, N = [], [], []
        for s in grid:
            tr = scaled(tree, s)
            D.append(esco_dense.loglik(tr))
            dt = esco_tt.loglik_detail(tr, eps=EPS)
            T.append(dt["loglik"])
            N.append(dt["n_round"])
            out.append(dict(tree=tag, s=s, dense=D[-1], tt=T[-1], err=T[-1] - D[-1], n_round=N[-1], rank=dt["max_rank"]))
            print(out[-1], flush=True)
        D, T, N = map(np.array, (D, T, N))
        x = grid - s_star
        fine = np.linspace(-0.4, 0.4, 8001)
        pd_, pt_ = np.polyfit(x, D, 4), np.polyfit(x, T, 4)
        ad, at = fine[np.argmax(np.polyval(pd_, fine))], fine[np.argmax(np.polyval(pt_, fine))]
        curv = -2 * pd_[2]  # 관측 정보량(로그 축) ≈ −d²ℓ/ds² at s*
        err = T - D
        slope = np.polyfit(x, err, 1)[0]
        cN = np.polyfit(N * EPS, err, 1)[0] if np.ptp(N) > 0 else np.nan
        rec = dict(tree=tag, s_star_dense=s_star, argmax_shift_tt_minus_dense=float(at - ad),
                   predicted_shift=float(slope / curv), err_min=float(err.min()), err_max=float(err.max()),
                   err_range=float(np.ptp(err)), err_slope_per_logm=float(slope), N_min=int(N.min()), N_max=int(N.max()),
                   c_fit=float(cN), info_logm=float(curv), lmax_diff=float(np.polyval(pt_, at) - np.polyval(pd_, ad)),
                   sec=time.time() - t0)
        summ.append(rec)
        print(rec, flush=True)
    with open(os.path.join(HERE, "rounding_bias_grid.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(HERE, "rounding_bias_grid_summary.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)


if __name__ == "__main__":
    main()
