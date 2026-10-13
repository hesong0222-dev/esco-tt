# TT·MASCOT vs dense (Kaggle, 동시 계통 13~14)

주의: dense 열만 Kaggle 계산이다. TT·MASCOT 값(`kaggle/xcheck.csv`)은 10-05 Mac(Accelerate)에서 돌았고, OpenBLAS로 재계산한 행은 모두 ≤3.3e-11로 같다(OpenBLAS 재계산, `bench/openblas_rerun.md`).

| 계통수 | 동시 | dense | TT 1e-6 오차 | TT 1e-7 오차 | MASCOT−dense |
|---|---|---|---|---|---|
| d4_k14_medium_r1 | 13 | -41.309116 | 8.1e-05 | 4.8e-06 (OpenBLAS 재계산; Accelerate 런은 −48.550 오답) | +0.214 |
| d4_k16_fast_r1 | 13 | -43.633813 | 1.1e-05 | 9.7e-07 | +0.005 |
| d4_k16_slow_r1 | 13 | -50.481523 | 2.2e-03 | 9.1e-04 | -0.091 |
| d4_k18_fast_r1 | 13 | -43.266281 | 3.2e-06 | 2.1e-07 | +0.042 |
| d4_k18_fast_r2 | 13 | -43.505705 | 1.8e-06 | 5.4e-07 | +0.025 |
| d4_k18_medium_r0 | 13 | -51.694801 | 2.0e-04 | 2.6e-05 | -0.029 |
| d4_k18_medium_r1 | 14 | -52.384007 | 2.8e-04 | 1.2e-05 | -0.285 |
| d4_k18_medium_r2 | 13 | -46.092416 | 9.1e-05 | 3.6e-06 | -0.088 |
| d4_k18_slow_r1 | 14 | -53.168403 | 5.8e-03 | 7.7e-04 | +0.044 |
| d4_k20_medium_r2 | 13 | -53.712660 | 1.6e-04 | 1.1e-05 | -0.501 |
| d4_k20_slow_r1 | 13 | -49.400618 | 6.0e-03 | 2.7e-04 | -3.586 |

## 대형 (dense 불가): TT(eps 1e-6, Kaggle) vs MASCOT
- d4_k50_medium_r0.json: TT -123.1554 (h/2와 차 2.4e-3, k50), MASCOT -122.9450, MASCOT−TT +0.210
- d4_k60_medium_r0.json: TT -132.1085 (h/2와 차 2.4e-3, k50), MASCOT -131.7602, MASCOT−TT +0.348

## 정정 (2026-10-06)
- d4_k14_medium_r1의 eps 1e-7 칸은 처음에 "—"(segfault)로 적었으나 잘못이었다. 해당 런은 32501초 뒤 끝났고 값이 −48.550으로 dense(−41.309)와 7.24 차이 나는 **오답**이다(독립 심사가 발견). 원인 조사 중.
- 따라서 "eps 1e-7에서 오차 ≤ 1e-3" 주장은 철회. 큰 나무 오차 막대(h/2 차이 2.4e-3)도 Richardson 해석이 틀렸다는 심사 지적이 있음(review/REVIEW.md).
