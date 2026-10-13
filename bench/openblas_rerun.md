# Accelerate → OpenBLAS 재계산 (2026-10-06)

배경: 10-06 03:38 LOG 항목. Mac `.venv`의 numpy/scipy가 링크한 Apple Accelerate LAPACK이 힙을 망가뜨려 숫자를 조용히 오염시킬 수 있었다. `.venv`는 03:40에 같은 버전(numpy 2.5.3, scipy 1.18.1)의 scipy-openblas 휠로 바뀌었다. 이 문서는 그 전에 Mac에서 낸 수치를 지금의 `.venv`로 다시 계산해 옛 값과 비교한 결과다.

- 환경 확인: `numpy.show_config(mode='dicts')['Build Dependencies']['lapack']['name']`와 scipy 쪽 모두 `scipy-openblas`.
- 실행 조건: OMP/OPENBLAS/VECLIB 스레드 1, `nice 10`. 다른 세션 때문에 부하 평균이 15–570이었다. 작업당 CPU는 8–60%만 받았으므로 시간 열은 비교에 쓰지 말 것.
- 드라이버·로그·비교 스크립트: `bench/openblas_rerun/` (`compare.py`, `tt_stats.py`, `xcheck_compare.py`, `*_driver.py`, `*.log`). 기존 스크립트는 수정하지 않았다. 드라이버는 기존 스크립트의 `run()`을 import해 쓰고, 출력만 새 파일로 보낸다.
- Kaggle은 쓰지 못했다(CPU 세션이 모두 사용 중). 이 절의 값은 전부 로컬 Mac(OpenBLAS) 계산이다.

## 결론

**옛 Accelerate 값이 조용히 틀린 경우는, 이미 알려진 d4_k14_medium_r1 eps 1e-7(−48.550) 하나뿐이다.** 그 밖의 재계산 값은 모두 옛 값과 |Δ| ≤ 1.1e-10으로 같다(TT 랭크·스텝 수까지 같음). 하나 예외가 더 있는데, 결과값이 아니라 오차추정 칸의 문제다. `tt_vs_dense.csv` 한 행에서 `err_trunc = inf`가 나왔다(eps/100 보조 런이 −inf를 냄). OpenBLAS에서는 유한값(1.8e-6)이다. 이 행은 보고된 통계를 바꾸지 않는다(아래).

| 파일 | 비교한 행 | max abs(old − new) | 1e-9 넘는 행 |
|---|---|---|---|
| `dense_vs_mascot.csv` → `dense_vs_mascot_openblas.csv` | 243/243 | ll_dense 0, ll_mascot 1e-10 (CSV가 소수 10자리) | 없음 |
| `tt_vs_dense.csv` → `tt_vs_dense_openblas.csv` | 181/181 (+ 옛날 미완 13그루 새로) | ll_dense 7.1e-15, ll_tt 1.1e-10, ll_tt_h 5.4e-11, err_step 1.0e-11; max_rank 181행 모두 같음 | `err_trunc`: d3_k18_medium_r2 old **inf**, new 1.81e-6 |
| `rank_study.csv` → `openblas_rerun/rank/*.csv` | 44/48 (4행은 2 h 상한으로 중단) | loglik 1.0e-10 (d3 k40 m1 1e-8); max_rank, n_steps, n_mig 44행 모두 같음 | 없음 |
| `kaggle/xcheck.csv`의 TT·MASCOT 값 → `openblas_rerun/xcheck_openblas.csv`, `xcheck_mascot_openblas.csv` | MASCOT 13/13, TT 1e-6 6/11, TT 1e-7 5/11 | 3.3e-11 | 없음 (재계산한 행 안에서) |

## 파일별

### dense_vs_mascot.csv
- 243그루 전부 재계산했다(dense 228 + MASCOT 243, 상한 4^12는 그대로). ll_dense는 출력 자릿수(1e-10)까지 전부 같다. ll_mascot은 최대 1e-10(마지막 자리 반올림) 차이다.
- `dense_vs_mascot.md`의 숫자는 바뀌지 않는다. 그래서 편집하지 않았다.

### tt_vs_dense.csv (eps 1e-8, Richardson, 오차추정)
- 옛 181그루를 모두 다시 계산했다. 결과값(ll_tt, ll_tt_h, diff, err_step)은 ≤1.1e-10으로 같고, max_rank도 모두 같다.
- **d3_k18_medium_r2의 `err_trunc`**: 옛 파일은 `inf`였고 새로는 1.81e-6이다. `inf`는 eps/100(=1e-10) 보조 런이 어느 합류에서 질량 합 `s ≤ 0`(또는 NaN)을 만나 −inf를 돌려줬다는 뜻이다. 같은 코드가 OpenBLAS에서는 정상으로 끝나므로, Accelerate 손상의 또 다른 흔적일 가능성이 높다. 이 행은 커버리지 통계를 바꾸지 않는다. (err_step + err_trunc) = 4.4e-6 ≥ |diff| 3.5e-6이라 1배 기준으로도 덮인다. `rank_study.md` 1절 표(최대오차 1.0e-6 / 2.0e-4 / 3.4e-6, 최대 랭크 29/98/86, 중앙값 9e-8)와 커버리지 3배 기준 95.6%(LOG의 "96%"), 1배 기준 65.7%(심사의 "66%"), Richardson 악화 11.6%는 새 파일로 다시 집계해도 그대로다.
- 옛날에 부하 때문에 못 한 24그루 중 13그루를 이번에 새로 계산했다. 비교할 옛 값이 없으므로 위 표에는 넣지 않았다. 194그루로 넓히면 d=4는 39그루, kmax 10, 최대 |Δ| 3.6e-5(d4_k10_slow_r0), 최대 랭크 86이다. d=3은 74그루, 최대 랭크 103이고, 커버리지는 3배 기준 95.4%, 1배 기준 64.9%다. 나머지 11그루(d3_k16_medium_r1, d4_k12_medium_r0/r1/r2, d4_k12_slow_r0/r2, d4_k14_fast_r2, d4_k14_slow_r1, d4_k16_fast_r0, d4_k16_slow_r2, d4_k20_slow_r2)는 부하 때문에 중단했다(선택 작업).

### rank_study.csv
- 48행 모두 `rank_study.md`의 표(eps별 랭크 표, 쌍 비교 표, 유실 재실행 표, 본문)에 들어간다. 그래서 전부 다시 돌렸고, 작업마다 2시간 상한을 걸었다.
- 끝난 44행은 loglik ≤1.0e-10이고 랭크·스텝 수가 같다.
- **2 h 상한으로 중단(미재계산)한 4행.** 사건별 verbose 로그를 옛 `bench/rank56/*.log`와 사건 단위로 대조했다.

| config (d,k,m,eps) | 옛 값 | 중단 시점 | 중단 전 사건 대조 |
|---|---|---|---|
| 4,40,1,1e-8 | rank 209, ll −82.365914 | 28/39 합류(t=0.443) | ll(소수 6자리)·사건 랭크 28개 전부 같음 |
| 6,10,1,1e-8 | rank 437, ll −34.764112 | 2/9 합류 | ll 같음. 사건 랭크 230 vs 231(±1) |
| 6,10,10,1e-8 | rank 225, ll −26.284494 | 1/9 합류 | ll 같음. 사건 랭크 187 vs 186 |
| 6,20,1,1e-6 | rank 143, ll −41.915446 | 11/19 합류 | 앞 9개 사건 완전히 같음. 10·11번째 사건 랭크 130 vs 131, ll 마지막 자리 1e-6 |

  ±1 랭크 차이는 절단 경계에 걸린 특이값 하나가 백엔드 반올림에 따라 남거나 잘린 것이다. 오염이라고 보기 어렵다. 같은 런의 ll이 6자리까지 같다.
- `rank_study.md`의 랭크·loglik 숫자는 바뀌지 않는다. 바꾼 곳은 아래 xcheck 표의 k14 칸 하나다.

### kaggle/xcheck.csv (보고서 `kaggle_xcheck.md`, `rank_study.md` 3절 표)
- **출처 정정.** 파일 이름과 표 제목은 "Kaggle"이다. 하지만 dense 열(`kaggle/dense_big.csv`)만 Kaggle에서 계산했다. TT(1e-6, 1e-7)와 MASCOT 값은 10-05에 Mac `.venv`(Accelerate)에서 `xargs`로 돌렸다(`kaggle/xcheck.err`의 `.venv/bin/python` 기록, 03:38 LOG의 macOS 크래시 리포트). 그래서 재계산 대상에 넣었다.
- 재계산 결과(OpenBLAS, 옛 값과 |Δ|).
  - MASCOT 13개 전부(11그루 + k50/k60 대형). ≤1.3e-11.
  - TT 1e-6: k16_fast_r1, k16_slow_r1, k18_fast_r1, k18_fast_r2, k18_medium_r2, k20_slow_r1. ≤6.6e-12.
  - TT 1e-7: k16_fast_r1, k16_slow_r1, k18_fast_r1, k18_fast_r2, k18_slow_r1, k20_slow_r1. ≤3.3e-11.
  - 03:38 디버깅 항목이 OpenBLAS로 이미 재현한 값(별도 확인). k14_medium_r1 1e-7 = **−41.30912114229258**(dense와 4.8e-6, 옛 Accelerate 값 −48.550은 오답). k18_medium_r2 1e-7 = −46.092419891767584(옛 값과 1.5e-12).
- **미재계산**(2 h 상한 또는 부하 중단): TT 1e-6 k14_medium_r1, k18_medium_r0, k18_medium_r1, k18_slow_r1, k20_medium_r2; TT 1e-7 k18_medium_r0, k18_medium_r1, k20_medium_r2. 1프로세스 순차 재실행(`xcheck_driver2.log`, 작업 목록 `xcheck_jobs_pass2.txt`)이 이 문서를 쓰는 지금도 돌고 있다. 끝나면 `python3 bench/openblas_rerun/xcheck_compare.py`로 대조할 것.
- 바뀐 보고 숫자: k14_medium_r1의 "TT 1e-7 오차" 칸이 7.24(오답) → **4.8e-6**. 그 밖의 칸은 재계산한 범위에서 모두 같다. 열 최대값(1e-6 ≤ 6.0e-3, 1e-7 ≤ 9.1e-4)과 MASCOT−dense 열도 그대로다.

## 바뀌는 주장
- 결과값을 바꾸는 것은 없다. dense 정확값, MASCOT 편향 표, TT–dense 대조 통계, 랭크 표, 대형 MASCOT 값은 OpenBLAS로 같은 값이 나온다(재계산한 범위에서 ≤1.1e-10).
- 남는 정정은 둘이다. (1) xcheck의 eps 1e-7 k14 오답은 Accelerate 오염이었고, OpenBLAS 값은 dense와 4.8e-6이다. 따라서 "eps 1e-7에서 13–14개 계통 오차 ≤ 9.1e-4"는 이 행을 포함해도 성립한다. 다만 위 미재계산 행이 끝나야 표 전체가 OpenBLAS 값이 된다. (2) "Kaggle 대조"라는 이름과 달리 TT·MASCOT 열은 Mac 계산이었다.

## 마감 (2026-10-08)
- `xcheck_driver2.log`의 8작업이 모두 끝남(`XCHECK_DONE`). `xcheck_compare.py` 결과: 새로 계산한 칸 전부 옛 값과 |Δ| ≤ **7.0e-11**.
- compare가 None으로 표시하는 두 칸(k14_medium_r1·k18_medium_r2의 1e-7)은 위 03:38 항목에서 따로 재현한 값을 쓴다: −41.30912114229258(dense와 4.8e-6), −46.092419891767584(dense와 3.9e-6).
- 그래서 xcheck TT·MASCOT 22+13칸 모두 OpenBLAS로 확인됨. 옛 Accelerate 값 중 틀린 것은 k14_medium_r1 1e-7 하나뿐이라는 결론은 그대로다. 열 최대값(1e-6 ≤ 6.0e-3, 1e-7 ≤ 9.1e-4)도 바뀌지 않음.
