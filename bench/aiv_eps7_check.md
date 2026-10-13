# AIV 핵심 봉우리 eps 1e-7 확인 (2026-10-13, 심사 대응)

**논문에 넣을 3줄 결론.**
1. 결론을 정하는 AIV 봉우리의 정제 모수에서 TT를 eps 1e-7로 다시 평가하면, 비대칭 봉우리 6개(masco ans 뿌리 허브·gal 뿌리, sisco·lisco gal·cha 뿌리)의 logL은 eps 1e-6 값과 |Δ| ≤ 3.2e-3(sisco cha, 그 밖은 ≤ 8.1e-4)으로 같다.
2. 가장 크게 움직인 것은 N_round가 가장 많은 masco 대칭 허브(−296.304 → −296.476, −0.17; h/2 짝 오차 0.65 안)이고, 그 결과 masco 정확 LRT는 29.59 → 29.94로 조금 커진다(MASCOT 14.1). 봉우리 차이(masco 허브 − gal 4.673 → 4.672, sisco gal − cha 4.570 → 4.567, lisco 3.793 → 3.794)는 그대로라서 sisco·lisco의 순위 역전은 eps에 둔감하다.
3. 정확 LRT와 봉우리 순서는 eps에 둔감하다. [대기: 세 나무의 대칭 보통 해 — 결과가 들어오면 이 줄을 고칠 것]

재생성: 점 파일 `bench/aiv_eps7/make_points.py` → `bench/aiv_eps7/points/*.json`(정제 모수 그대로), Kaggle 결과 `kaggle/out/aiveps7/<계정>-eps7-<점>/eps7_<점>.json`, 요약 `.venv/bin/python bench/aiv_eps7/summarize.py` → `bench/aiv_eps7/eps7_results.csv`. Kaggle: 점 하나당 커널 하나(4 CPU, 평가는 단일 프로세스, BLAS 스레드 1), 코드는 현재 `tt/`·`bench/realdata2_eval.py`, 데이터셋 `<계정>/esco-aiveps7`. masco gal 뿌리는 같은 정제점의 기존 eps 1e-7 값(`bench/realdata2/aivorder_pair_B_e7.json`, 2026-10-06)을 썼다.

## 점 고르기
- 정제가 수렴한 봉우리는 `_fit.json`의 모수를 썼다.
- '최신 점'(아직 하한인 봉우리)은 정제 JSON에 저장된 마지막(최고) 반복점이다. 그 뒤 줄 탐색 시도점은 θ가 저장되지 않아 쓸 수 없었다.
  - sisco cha 뿌리: 반복점 −270.6295. 문서의 하한 −270.625는 그 뒤 a = 0.25 시도점이다(차 0.005).
  - lisco 대칭 보통 해: 반복점 −274.0767. 문서의 하한 −274.044는 그 뒤 a = 1 시도점이다(차 0.033). 그래서 이 표의 lisco LRT(31.01)는 문서의 ≤ 30.94와 0.07 다르다.

## 결과

| 나무 | 봉우리 | eps 1e-6 | eps 1e-7 | 1e-7 − 1e-6 | 랭크 (1e-7) | 시간 (s) | 뿌리 (1e-7, ans/cha/gal) | 기존 h/2 짝 오차 |
|---|---|---|---|---|---|---|---|---|
| masco | ans 뿌리 허브 | −281.50717 | −281.50692 | +2.5e-4 | 27 | 830 | 1.000 / 0.000 / 0.000 | 2.4e-4 |
| masco | gal 뿌리 | −286.17996 | −286.17915 | +8.1e-4 | 35 | 1,486 | 0.000 / 0.001 / 0.999 | 4.5e-4 |
| masco | 대칭 허브 | −296.30429 | −296.47569 | **−0.171** | 26 | 7,292 | 0.100 / 0.022 / 0.877 | 0.65 |
| masco | 대칭 보통 해 | −301.92965 | 대기 | | | | | 2.1e-3 |
| sisco | gal 뿌리 | −266.05994 | −266.05979 | +1.5e-4 | 31 | 2,042 | 0.000 / 0.001 / 0.999 | 2.4e-4 |
| sisco | cha 뿌리 (반복점) | −270.62950 | −270.62630 | +3.2e-3 | 92 | 13,608 | 0.000 / 0.937 / 0.063 | — |
| sisco | 대칭 보통 해 | −281.04732 | 대기 | | | | | 2.6e-2 |
| lisco | gal 뿌리 | −258.57188 | −258.57133 | +5.4e-4 | 31 | 2,141 | 0.000 / 0.001 / 0.999 | 7.2e-4 |
| lisco | cha 뿌리 | −262.36525 | −262.36541 | −1.6e-4 | 77 | 10,432 | 0.000 / 0.934 / 0.067 | 3.0e-3 |
| lisco | 대칭 보통 해 (반복점) | −274.07665 | 대기 | | | | | — |

## 결론을 정하는 양

| 나무 | 양 | eps 1e-6 | eps 1e-7 | MASCOT |
|---|---|---|---|---|
| masco | 비대칭 최고(ans 허브) − gal 뿌리 | 4.673 | 4.672 | 3.09 (−281.478 vs −284.569) |
| masco | LRT 2·(ans 허브 − 대칭 허브) | 29.59 | **29.94** | 14.1 |
| sisco | gal 뿌리 − cha 뿌리 | 4.570 | 4.567 | −0.035 (cha가 위) |
| sisco | LRT 2·(gal − 대칭 보통 해) | 29.97 | 대기 | 18.8 |
| lisco | gal 뿌리 − cha 뿌리 | 3.793 | 3.794 | −0.245 (cha가 위) |
| lisco | LRT 2·(gal − 대칭 보통 해 반복점) | 31.01 | 대기 | 20.4 |

## 해석
- 비대칭 봉우리는 eps를 10배 줄여도 ≤ 8e-4만 움직인다. 봉우리 사이 차이(3.8–4.7)와 비교하면 무시할 크기다. 뿌리 분포도 셋째 자리까지 같다.
- masco 대칭 허브는 −0.17 움직였다. 이 봉우리는 cha–gal 이동이 22/년이라 N_round가 37.7만으로 가장 크고(`bench/rounding_bias.md` (c)), 예전 h/2 짝 오차도 0.65로 가장 컸다. 움직임은 그 오차 안이고, 방향(정확값이 더 낮음)도 h/2 쪽과 같다. 그래서 LRT는 약해지지 않고 오히려 커진다.
- lisco cha 뿌리는 1e-7에서 랭크가 77로(gal 31의 2.5배) 커서 2.9 h 걸렸지만 값은 −1.6e-4만 바뀌었다.
- sisco cha 뿌리도 1e-7에서 랭크 92(3.8 h)였고 +3.2e-3만 바뀌었다.
- 남은 대칭 보통 해 3개는 Kaggle에서 도는 중이다(아래). 대칭 보통 해의 h/2 오차가 2e-3–3e-2였으므로 LRT가 0.1 넘게 움직일 것으로 보지 않는다. 결과가 오면 위 표와 3줄 결론의 3번을 채운다.

## Kaggle 커널
| 커널 | 점 | 상태 (2026-10-13 20:45) |
|---|---|---|
| kaggle/eps7-masco-asym-hubans | masco ans 허브 | COMPLETE |
| kaggle/eps7-masco-sym-hub | masco 대칭 허브 | COMPLETE |
| kaggle/eps7-masco-sym-regular | masco 대칭 보통 해 | RUNNING |
| kaggle/eps7-sisco-asym-gal | sisco gal 뿌리 | COMPLETE |
| kaggle/eps7-sisco-asym-cha | sisco cha 뿌리 | COMPLETE |
| kaggle/eps7-sisco-sym-regular | sisco 대칭 보통 해 | RUNNING |
| kaggle/eps7-lisco-asym-gal | lisco gal 뿌리 | COMPLETE |
| kaggle/eps7-lisco-asym-cha | lisco cha 뿌리 | COMPLETE |
| kaggle/eps7-lisco-sym-regular | lisco 대칭 보통 해 | RUNNING |

수합: `kaggle/aiveps7/collect.sh loop`(nohup)가 10분마다 끝난 커널을 받고 장부 상태를 바꾼다. 다 끝나면 collect.log에 ALL_COLLECTED를 남기고 멈춘다. 그 뒤 `summarize.py`를 돌려 이 문서의 '대기' 칸을 채운다. 12 h 한도에 걸려 CANCEL되면 그 점은 결론에 필요 없는 확인이므로 다시 올리지 않고 "미완"으로 적는다.
