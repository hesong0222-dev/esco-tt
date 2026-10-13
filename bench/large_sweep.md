# Large-tree sweep: exact TT (eps 1e-6) vs MASCOT, with a paired error run per tree

Source: `bench/large_sweep.csv`, rebuilt by `.venv/bin/python bench/large_sweep_build.py` from every `headline.py` JSON
under `kaggle/out/` plus `kaggle/large*_mascot.csv` (2026-10-06 21:45 KST). Paper numbers: `paper/figures/stats.py`.

Trees: `sim/data/large` (d=4, k 30–60, r0/r1, 24), `sim/data/large4x` (d=4 k40 r2/r3, 6), `sim/data/large3`
(d=3, k 30–60, r0/r1, 24) = 54. Peak simultaneous lineages 15–35.

## Completeness
- Main run (eps 1e-6, step h): **53/54**. Missing: `d4_k60_medium_r1` — did not finish within Kaggle's 12 h limit
  (`kaggle/esco-tt-large-l2`, stopped at t = 1.22 of 10.0, k = 17, rank 155, 11.65 h). Its h/2 run
  (`kaggle/esco-tt-large-sm2-c2`) also stopped (t = 1.89, rank 126). Not feasible in one ≤ 9 h kernel with the
  current numpy implementation; the tree is excluded from every table.
- Paired error run (eps 1e-6, h/2): **53/53**. 23 on Kaggle (kaggle sm2-c0..c4, k4), 30 on the local Mac
  (`kaggle/out/local-mac-fill/`, OpenBLAS .venv, same code as the Kaggle runs; all d=3 trees and the six large4x
  trees). A local rerun of `d3_k50_slow_r1` at h reproduced the Kaggle value to 2e-13.
- eps 1e-7 at h (second error proxy): 5 trees, d4 k30 (slow r0/r1, medium r1, fast r0/r1). `d4_k30_medium_r0` eps 1e-7
  stopped at the 12 h limit (t = 1.24, rank 244).

## Per d × regime

| d | regime | n | MASCOT−exact median | range | rank median | rank max | n paired (h/2) | max abs(h − h/2) |
|---|---|---|---|---|---|---|---|---|
| 3 | slow | 8 | -0.123 | -0.76 … +0.63 | 9 | 11 | 8 | 2.4e-02 |
| 3 | medium | 8 | -0.064 | -0.61 … +0.56 | 48 | 59 | 8 | 3.8e-03 |
| 3 | fast | 8 | +0.037 | -0.09 … +0.09 | 32 | 40 | 8 | 1.5e-04 |
| 4 | slow | 10 | -0.336 | -1.20 … +0.58 | 18 | 40 | 10 | 2.6e-02 |
| 4 | medium | 9 | -0.027 | -0.63 … +0.95 | 123 | 173 | 9 | 3.9e-03 |
| 4 | fast | 10 | +0.030 | -0.08 … +0.12 | 48 | 62 | 10 | 3.0e-04 |

eps 1e-7 − eps 1e-6 (same h): d4_k30 fast r0/r1 6e-5, 4e-5; medium r1 5.2e-4; slow r0/r1 4.8e-4, 2.1e-3.

## Notes
- The paired error is largest in the **slow** regime (h−h/2 up to 2.6e-2, d4_k40_slow_r0; 2.4e-2, d3_k50_slow_r1),
  not in medium where the rank is highest. Fast trees: ≤ 3e-4.
- |MASCOT−exact| exceeds 3× the paired error on 52/53 trees. Exception: `d3_k50_slow_r1`, gap +0.029 vs
  |h−h/2| 0.024 (ratio 1.2) — MASCOT and exact are not resolved there. Next-smallest ratio 22 (d4_k40_slow_r0).
  The old minimum ratio ≥ 31 (16 trees) no longer holds.
- |MASCOT−exact| > 0.1 on 34/53 trees (64%).
- h/2 runs usually reach a lower max rank than h runs (e.g. d4_k40_medium_r2 126 → 95; d4_k60_medium_r1 partial
  155 vs 126 at the same t). Smaller steps → smaller per-step truncation growth. Not studied further.
- `sec` (main runs) are Kaggle wall times with 4 jobs per 4-core kernel. `sm2_sec` of the local runs come from a shared
  Mac under load 5–900 and are not comparable.
- Code: Kaggle runs used the tt/ snapshot of 2026-10-05 (ds3*). The B1 underflow fix (commit 8a30539) is bit-identical
  on non-underflowing trees (LOG 2026-10-06 15:50); no large tree comes near underflow (LOG: H ≤ 86).
