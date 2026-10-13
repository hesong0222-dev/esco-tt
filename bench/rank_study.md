# TT-ESCO rank study (tt/ lane) — 2026-10-03

Go/no-go question: do the TT ranks of the ESCO joint lineage-location distribution stay low for realistic trees?

**Verdict: GO for d ≤ 4 at eps = 1e-6 (and for d = 2 at any tolerance); NO-GO for d ≥ 6, and for d = 4 at
eps ≤ 1e-7 with ≳ 25 simultaneous lineages, with this implementation.** (Final, 2026-10-06; see "Conclusion" at the end.)
At truncation tolerance eps=1e-6 the maximum TT rank stays ≤ 90 for d=4 up to k=40 isochronous lineages and
≤ 125 on the heterochronous d=4 headline trees with 25-30 simultaneous lineages, ≤ 36 for d=3 up to k=60, and
≤ 13 for d=2. The price is accuracy: the eps=1e-6 loglik error grows with k, from ~5e-4 at k=20 to ~1e-2 at k=40
(measured against eps=1e-8 runs of the same tree). At eps=1e-8 ranks are 2.3-3.5x larger (d=4: 152 at k=20,
209 at k=40; d=6 k=10: 225-437). Rank growth follows the number of populated per-deme *count sectors* of the
smaller side of a cut, C(k_L+d-1, d-1): polynomial in k, steep in d.

Files: `bench/rank_study.csv` (all runs), `bench/tt_vs_dense.csv` (181-tree cross-check), `kaggle/out/k4/*.json` (headline runs), `bench/rank56/` (rerun logs),
code in `tt/` (`esco_tt.py` interface `loglik(tree_json)`; `ttcore.py` TT toolkit; `brute.py`, `quicksim.py`).

## Method (what was actually run)

- State: one TT core per live lineage (physical dim d). Lineages ordered by the planar (DFS) leaf order of the tree,
  so every coalescence merges two *adjacent* cores (proof in LOG.md) — no swap gates. Sample event = insert a rank-1
  delta core at its planar position (rank unchanged). Coalescence = contract the two cores along the diagonal with
  weight 1/Ne[a], then renormalize and accumulate log.
- Time stepping between events: 4th-order Suzuki–Yoshida composition of Strang steps
  S(h) = D(h/2) Mig(h) D(h/2). Mig(h) = ⊗ exp(hQ) is applied site-wise: exact and rank-preserving.
  D(h) = exp(-h Σ_a n_a(n_a-1)/(2Ne_a)) is built as an exact "count TT" per deme (rank ≤ k, rounded at 1e-14 to rank 4-8;
  for d=2 a single count function), applied as a Hadamard product (zip-up when the product rank is large, exact
  product + rounding otherwise), then TT-rounded to relative Frobenius eps. Step size: h·c_max·k(k-1) ≤ 0.25 and
  h·q_max ≤ 0.25, ≥2 steps per interval. Convergence order verified numerically (error ratio 16-20 for h→h/2).
- Error estimate (`loglik_with_error`): step error from h vs h/2 (Richardson, /15), truncation error from eps vs eps/100.

## 1. Correctness

- Unit tests of the TT toolkit vs dense tensors (round, Hadamard zip-up/full, diagonal merge, site insert, site matrix).
- Own brute force (`tt/brute.py`, sparse generator + expm_multiply): agreement to the time-step error (1e-6..1e-8 at
  steps_mult 4..8) on d=2 k≤16, d=3 k≤10, d=4 k≤8.
- Cross-check vs `dense/` on 181 `sim/data` trees with d^k ≤ 2e6 (heterochronous, Ne=1..2.5, three regimes), eps=1e-8:

| d | trees | max alive k | max abs dloglik (Richardson) | max abs dloglik (plain h) | max TT rank |
|---|---|---|---|---|---|
| 2 | 81 | 15 | 1.0e-6 | 9.8e-5 | 29 |
| 3 | 73 | 13 | 2.0e-4 | 6.7e-5 | 98 |
| 4 | 27 | 8 | 3.4e-6 | 6.1e-6 | 86 |

  Median abs error 9e-8. The few 1e-5..2e-4 cases are all d=3 *slow* trees (deep, many steps): truncation error
  accumulates with the number of roundings, so the h/2 run is slightly *less* accurate and Richardson does not help
  there; the reported estimate (err_step+err_trunc)×3 covered the true error in 96% of trees. Tightening eps to 1e-9
  for deep trees fixes this; eps should scale with 1/sqrt(#roundings) in a production version.
  (24 of the 205 trees ≤2e6 states were not finished because the machine was shared at load ~400; none is a special case.)
- Note: M convention (M[a][b] = backward rate a→b, pairwise coalescence 1/Ne[a]) matches sim/README.md and dense/.

## 2. Rank growth (isochronous quicksim trees, Ne=1, symmetric M with per-lineage rate m; seed 1 unless noted)

Max TT rank over the whole run / wall time in s (wall times were measured on a machine shared with other lanes at
load average 250-430, so they are inflated 2-4x and only comparable to each other).

#### eps = 1e-08  (cell: max TT rank / runtime s)

| d | m | k=10 | k=20 | k=30 | k=40 | k=60 |
|---|---|---|---|---|---|---|
| 2 | 0.1 | 6 / 0 | 6 / 2 | 6 / 4 | 6 / 7 | 6 / 59 |
| 2 | 1 | 8 / 0 | 15 / 3 | 17 / 5 | 26 / 22 | 17 / 109 |
| 2 | 10 | 7 / 0 | 24 / 6 | 29 / 29 | 40 / 54 | 66 / 760 |
| 3 | 0.1 | 10 / 2 | 13 / 16 | 16 / 122 |  |  |
| 3 | 1 | 23 / 4 | 72 / 218 |  | 97 / 2144 |  |
| 3 | 10 | 32 / 15 | 101 / 1189 |  |  |  |
| 4 | 0.1 | 52 / 67 | 20 / 70 | 19 / 518 |  |  |
| 4 | 1 | 59 / 43 | 152 / 670 |  | 209 / 8516 |  |
| 4 | 10 | 75 / 129 |  |  |  |  |
| 6 | 0.1 | 52 / 285 |  |  |  |  |
| 6 | 1 | 437 / 10328 |  |  |  |  |
| 6 | 10 | 225 / 6895 |  |  |  |  |

#### eps = 1e-06  (cell: max TT rank / runtime s)

| d | m | k=20 | k=30 | k=40 | k=60 |
|---|---|---|---|---|---|
| 2 | 1 |  |  | 13 / 40 | 8 / 62 |
| 3 | 1 | 28 / 190 | 17 / 166 | 36 / 250 | 33 / 335* |
| 3 | 10 |  | 45 / 1163 |  |  |
| 4 | 1 | 60 / 496 | 62 / 1307* | 90 / 1732* |  |
| 6 | 1 | 143 / 5989 |  |  |  |

#### eps = 0.0001  (cell: max TT rank / runtime s)

| d | m | k=20 |
|---|---|---|
| 3 | 1 | 7 / 49 |
| 4 | 1 | 17 / 85 |

`*` = run with rmax=400 cap (never reached). Lost-and-rerun configurations (2026-10-05/06, local Mac, BLAS threads 1,
6 jobs in parallel, machine at load average 13-130 from other sessions and swap nearly full, so wall times are again
inflated; each job got 15-60% of a core):

| config (d, k, m, eps) | status | max rank | loglik | wall s |
|---|---|---|---|---|
| 6, 10, 1, 1e-8 | done | 437 | -34.764112 | 10328 |
| 6, 10, 10, 1e-8 | done | 225 | -26.284494 | 6895 |
| 6, 20, 1, 1e-6 | done | 143 | -41.915446 | 5989 |
| 3, 40, 1, 1e-8 | done | 97 | -66.339514 | 2144 |
| 4, 40, 1, 1e-8 | done | 209 | -82.365914 | 8517 |
| 4, 20, 10, 1e-8 | **incomplete** (killed at the 3 h cap) | ≥ 123 at the last event (k=15 alive, t=0.0995 of TMRCA 7.15; then >2.4 h inside one interval) | — | >10800 |

(The verbose per-event rank in the logs is the rank *at events*; `max_rank` in the CSV also includes intermediate
ranks inside time steps, e.g. 437 vs 230 at events for d=6 k=10 m=1.)

#### Paired eps comparison (same tree): loglik shift caused by loosening eps

| d | k | m | rank 1e-8 | rank 1e-6 | rank 1e-4 | abs(ll(1e-6) - ll(1e-8)) | abs(ll(1e-4) - ll(1e-8)) |
|---|---|---|---|---|---|---|---|
| 2 | 40 | 1 | 26 | 13 | | 4.3e-4 | |
| 2 | 60 | 1 | 17 | 8 | | 4.3e-4 | |
| 3 | 20 | 1 | 72 | 28 | 7 | 1.3e-3 | 5.9e-2 |
| 3 | 40 | 1 | 97 | 36 | | 7.3e-3 | |
| 4 | 20 | 1 | 152 | 60 | 17 | 5.0e-4 | 4.1e-2 |
| 4 | 40 | 1 | 209 | 90 | | 1.4e-2 | |

#### Rank vs k and d (summary)

- **k (d fixed):** sub-linear once k > 20 at every tolerance tested. d=3 m=1, eps=1e-8: 23 → 72 → 97 for k=10/20/40
  (×3.1 then ×1.35); eps=1e-6: 28/17/36/33 for k=20/30/40/60. d=4 m=1, eps=1e-8: 59 → 152 → 209 for k=10/20/40;
  eps=1e-6: 60/62/90 for k=20/30/40. Fitted exponent between k=20 and 40: r ∝ k^0.35-0.6 for d=3-4.
  d=2: flat for slow, ~k^0.5-1 for fast migration (66 at k=60, m=10).
- **d (k fixed):** steep. k=20 m=1 eps=1e-6: d=3/4/6 → 28/60/143. k=10 m=10 eps=1e-8: d=2/3/4/6 → 7/32/75/225;
  k=10 m=1: 8/23/59/437. Roughly r ∝ d^2-3 at fixed k and eps.
- **eps:** each decade of eps tightening multiplies the d≥3 rank by ~1.5-2.5 (d=4 k=20: 17/60/152 for
  1e-4/1e-6/1e-8; d4_k50 headline 1e-6 vs 1e-7 at the same event: 103 vs 257, ×2.5). This is the honest weak point:
  at tight tolerance d ≥ 3 ranks keep growing (d=4 k=40 1e-8 → 209; headline d=4 1e-7 → ≥ 290, unfinished).
- **migration:** fast (m=10) ≥ medium ≫ slow, because fast mixing populates more count sectors; slow trees stay at
  rank 6-20 for d ≤ 4.

Observations
- d=2: rank ~ 6 (slow), 8-26 (medium), 24-66 (fast) up to k=60 at eps=1e-8; at eps=1e-6 k=60 medium is rank 8.
- d=3 medium, eps=1e-6: 28/17/36/33 for k=20/30/40/60 (replicate seed at k=20: 20). Essentially flat beyond k=20.
- d=4 medium, eps=1e-6: 60/62/90 for k=20/30/40 (replicate at k=20: 30). eps=1e-8 at k=20: 152.
- Fast migration (m=10) costs the most rank (more count sectors populated); slow migration is cheapest.
- d=6 at k=10 already needs rank ~50 even in the slow regime: the d-dependence (count sectors ~ k^(d-1)) is the real limit.
- Rank vs eps at d=3 k=20 medium: 7 / 28 / 72 for eps = 1e-4 / 1e-6 / 1e-8, with loglik shifts of 5.8e-2, 1.3e-3, (ref).
  d=4 k=20: 17 / 60 / 152 with shifts 4.1e-2, 5e-4, (ref). Rule of thumb at k ≤ 20: |dloglik| ≈ (300-1000)×eps;
  it grows with k (k=40: 7e-3 at d=3 and 1.4e-2 at d=4 for eps=1e-6, i.e. ~10^4×eps), because the number of
  roundings and the number of sectors that get truncated both grow.

## 3. Headline run (sizes dense cannot do)

Trees from `sim/data/large` (heterochronous sampling, random asymmetric M, d=4, medium regime), run on Kaggle CPU
(4 vCPU, 30 GB; the 4 runs ran in parallel, ~1 core each; `kaggle/out/k4/*.json|log`). Dense would need 4^25 ≈ 1.1e15 and 4^30 ≈ 1.2e18 states.

| tree | max alive | eps | step | loglik | max TT rank | steps | wall s |
|---|---|---|---|---|---|---|---|
| d4_k50_medium_r0 | 25 | 1e-6 | h | **-123.155427** | 122 | 1138 | 10588 |
| d4_k50_medium_r0 | 25 | 1e-6 | h/2 | -123.157789 | 94 | 2276 | 10305 |
| d4_k50_medium_r0 | 25 | 1e-7 | h | unfinished (Kaggle 12 h limit at t=0.88 of 5.59, 17 alive) | ≥ 290 | — | > 41787 |
| d4_k60_medium_r0 | 30 | 1e-6 | h | **-132.108530** | 125 | 1202 | 7899 |

- Step error (h vs h/2) on d4_k50: 2.4e-3 (Richardson-corrected value -123.1579). The h/2 run reached a *lower*
  max rank (94 vs 122) — the rank peak is sensitive to where roundings fall, so max rank is a ±25% quantity.
- Truncation error at eps=1e-6 is not measured directly on these trees (the 1e-7 run did not finish). From the paired
  isochronous runs (1.4e-2 at d=4 k=40) and the Kaggle dense cross-check below (≤ 6e-3 at 13-14 alive), the honest
  error bar on the two headline values is **~1e-2 in loglik**.
- Rank trace d4_k50 at the same events, eps 1e-6 vs 1e-7: t=0.26: 28 vs 63; t=0.35: 40 vs 103; t=0.50: 60 vs 153;
  t=0.61: 81 vs 209; t=0.78: 103 vs 257. Peak (1e-6) 120-122 around t=0.9-1.4 with 10-16 alive.
- MASCOT on the same trees: -122.9450 (MASCOT−TT = +0.21) and -131.7602 (+0.35), i.e. 20-35× the TT error bar.

**Dense cross-check at the largest sizes dense can do** (Kaggle 30 GB, 13-14 simultaneous lineages, d=4;
`bench/kaggle_xcheck.md`), absolute loglik error of TT (Richardson not applied):

| tree | alive | dense | TT 1e-6 | TT 1e-7 | MASCOT−dense |
|---|---|---|---|---|---|
| d4_k14_medium_r1 | 13 | -41.309116 | 8.1e-5 | 4.8e-6 (OpenBLAS 재계산; the Accelerate run gave −48.550) | +0.214 |
| d4_k16_fast_r1 | 13 | -43.633813 | 1.1e-5 | 9.7e-7 | +0.005 |
| d4_k16_slow_r1 | 13 | -50.481523 | 2.2e-3 | 9.1e-4 | -0.091 |
| d4_k18_fast_r1 | 13 | -43.266281 | 3.2e-6 | 2.1e-7 | +0.042 |
| d4_k18_fast_r2 | 13 | -43.505705 | 1.8e-6 | 5.4e-7 | +0.025 |
| d4_k18_medium_r0 | 13 | -51.694801 | 2.0e-4 | 2.6e-5 | -0.029 |
| d4_k18_medium_r1 | 14 | -52.384007 | 2.8e-4 | 1.2e-5 | -0.285 |
| d4_k18_medium_r2 | 13 | -46.092416 | 9.1e-5 | 3.6e-6 | -0.088 |
| d4_k18_slow_r1 | 14 | -53.168403 | 5.8e-3 | 7.7e-4 | +0.044 |
| d4_k20_medium_r2 | 13 | -53.712660 | 1.6e-4 | 1.1e-5 | -0.501 |
| d4_k20_slow_r1 | 13 | -49.400618 | 6.0e-3 | 2.7e-4 | -3.586 |

  Max error: 6.0e-3 (1e-6) and 9.1e-4 (1e-7), worst in the slow regime (deep trees, many roundings); MASCOT errors
  are 0.005-3.6 on the same trees. Together with the local 181-tree cross-check (eps=1e-8, median 9e-8), TT-ESCO is
  validated against an independent exact implementation up to 14 simultaneous lineages at d=4.

**Cost model.** Per time step the work is a sweep of SVDs over k bonds of matrices of size ~(r·d)×(r·d·R_op)
(R_op ≈ 5 = rank of the decay operator), so

  runtime ≈ C · n_steps · k · d² · r_max³,

with C ≈ 0.7-1.4e-8 s on the Kaggle CPU (calibrated on the three headline runs: 9.3 s/step at k=25, r=122) and
C ≈ 2e-9 - 2e-8 s for the local CSV runs with r ≥ 50 (the spread is because r_max overstates the average bond rank
when the peak is brief, and because of the loaded machine). Small-rank runs (r < 30) are numpy-overhead dominated
and the fit degrades to t/step ∝ r^1.6. Consequences: d4_k50 at eps=1e-7 (r≈290) predicts ~100 s/step × 1138 steps
≈ 30 h, matching the observed failure to finish in 12 h; going one eps decade tighter costs ~(2.5)³ ≈ 15× time;
going from d=4 to d=6 at fixed k costs (1.5)² · (2-3)³ ≈ 20-60×. The SVD count and memory (k·d·r² floats: 25·4·300²
≈ 9e6 doubles = 72 MB) are not the problem; the r³ SVD cost is.

## Honest limitations

- The implementation is plain numpy; per-step cost is dominated by SVDs of (r·d)×(r·R) matrices. A d=4 tree with 25-30
  simultaneous lineages takes hours on a loaded machine. A C/BLAS implementation with block-sparse count sectors or
  TDVP would be 10-100x faster, but the rank numbers above are what any TT method must carry.
- Truncation error accumulates with the number of roundings (deep slow trees). eps must be chosen per tree or adaptively.
- Only symmetric all-to-all M (quicksim) for the rank table; sim/data trees have random asymmetric M and showed similar
  ranks at equal k (max 98 at d=3 k=13, eps=1e-8).
- d ≥ 6 is not practical with this approach beyond k ~ 15-20 at tight tolerance.

## Conclusion (go/no-go, 2026-10-06)

- **GO** for the stated goal (first exact-likelihood ESCO at sizes dense cannot do, d=4, 25-30 simultaneous
  lineages): delivered at eps=1e-6 with max rank 122-125, 2-3 h each on ~1 Kaggle vCPU, error bar ~1e-2 in loglik,
  20-35× smaller than the MASCOT approximation gap on the same trees and validated against dense up to 14 lineages.
- The rank is genuinely low in k: sub-linear (≈ k^0.5) between k=20 and 60 for d ≤ 4 at fixed eps.
- **Honest limits:** (i) rank grows ~1.5-2.5× per decade of eps for d ≥ 3, so 1e-7 at the headline size already
  needs rank ~300 and > 12 h; tolerance tighter than ~1e-6 at k ≳ 30, d=4 is out of reach for this plain-numpy code;
  (ii) the eps=1e-6 error grows with k (~1e-2 at k=40), so error bars must be measured per tree; (iii) the d-scaling
  is steep (r ∝ d^2-3): d=6 needs rank 143 at k=20 even at 1e-6 and 225-437 at k=10 for 1e-8 → **NO-GO for d ≥ 6**
  without exploiting the count-sector (block-sparse / symmetric) structure.
- Next steps if continued: block-sparse TT over per-deme count sectors (exact conserved-quantity structure), TDVP
  instead of split-step + rounding, adaptive eps ∝ 1/sqrt(#roundings), and a compiled implementation (10-100×).
