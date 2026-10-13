# TT-ESCO robustness (REVIEW.md required fixes 2, 3, 4) — 2026-10-06

All runs: OpenBLAS `.venv` (Mac, 1 thread per process, machine shared at load 40–370, so run times are only
comparable within a row) or Kaggle kaggle (Linux). Code: `tt/esco_tt.py` (new flags `mass_fix`, `eps_total`, new
output `n_round`; defaults unchanged, so every earlier number is reproduced bit-for-bit — checked on
d3_k12_slow_r1, d2_k20_medium_r0, d4_k08_slow_r0, and the Kaggle 13–14-lineage values in `kaggle/xcheck.csv`).
Runner `bench/robustness.py`, tables `bench/robustness_analyze.py`, raw CSVs in `bench/robustness/`.

**Status (2026-10-08): complete.** The Kaggle runs R1–R3, KB, KC1–KC3 finished (dataset v2 = current `tt/`;
the v1 kernels of 10-07 had all failed at start). The only missing cell is d3_k60_medium_r1 at eps 1e-8 (KC2 was
cancelled after 17/18 jobs), so that tree's reference is eps 1e-7. Section 4 has the new large-tree results.

## 1. Truncation vs time-step error

Notation: N_round = number of TT roundings in one likelihood evaluation (one per decay-operator application, one
per coalescence). It is now returned by `loglik_detail` and can be computed beforehand by
`esco_tt.count_roundings(tree)`. Default step rule (s_max = 0.25, 4th-order Suzuki–Yoshida) gives N_round ≈ 1e3–2e4
on the small trees and 1.2e4–3.9e4 on the large ones (d4 k50/k60 medium: 2.30e4 / 2.38e4; d4 k60 slow 3.9e4).

**Set A** — 19 `sim/data` trees with dense reference (d = 2, 3, 4; 2–3 per d × regime; up to 15 lineages alive):
eps ∈ {1e-5, 1e-6, 1e-7, 1e-8} × {h, h/2} × mass_fix {off, on}, plus eps = 1e-10 at h and h/2 (pure step error).
CSV `bench/robustness/setA_small_dense.csv`, split in `summary_truncation_split.csv`.

**Set B** — the five slow/fast Kaggle trees with 13–14 lineages alive and a dense reference (`kaggle/dense_big.csv`):
eps ∈ {1e-5, 1e-6, 1e-7, 1e-8}, h/2 at 1e-6, mass_fix on/off, eps_total = 1e-3. CSV `setB_big_dense.csv`.

**Set C** — large trees without dense reference (reference = eps 1e-8 on the same tree): d3_k60_slow_r0 (28 alive)
and d4_k30_slow_r1 (19 alive). CSV `setC_large_local.csv`, `summary_large_ladder.csv`.

### Step error (eps = 1e-10, default h, minus dense; Set A)

| regime | median | max |
|---|---|---|
| slow | 3.3e-7 | 1.1e-6 |
| medium | 5.0e-6 | 3.9e-5 |
| fast | 4.1e-6 | 1.5e-5 |

h → h/2 reduces it by the expected ~16×. At every eps ≥ 1e-7 the step error is negligible against truncation,
except medium/fast trees at eps ≤ 1e-7 where both are ~1e-5.

### Truncation error ∝ eps · N_round

Truncation part = logL(eps, h) − logL(1e-10, h) (Set A) or logL(eps) − dense (Set B; step error there ≤ 1e-7).
c := truncation part / (eps · N_round).

| set | regime | (tree, eps) pairs | median abs c | max abs c | fraction negative |
|---|---|---|---|---|---|
| A | slow | 28 | 0.074 | 0.81 | 0.71 |
| A | medium | 24 | 0.042 | 0.40 (1.29 at eps 1e-8, step-dominated) | 0.63 |
| A | fast | 24 | 0.003 | 0.57 (eps 1e-8, step-dominated) | 0.25 |
| B (d4, 13–14 alive) | slow | 12 | 0.37 | 0.53 | 0.92 |
| B (d4, 13–14 alive) | fast | 8 | 0.0004 | 0.001 | 0.88 |
| C (large slow) | slow | 4 (eps 1e-5…1e-7) | 0.09 | 0.10 | 0.75 |

Max |truncation error| per eps over Set A+B: 5.8e-2 (1e-5), 6.0e-3 (1e-6), 9.1e-4 (1e-7), 8.2e-5 (1e-8). The
error falls roughly 10× per decade of eps (less regularly for small trees, where the rank changes in jumps).

**Step ladder at fixed eps.** Halving h doubles N_round; the truncation part grew by a median factor 1.76
(IQR 1.33–2.06, Set A, eps 1e-6; Set B: 0.8–2.0; Set C: 1.7 and 0.6). So h vs h/2 differences at loose eps measure
truncation, not step error — the Richardson correction is not meaningful there (REVIEW B.2 confirmed). This also
explains why h/2 runs on large trees have *lower* rank (more, smaller truncations).

**Empirical error model (usable in the paper).**

  logL_TT − logL_exact ≈ c · eps · N_round + O(h⁴),  |c| ≤ 0.81 in all 96 (tree, eps) pairs with a
  dense reference (truncation part only; 76 Set A + 20 Set B), median |c| 0.003 (fast) – 0.07 (slow small) – 0.37 (slow d=4, 13–14 lineages).

c has a per-tree sign that stays fixed along the trajectory (it is a systematic drift, not a random walk: error ∝
N_round, not √N_round); it is negative in 84% of pairs. Slow migration has the largest |c|. Applied to the headline
trees (eps 1e-6, N_round ≈ 2.3e4): envelope |error| ≤ 0.81 × 0.023 ≈ 2e-2; typical slow/medium c 0.1–0.4 gives
2e-3–1e-2, consistent with the k50 h/2 difference (2.4e-3 ≈ c · eps · ΔN with c ≈ 0.1) and with the reviewer's
3–5e-3 extrapolation. Not a proven bound: it is a fit over 24 + 2 trees.

## 2. Rounding fixes (behind flags; defaults = old algorithm)

### (a) `eps_total` — per-rounding tolerance eps = eps_total / N_round

Calibration (Set B, eps_total = 1e-3, i.e. eps ≈ 6e-8 – 1.2e-7 per rounding):

| tree | |error| / eps_total (plain) | (mass_fix) | rank | run time vs eps 1e-7 run |
|---|---|---|---|---|
| d4_k16_slow_r1 | 0.46 | 0.38 | 52 | 1.2× |
| d4_k18_slow_r1 | 0.33 | 0.25 | 61 | 1.3× |
| d4_k20_slow_r1 | 0.089 | 0.031 | 32 | 1.6× |
| d4_k16_fast_r1 | 0.0011 | 0.0002 | 38 | 0.7× |
| d4_k18_fast_r1 | 0.0002 | 0.00004 | 47 | 0.8× |

All five land below target, consistent with |c| ≤ 0.81. eps_total does not change cost-accuracy (it is the same
algorithm with a tree-dependent eps); it makes the tolerance mean "log-likelihood error budget" so a user can ask for
1e-3 and get ≤ ~1e-3 on the trees tested. Large-tree eps_total = 1e-2 runs: d3_k60_slow and d4_k30_slow done
(Set C, |diff vs eps 1e-8| 5.4e-4 and 1.1e-4, i.e. 0.05 and 0.01 of target); medium/fast large pending.

### (b) `mass_fix` — rescale to the exact mass after every rounding

Exact mass before rounding is cheap: Σ(P∘W) = ⟨P, W⟩ for a decay step (one TT contraction), 1 after a
coalescence. Cost +0–25% (Set A/B run times). Effect on |error| (mass_fix / plain, same eps, eps ≥ 1e-7):

| set | regime | n | median ratio | fraction improved | mean signed error plain → mass_fix |
|---|---|---|---|---|---|
| A (small) | slow | 21 | 1.31 | 0.43 | −1.7e-3 → −1.1e-3 |
| A (small) | medium | 18 | 0.79 | 0.67 | −4.4e-4 → −2.8e-4 |
| A (small) | fast | 18 | 1.11 | 0.28 | +2.8e-7 → +6.5e-6 |
| B (d4, 13–14) | slow | 9 | 0.71 | 0.89 | −1.2e-2 → −8.2e-3 |
| B (d4, 13–14) | fast | 6 | 0.11 | 1.00 | −1.6e-5 → +5.5e-7 |
| C (large slow), eps 1e-6 | slow | 2 | 0.17, 0.03 | 2/2 | −1.7e-3, −2.3e-3 → +2.9e-4, −7e-5 |
| C (large slow), eps 1e-5 | slow | 2 | 1.7, 2.7 | 0/2 | |

Interpretation: Frobenius truncation always shrinks the total mass (one-signed), and mass_fix removes exactly that
part; the remaining error is a change of the *shape* of P, whose effect on later coalescence factors has a tree-dependent sign.
mass_fix therefore reduces the mean negative bias everywhere (by ~35%) and helps a lot on the d = 4 trees with
13–30 lineages at eps 1e-6–1e-7 (median 0.1–0.7, up to 30× on large slow), but makes 57% of small slow trees and
72% of small fast trees worse (mostly at errors < 1e-4) and is worse at eps 1e-5. **It does not remove the drift**;
it is kept as an option (`mass_fix=True`), not as the default.

### Rejected alternatives (pilot only, `pilot_step.csv`, `pilot_order2.csv`)
- Larger steps (s_max 0.5/1.0): fewer roundings → less truncation in slow trees, but step error 1e-4–9e-3 in
  medium/fast. No regime-independent gain.
- 2nd-order Strang (1 rounding per step instead of 5): step error 1e-4–1e-3 at s_max 0.25; no better at equal cost.

## 3. Rank study v2 (REVIEW fix #4)

Trees: `bench/robustness/rank_trees/` (generator `bench/robustness_make_rank_trees.py`) — `sim/simulate.generate`,
i.e. heterochronous sampling (half at t = 0, rest U(0,1)), asymmetric per-pair M = factor · c̄ · U(0.5, 1.5),
Ne = 1 + 0.5a, regimes slow/medium/fast = factor 0.1/1/10; 3 seeds per (d, k, regime); seed =
20261006 + 100000 d + 1000 k + 10 regime + rep (disjoint from sim/data). The old study (`rank_study.csv`) used
isochronous `quicksim` trees with symmetric M, Ne = 1 and mostly one seed. Plain algorithm (mass_fix off).
Max lineages alive ≈ 0.5–0.6 k (k = 40 → 20–26). Table: max TT rank, mean ± sd over 3 seeds
(`summary_rank_v2.csv`).

eps = 1e-6

| d | regime | k=10 | k=20 | k=30 | k=40 |
|---|---|---|---|---|---|
| 2 | slow | 4 ± 0 | 4 ± 0 | 4 ± 1 | 4 ± 1 |
| 2 | medium | 6 ± 1 | 8 ± 2 | 11 ± 4 | 13 ± 2 |
| 2 | fast | 7 ± 1 | 11 ± 2 | 11 ± 1 | 16 ± 2 |
| 3 | slow | 8 ± 1 | 9 ± 1 | 9 ± 2 | 11 ± 5 |
| 3 | medium | 18 ± 2 | 34 ± 6 | 49 ± 12 | 61 ± 18 |
| 3 | fast | 11 ± 2 | 22 ± 1 | 26 ± 5 | 33 ± 2 |
| 4 | slow | 12 ± 3 | 27 ± 7 | 19 ± 4 | 25 ± 8 |
| 4 | medium | 38 ± 16 | 79 ± 7 | 114 ± 19 | 126 ± 33 |
| 4 | fast | 17 ± 1 | 29 ± 2 | 36 ± 3 | 42 ± 4 |

eps = 1e-7 (k ≤ 20)

| d | regime | k=10 | k=20 |
|---|---|---|---|
| 2 | slow / medium / fast | 5 ± 1 / 8 ± 1 / 7 ± 0 | 5 ± 1 / 10 ± 3 / 14 ± 3 |
| 3 | slow / medium / fast | 9 ± 0 / 26 ± 5 / 17 ± 3 | 12 ± 3 / 53 ± 10 / 31 ± 5 |
| 4 | slow / medium / fast | 17 ± 7 / 66 ± 29 / 31 ± 3 | 46 ± 12 / 154 ± 23 / 45 ± 3 |

Paired accuracy (|logL(1e-6) − logL(1e-7)|, max over 3 seeds): ≤ 1.7e-4 for d = 2 and d = 3 medium/fast at k ≤ 20;
2.4e-3 (d3 slow k10), 1.5e-3 (d3 slow k20), 2.4e-4 (d4 slow k10), **8.9e-3 (d4 slow k20)**; d4 k20 medium 4.1e-4,
fast 2.6e-5 — slow migration is again
where eps 1e-6 is least accurate even though its rank is smallest.

Findings so far: (i) the regime ordering medium > fast > slow holds for d = 3, 4 at every k; (ii) for slow migration
the rank does not grow with k (d = 4: 12–27 from k = 10 to 40); (iii) seed-to-seed sd is 10–40% of the mean (up to
±16 on 38 for d4 medium k10), so single-seed numbers are not reliable; (iv) d3 medium grows ~linearly in k
(18 → 61), not yet distinguishable from mild polynomial growth with 4 k-values; d4 medium grows 38 → 79 → 114 → 126
and flattens at k = 40 (sd 33, so the flattening is within noise), d4 fast 17 → 42; (v) eps 1e-6 → 1e-7 multiplies
rank by 1.2–2.0 (d4 medium k20: 79 → 154). Medium migration is the expensive regime for d = 4: k = 40 takes a median
2.8 h per evaluation on Kaggle (4 cores shared by 4 jobs) vs 0.1 h slow, 0.5 h fast.

## 4. Large trees and the 13–14-lineage medium trees (Kaggle KB, KC1–KC3, 2026-10-08)

**Set B medium (KB, dense reference):** c for d4_k14/k18/k20 medium at eps 1e-5…1e-7: median |c| 0.015, max 0.030,
all negative (n = 6). mass_fix reduces |error| on all six (median ratio 0.21). eps_total = 1e-2 gives |error| / target
0.009, 0.028, 0.014 (rank 72–85, eps per rounding 0.8–1.2e-6).

**Large ladders (no dense reference; reference = tightest eps on the same tree, plain h):**

| tree | N_round | err at 1e-5 | 1e-6 | 1e-7 | h/2 at 1e-6 | c at 1e-6 | rank at 1e-6 |
|---|---|---|---|---|---|---|---|
| d3_k50_fast_r0 | 12 597 | −1.0e-3 | −2.0e-4 | −2.7e-5 | −3.0e-4 | 0.016 | 31 |
| d3_k60_slow_r0 | 18 742 | −6.9e-3 | −1.7e-3 | −2.0e-4 | −2.8e-3 | 0.089 | 9 |
| d3_k60_medium_r1 (ref 1e-7) | 18 198 | −2.2e-2 | −1.9e-3 | — | −5.7e-3 | 0.10 | 49 |
| d4_k30_fast_r0 | 13 440 | −4.0e-4 | −6.6e-5 | −6.6e-6 | −1.1e-4 | 0.005 | 39 |
| d4_k30_slow_r1 | 22 422 | +1.4e-2 | −2.3e-3 | −2.2e-4 | −1.3e-3 | 0.10 | 24 |
| d4_k30_medium_r1 (ref 1e-6) | 11 626 | −2.0e-3 | — | (mass_fix +5.7e-4) | — | — | 102 |
| d4_k60_slow_r0 | 38 795 | −6.4e-2 | −9.5e-3 | −1.8e-3 | −2.0e-2 | 0.25 | 16 |
| d4_k60_medium_r0 (ref 1e-5 plain) | 23 818 | 0 | (mass_fix +2.5e-2) | — | — | — | 125 |

- **The error model holds on every large tree:** |c| ≤ 0.25 at eps 1e-6 (envelope 0.81), and the error falls about
  10× per decade of eps. At the headline eps 1e-6 the worst large tree is d4_k60_slow (−9.5e-3). The large-sweep
  paired estimate |h − h/2| ≤ 2.6e-2 is consistent with this: h/2 doubles N_round, and on d4_k60_slow it gives −2.0e-2.
- d4_k60_medium ("k60 at two eps", REVIEW): plain 1e-5 and mass_fix 1e-6 differ by 2.5e-2. With c ≈ 0.1 the
  plain 1e-5 error is about 2e-2, so the eps 1e-6 headline value is accurate to roughly 2e-3. That is an estimate,
  not a reference: a 1e-7 run (rank > 200) is out of reach in 12 h.
- mass_fix on large trees: better at eps 1e-6–1e-7 (d4_k60_slow 1e-6: −9.5e-3 → −2.9e-3; d4_k30_slow −2.3e-3 → −7e-5),
  worse at 1e-5 on slow trees (sign flips, +1.2e-2 / +3.9e-2). Same conclusion as §2(b): keep it as an option.
- d4_k30_slow_r1 at eps 1e-5 is the one positive error (+1.4e-2). Its sign differs from the 1e-6/1e-7 errors on the
  same tree, so at loose eps the per-tree sign rule of §1 can break. Do not use eps 1e-5 for inference.
