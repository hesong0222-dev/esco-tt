# Continuous migration axis: is "MASCOT underestimates for d ≥ 3 under slow migration" a d effect or a migration-scale effect?

Answers review/AUDIT_DESIGN.md M1.3 (line 39). Scripts: `bench/regime_metrics.py` (measures),
`bench/continuous_axis.py` (all tables below + figure; full printout `bench/continuous_axis_tables.txt`),
`bench/continuous_axis_matched.py`, `bench/continuous_axis_replicate.py`. Per-tree data: `bench/continuous_axis.csv`.
Figure: `paper/figures/fig_continuous_axis.{png,pdf}`. No new TT runs. Parts B and C are new **dense** runs on small trees
(k = 10–14, peak ≤ 10 lineages; 920 trees, about 6 min in total).

## Bottom line (for the paper)
1. In the original design the pairwise rate is fixed per regime, so d and scaled migration are almost collinear. For slow
   migration, Ne·m_out has medians of 0.10, 0.23 and 0.34 for d = 2, 3, 4, with non-overlapping ranges. The existing CSVs
   alone therefore cannot separate the two effects. A regression on them puts the effect on d, but it explains little
   (R² ≤ 0.07).
2. A matched-outflow design holds Ne·m_out equal across d (720 trees), and a re-run of the original generator with new
   seeds adds 200 trees. Neither shows a consistent sign. For every d and every Ne·m_out from 0.05 to 1.8, the median
   MASCOT − exact is close to zero (−0.20 to +0.21), and 32–60% of trees are negative. So the "d ≥ 3 underestimates"
   pattern in the 27-tree cells is mostly sampling noise in the median. On top of that sits a heavy negative tail
   (cell means −0.08 to −0.74, worst −6.5), which is heavier for larger d.
3. At a fixed scaled migration rate, the **magnitude** of the error depends on d. For Ne·m_out of 0.05–0.4 the median
   |MASCOT − exact| (k = 10) is 0.07–0.17 for d = 2, 0.32–0.43 for d = 3 and 0.50–0.64 for d = 4. Errors fall once
   Ne·m_out rises above about 1. Suggested wording: "at fixed scaled migration, the spread and the negative tail of
   MASCOT's error grow with d. Its sign is not systematic."

## Measures (`regime_metrics.py`)
- `Ne·m_out` = mean over demes a of Ne[a]·Σ_b M[a][b]. This is the scaled per-lineage outflow ("Nm").
- `outflow/coal rate` = mean m_out / mean (1/Ne).
- `E[mig]/coal` = total branch length × mean m_out / (k−1).
- `Fitch/coal` = Fitch minimum number of deme changes / (k−1). This one needs no parameters and can also be computed on
  real trees.
- Response: e = (MASCOT − exact)/(k−1), per coalescence and signed. MASCOT is the converged `approx/mascot.py`.

## A. Existing design (292 sim trees with an exact value: 239 small dense + 53 large TT)

| d | regime | n | Ne·m_out median (range) | outflow/coal | E[mig]/coal | Fitch/coal | median MASCOT−exact | median per coal |
|---|---|---|---|---|---|---|---|---|
| 2 | slow | 27 | 0.095 (0.07–0.14) | 0.10 | 0.13 | 0.11 | +0.085 | +0.0087 |
| 3 | slow | 35 | 0.231 (0.15–0.26) | 0.21 | 0.63 | 0.24 | −0.161 | −0.0110 |
| 4 | slow | 36 | 0.338 (0.27–0.40) | 0.30 | 1.10 | 0.36 | −0.312 | −0.0161 |
| 2 | medium | 27 | 0.993 (0.61–1.34) | 0.96 | 1.20 | 0.29 | +0.028 | +0.0022 |
| 3 | medium | 35 | 2.190 (1.50–2.54) | 2.02 | 2.94 | 0.47 | −0.083 | −0.0049 |
| 4 | medium | 35 | 3.472 (2.60–3.84) | 3.04 | 6.12 | 0.51 | −0.005 | −0.0007 |
| 2 | fast | 27 | 11.2 (5.7–15.3) | 10.7 | 10.9 | 0.37 | +0.002 | +0.0002 |
| 3 | fast | 35 | 21.7 (17.3–26.5) | 20.0 | 28.4 | 0.49 | +0.013 | +0.0004 |
| 4 | fast | 35 | 32.8 (27.1–40.1) | 29.1 | 57.9 | 0.57 | −0.007 | −0.0006 |

OLS of e with 95% CIs from a tree-level bootstrap (2000 resamples):

| subset | model | R² | d=3 | d=4 | log10 Nm |
|---|---|---|---|---|---|
| all (n=292) | d only | 0.017 | −0.019 [−0.033, −0.006] | −0.026 [−0.049, −0.006] | — |
| all | Nm (quadratic) only | 0.028 | — | — | +0.013 [+0.003, +0.026] |
| all | Nm + d | 0.058 | −0.025 [−0.044, −0.010] | −0.036 [−0.064, −0.013] | +0.018 [+0.006, +0.034] |
| Ne·m_out < 1 (n=112) | Nm + d | 0.071 | −0.073 [−0.128, −0.028] | −0.102 [−0.179, −0.036] | (quadratic, sign flips) |

The bins barely overlap. Only Ne·m_out ∈ [0.1, 0.2) holds trees of two values of d: 12 trees with d = 2 (median e
+0.008, 17% negative) and 9 with d = 3 (−0.030, 100% negative). Taken alone, this looks like a d effect at equal
migration scale. Parts B and C show that it does not replicate.

## B. Matched design: per-lineage outflow fixed across d (720 trees, dense exact)
M[a][b] = g·cbar/(d−1)·U(0.5,1.5), so Ne·m_out does not depend on d. Everything else follows sim/simulate.py:
Ne[a] = 1 + 0.5a, heterochronous sampling in U(0,1), uniform demes, k = 10. There are 40 reps per (g, d) and
g ∈ {0.05, …, 1.6}. Note that g = 0.1·(d−1) reproduces the original slow regime, and g = 0.2 equals d = 3 slow exactly.

| g | d | Ne·m_out | Fitch/coal | median MASCOT−exact | IQR | frac < 0 | median abs MASCOT | median abs LISCO | median abs SISCO |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 2 | 0.054 | 0.11 | +0.036 | −0.06 … +0.07 | 38% | 0.072 | 0.417 | 0.867 |
| 0.05 | 3 | 0.053 | 0.22 | +0.044 | −0.38 … +0.31 | 48% | 0.316 | 0.984 | 6.022 |
| 0.05 | 4 | 0.057 | 0.33 | −0.084 | −0.80 … +0.26 | 60% | 0.517 | 2.098 | 9.021 |
| 0.1 | 2 | 0.106 | 0.11 | +0.056 | −0.16 … +0.15 | 40% | 0.161 | 0.488 | 0.944 |
| 0.1 | 3 | 0.108 | 0.22 | −0.067 | −0.40 … +0.36 | 52% | 0.376 | 0.899 | 2.279 |
| 0.1 | 4 | 0.114 | 0.33 | +0.045 | −1.63 … +0.45 | 48% | 0.496 | 1.050 | 3.056 |
| 0.2 | 2 | 0.216 | 0.22 | +0.040 | −0.28 … +0.14 | 42% | 0.173 | 0.413 | 0.626 |
| 0.2 | 3 | 0.225 | 0.28 | +0.157 | −0.60 … +0.33 | 42% | 0.426 | 0.763 | 1.086 |
| 0.2 | 4 | 0.221 | 0.33 | +0.027 | −0.71 … +0.57 | 48% | 0.638 | 0.941 | 1.516 |
| 0.4 | 2 | 0.449 | 0.22 | +0.068 | −0.17 … +0.17 | 40% | 0.169 | 0.247 | 0.493 |
| 0.4 | 3 | 0.428 | 0.33 | +0.214 | −0.20 … +0.46 | 32% | 0.432 | 0.538 | 0.729 |
| 0.4 | 4 | 0.445 | 0.44 | −0.199 | −1.18 … +0.24 | 57% | 0.510 | 0.784 | 1.267 |
| 0.8 | 2 | 0.756 | 0.22 | +0.068 | −0.22 … +0.16 | 42% | 0.179 | 0.216 | 0.282 |
| 0.8 | 3 | 0.833 | 0.39 | +0.096 | −0.38 … +0.27 | 45% | 0.327 | 0.395 | 0.550 |
| 0.8 | 4 | 0.903 | 0.56 | −0.014 | −0.49 … +0.30 | 52% | 0.346 | 0.356 | 0.535 |
| 1.6 | 2 | 1.675 | 0.33 | −0.047 | −0.15 … +0.06 | 60% | 0.108 | 0.175 | 0.254 |
| 1.6 | 3 | 1.749 | 0.44 | +0.038 | −0.11 … +0.18 | 45% | 0.167 | 0.239 | 0.329 |
| 1.6 | 4 | 1.823 | 0.56 | −0.068 | −0.30 … +0.22 | 60% | 0.277 | 0.279 | 0.340 |

Median difference at equal g, with 95% bootstrap CIs:

| g | d=3 − d=2 | d=4 − d=2 |
|---|---|---|
| 0.05 | +0.008 [−0.185, +0.157] | −0.120 [−0.691, +0.032] |
| 0.1 | −0.123 [−0.260, +0.188] | −0.011 [−0.941, +0.225] |
| 0.2 | +0.117 [−0.388, +0.317] | −0.013 [−0.447, +0.369] |
| 0.4 | +0.146 [−0.015, +0.352] | −0.267 [−0.801, +0.082] |
| 0.8 | +0.027 [−0.357, +0.210] | −0.082 [−0.363, +0.157] |
| 1.6 | +0.085 [−0.033, +0.181] | −0.021 [−0.126, +0.223] |

OLS on the matched set (n = 720) with Nm + d: d=3 −0.005 [−0.015, +0.006], d=4 −0.042 [−0.062, −0.022] per coalescence,
R² 0.049. The d=4 coefficient is a statement about the **mean**, so it is driven by the negative tail; the d = 4 medians
show no consistent sign. Pooled with A (n = 1012): d=3 −0.011 [−0.019, −0.002], d=4 −0.040 [−0.057, −0.025].

## C. Replication of the original slow cells with new seeds (original generator, 40 per cell, dense exact)

| d | k | n | Ne·m_out | median MASCOT−exact [95% CI] | mean | frac < 0 | median abs | min |
|---|---|---|---|---|---|---|---|---|
| 2 | 10 | 40 | 0.105 | +0.043 [−0.056, +0.092] | −0.247 | 42% | 0.139 | −3.76 |
| 2 | 14 | 40 | 0.101 | +0.043 [−0.002, +0.105] | −0.080 | 38% | 0.125 | −3.25 |
| 3 | 10 | 40 | 0.216 | −0.143 [−0.719, +0.163] | −0.739 | 57% | 0.365 | −6.52 |
| 3 | 14 | 40 | 0.213 | −0.121 [−0.466, +0.353] | −0.191 | 52% | 0.485 | −2.54 |
| 4 | 10 | 40 | 0.334 | −0.128 [−0.925, +0.336] | −0.483 | 60% | 0.770 | −3.24 |

In the original 27-tree cells, 15% / 70% / 62% of trees were negative for d = 2 / 3 / 4. With new seeds the figures are
38–42% / 52–57% / 60%. The d = 3 and d = 4 medians are still negative, but every CI includes 0. Even d = 2 has a negative
tail (min −3.8).

## Interpretation
- **Sign.** No d effect and no migration-scale effect is robust. The median sits near 0 everywhere. "Underestimates"
  describes a skewed distribution: most trees fall within ±(a few tenths), while a minority with very negative errors
  (MASCOT gives far too little likelihood) pulls the mean down. That tail grows with d at fixed Ne·m_out.
- **Magnitude.** At fixed Ne·m_out, d matters: median |error| at d = 4 is 2–7× that at d = 2 (B). Migration scale matters
  too: errors shrink once Ne·m_out > 1 (A, fast regime: median |e| < 1e-3 per coalescence). Over 0.05–0.4 the median
  |error| changes little, and if anything rises slightly with Ne·m_out (d = 2: 0.07 → 0.17; d = 3: 0.32 → 0.43;
  d = 4: 0.50–0.64, no trend). So "slow is worse" holds for slow versus fast. Within the slow range it is not true.
- **The other closures** (B, last two columns) show the same pattern more strongly. SISCO's error explodes as migration
  slows (median |error| 9.0 at d = 4, Ne·m_out 0.05) while MASCOT stays at 0.5. See bench/sisco.md.

## Caveats
- Ne heterogeneity also grows with d in this design (Ne = 1 + 0.5a, so the maximum/minimum Ne ratio is 1.5, 2, 2.5).
  "d" here therefore means "d together with this Ne pattern".
- B and C use small trees (k = 10–14), because only then can dense be run on hundreds of trees. The large-tree TT values (A)
  cannot be re-run under the matched design without new TT runs.
- OLS uses the per-coalescence error and has low R² throughout (≤ 0.07). Most of the variation lies between trees
  (topology and sampling), not in d or Nm.
