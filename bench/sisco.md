# SISCO and LISCO vs exact, next to MASCOT, on every tree with an exact value

Answers review/AUDIT_DESIGN.md M2 ("other approximations", line 70; action 5, line 134).
Code: `approx/independence.py`. Checks: `bench/sisco_validate.py` (output in `bench/sisco_validate.txt`).
Evaluation: `bench/sisco_eval.py` → `bench/sisco_eval.csv`; H3N2 posterior states: `bench/h3n2_posterior_eval.csv`.
Tables: `bench/sisco_summarize.py`.

## Bottom line (for the paper)
1. Of the three closures of Müller et al. (2017), the order MASCOT < LISCO < SISCO in error holds on the 292 simulated
   trees with exact values. MASCOT is closest on 78% of trees, and the median |error| is 0.097 / 0.168 / 0.279. In the
   slow regime the 90th percentile is 1.8 / 2.5 / 4.0 log units. So the MASCOT errors mapped in the paper are the
   smallest available from this family of approximations.
2. SISCO ignores the conditioning on no coalescence. It fails worst exactly where real data sit (slow migration, d ≥ 3),
   with median errors of −1.2 to −1.7 and extremes up to −11.7. On the 361 H3N2 posterior states it is off by −1.5
   (median, worst −75), and on the H3N2 fit points by −4.4 (median). On AIV fit points it is off by up to −122.
3. At the AIV host-order MASCOT optima, MASCOT overestimates by +0.2 to +8.0. LISCO is closer there on 7 of 8 points
   (median +0.40). These points were chosen as MASCOT optima, so MASCOT's error is selected to be large and positive
   there. The comparison does not show LISCO to be better in general, but it does show that the ranking of closures can
   flip in real-data regimes.

## Definitions (verified against the paper and the upstream code)
Müller, Rasmussen & Stadler (2017) MBE 34(11):2970; code github.com/nicfel/The-Structured-Coalescent @81b647f
(`esco/src/beast/structuredCoalescent/...`, read on 2026-10-09; not copied into the repo).

All three keep one normalised d-vector p_i per lineage. They share three rules:
- interval: d log P/dt = −Σ_a λ_a Σ_{i<j} p_i(a)p_j(a);
- coalescence: P ×= Σ_a λ_a p_i(a)p_j(a), and the parent gets ∝ λ_a p_i p_j;
- sampling: the new lineage starts as a one-hot vector.

The lineage ODE is

  dp_i(a)/dt = Σ_b (m_ba p_i(b) − m_ab p_i(a)) + κ·[ p_i(a) Σ_b λ_b p_i(b) o_i(b) − λ_a p_i(a) o_i(a) ],
  with o_i(a) = Σ_{k≠i} p_k(a).

- **κ = 1, MASCO / MASCOT** (paper eq. 3, after normalising by P(T)). `approx/mascot.py`.
- **κ = 0, SISCO** (paper eq. 4: "changes in the probabilities of lineages being in a certain state are only dependent on
  the migration rates"). The upstream `ApproximateStructuredCoalescent.java` propagates p with the migration matrix only
  and uses the same interval and event terms.
- **κ = 1/2, LISCO** (`IndependentStructuredCoalescent.java` + `math/independent_ode_integrator.java`; the paper text
  covers ESCO, MASCO and SISCO). The code integrates *unnormalised* q_i:
  dq_i(a) = mig(q_i) − (λ_a/2) q_i(a) o_i(a), where o is built from the normalised p. At each coalescence it adds
  Σ_i log Σ_a q_i(a) to logP and renormalises. Writing q_i = w_i p_i gives the κ = 1/2 ODE for p_i. The masses also give
  Σ_i d log w_i/dt = −Σ_a λ_a Σ_{i<j} p_i p_j, which is the common interval term. In words, each pair's no-coalescence
  rate is split half to each lineage, instead of being charged to each lineage in full as in MASCOT.

## Validation (`bench/sisco_validate.txt`)
| check | result |
|---|---|
| κ = 1 vs `approx/mascot.py`, 243 small trees | max abs diff 6.6e-11 |
| LISCO κ form vs literal Java q-form (unnormalised, masses logged at coalescence; DOP853 instead of RK4), 61 trees | max 1.0e-9 |
| d = 1 vs Kingman closed form, k = 2–40, all three | max 1.1e-14 |
| fast-migration limit: M × s on 3 d=3 k=8 trees vs dense | all three → exact as 1/s (error at s = 1000: SISCO ≤ 3.6e-4, LISCO ≤ 2.7e-4, MASCOT ≤ 1.9e-4) |
| slow scan (M × 0.1 … 0.001, not a convergent limit) | SISCO > LISCO > MASCOT throughout (e.g. +4.6 / +3.3 / +1.7 at × 0.001) |

**No published SISCO or LISCO value was reproduced.** The repository contains only BEAST logs of MASCOT (H3N2 tutorial)
and MASCO (`data/raw2/aiv_AIV_1masco.log`). The upstream repository has LISCO and SISCO AIV logs (`AIV/structcoal/`).
Reproducing one would mean downloading them together with their posterior trees, which needs the user's approval (the
next step if a reviewer asks). The checks above rest on the equations and the code logic, not on a logged value.

**BASTA was not implemented.** No BEAST2-BASTA logged density exists in the repository, so a re-implementation could not be
checked against the reference program. BASTA and SISCO are not the same closure, and BASTA is not covered here.

## Results: approx − exact (log units), by set, d and regime

Exact references:
- small: dense (239 of 243 trees; 4 d=4 k=20 trees with peak 15–16 have no dense value);
- large: TT eps 1e-6 (53 trees, paired h/2 error ≤ 2.6e-2, below every difference reported here except d3_k50_slow_r1
  for MASCOT, see large_sweep.md);
- H3N2 fit points: dense, recomputed (25 points = ML optima of MASCOT and exact on the MCC tree + 4 posterior trees,
  bench/realdata.csv);
- H3N2 posterior: the logged states after 10% burn-in (dense; TT 1e-6 for peak ≥ 16);
- AIV: the host-order (3 demes) masco-MCC points of bench/realdata2.csv, TT exact (error ≤ 0.04, realdata2.md).

| set | d | regime | n | method | median (approx − exact) | range | median abs | abs>0.1 | max abs |
|---|---|---|---|---|---|---|---|---|---|
| small | 2 | slow | 27 | MASCOT | +0.085 | -0.19 … +0.30 | 0.094 | 41% | 0.30 |
| small | 2 | slow | 27 | LISCO | +0.259 | -1.48 … +0.50 | 0.298 | 74% | 1.48 |
| small | 2 | slow | 27 | SISCO | +0.279 | -3.80 … +0.86 | 0.455 | 96% | 3.80 |
| small | 2 | medium | 27 | MASCOT | +0.028 | -0.70 … +0.53 | 0.168 | 70% | 0.70 |
| small | 2 | medium | 27 | LISCO | +0.004 | -0.85 … +0.51 | 0.244 | 74% | 0.85 |
| small | 2 | medium | 27 | SISCO | -0.065 | -1.02 … +0.66 | 0.268 | 81% | 1.02 |
| small | 2 | fast | 27 | MASCOT | +0.002 | -0.07 … +0.04 | 0.016 | 0% | 0.07 |
| small | 2 | fast | 27 | LISCO | +0.004 | -0.15 … +0.06 | 0.023 | 11% | 0.15 |
| small | 2 | fast | 27 | SISCO | +0.007 | -0.24 … +0.11 | 0.034 | 19% | 0.24 |
| small | 3 | slow | 27 | MASCOT | -0.161 | -2.71 … +0.55 | 0.298 | 78% | 2.71 |
| small | 3 | slow | 27 | LISCO | -0.824 | -4.48 … +0.74 | 0.824 | 89% | 4.48 |
| small | 3 | slow | 27 | SISCO | -1.672 | -6.61 … +0.73 | 1.672 | 89% | 6.61 |
| small | 3 | medium | 27 | MASCOT | -0.083 | -0.69 … +0.40 | 0.158 | 67% | 0.69 |
| small | 3 | medium | 27 | LISCO | -0.118 | -0.76 … +0.70 | 0.211 | 78% | 0.76 |
| small | 3 | medium | 27 | SISCO | -0.234 | -1.37 … +1.14 | 0.326 | 74% | 1.37 |
| small | 3 | fast | 27 | MASCOT | -0.004 | -0.12 … +0.06 | 0.029 | 4% | 0.12 |
| small | 3 | fast | 27 | LISCO | -0.005 | -0.15 … +0.11 | 0.037 | 11% | 0.15 |
| small | 3 | fast | 27 | SISCO | -0.003 | -0.23 … +0.20 | 0.044 | 19% | 0.23 |
| small | 4 | slow | 26 | MASCOT | -0.284 | -6.46 … +0.80 | 0.611 | 88% | 6.46 |
| small | 4 | slow | 26 | LISCO | -0.278 | -7.87 … +1.10 | 0.842 | 88% | 7.87 |
| small | 4 | slow | 26 | SISCO | -1.135 | -11.69 … +1.91 | 1.316 | 92% | 11.69 |
| small | 4 | medium | 26 | MASCOT | +0.002 | -0.68 … +0.34 | 0.179 | 69% | 0.68 |
| small | 4 | medium | 26 | LISCO | +0.041 | -0.78 … +0.46 | 0.285 | 69% | 0.78 |
| small | 4 | medium | 26 | SISCO | -0.020 | -1.05 … +0.61 | 0.404 | 77% | 1.05 |
| small | 4 | fast | 25 | MASCOT | -0.010 | -0.05 … +0.05 | 0.020 | 0% | 0.05 |
| small | 4 | fast | 25 | LISCO | -0.012 | -0.11 … +0.09 | 0.030 | 4% | 0.11 |
| small | 4 | fast | 25 | SISCO | -0.013 | -0.18 … +0.14 | 0.036 | 16% | 0.18 |
| large | 3 | slow | 8 | MASCOT | -0.123 | -0.76 … +0.63 | 0.406 | 88% | 0.76 |
| large | 3 | slow | 8 | LISCO | -0.386 | -1.96 … +0.40 | 0.491 | 100% | 1.96 |
| large | 3 | slow | 8 | SISCO | -1.155 | -2.70 … +0.11 | 1.155 | 88% | 2.70 |
| large | 3 | medium | 8 | MASCOT | -0.064 | -0.61 … +0.56 | 0.292 | 100% | 0.61 |
| large | 3 | medium | 8 | LISCO | -0.079 | -0.83 … +1.17 | 0.643 | 75% | 1.17 |
| large | 3 | medium | 8 | SISCO | -0.344 | -2.34 … +2.00 | 0.978 | 88% | 2.34 |
| large | 3 | fast | 8 | MASCOT | +0.037 | -0.09 … +0.09 | 0.058 | 0% | 0.09 |
| large | 3 | fast | 8 | LISCO | +0.049 | -0.30 … +0.16 | 0.154 | 75% | 0.30 |
| large | 3 | fast | 8 | SISCO | +0.053 | -0.58 … +0.27 | 0.241 | 75% | 0.58 |
| large | 4 | slow | 10 | MASCOT | -0.336 | -1.20 … +0.58 | 0.405 | 90% | 1.20 |
| large | 4 | slow | 10 | LISCO | -0.699 | -1.52 … +0.83 | 0.917 | 90% | 1.52 |
| large | 4 | slow | 10 | SISCO | -1.349 | -3.48 … +0.77 | 1.349 | 100% | 3.48 |
| large | 4 | medium | 9 | MASCOT | -0.027 | -0.63 … +0.95 | 0.296 | 89% | 0.95 |
| large | 4 | medium | 9 | LISCO | -0.153 | -0.85 … +1.44 | 0.452 | 89% | 1.44 |
| large | 4 | medium | 9 | SISCO | -0.546 | -2.30 … +2.12 | 0.731 | 100% | 2.30 |
| large | 4 | fast | 10 | MASCOT | +0.030 | -0.08 … +0.12 | 0.065 | 20% | 0.12 |
| large | 4 | fast | 10 | LISCO | +0.026 | -0.31 … +0.48 | 0.198 | 70% | 0.48 |
| large | 4 | fast | 10 | SISCO | +0.019 | -0.64 … +0.90 | 0.343 | 70% | 0.90 |
| H3N2 fit points | 3 | real | 25 | MASCOT | +0.254 | -0.20 … +1.67 | 0.254 | 72% | 1.67 |
| H3N2 fit points | 3 | real | 25 | LISCO | -0.617 | -3.61 … +0.44 | 0.617 | 96% | 3.61 |
| H3N2 fit points | 3 | real | 25 | SISCO | -4.371 | -16.96 … +0.49 | 4.371 | 100% | 16.96 |
| H3N2 posterior states (post burn-in) | 3 | real | 361 | MASCOT | +0.350 | -0.40 … +2.92 | 0.350 | 89% | 2.92 |
| H3N2 posterior states (post burn-in) | 3 | real | 361 | LISCO | -0.036 | -6.63 … +2.15 | 0.610 | 93% | 6.63 |
| H3N2 posterior states (post burn-in) | 3 | real | 361 | SISCO | -1.462 | -74.75 … +2.43 | 1.758 | 96% | 74.75 |
| AIV host-order fit points | 3 | real | 8 | MASCOT | +6.545 | +0.24 … +7.99 | 6.545 | 100% | 7.99 |
| AIV host-order fit points | 3 | real | 8 | LISCO | +0.395 | -1.48 … +3.14 | 1.253 | 100% | 3.14 |
| AIV host-order fit points | 3 | real | 8 | SISCO | -20.646 | -122.21 … +2.29 | 20.646 | 100% | 122.21 |

Closest closure per tree or point, with the ratio of median |error| to MASCOT's:

| set | d | regime | n | MASCOT best | LISCO best | SISCO best | median abs ratio LISCO/MASCOT | SISCO/MASCOT |
|---|---|---|---|---|---|---|---|---|
| small | 2 | slow | 27 | 21 | 5 | 1 | 3.04 | 5.65 |
| small | 2 | medium | 27 | 18 | 3 | 6 | 1.22 | 1.60 |
| small | 2 | fast | 27 | 21 | 4 | 2 | 1.25 | 2.07 |
| small | 3 | slow | 27 | 22 | 2 | 3 | 2.03 | 2.52 |
| small | 3 | medium | 27 | 22 | 1 | 4 | 1.26 | 1.77 |
| small | 3 | fast | 27 | 24 | 0 | 3 | 1.31 | 1.68 |
| small | 4 | slow | 26 | 16 | 4 | 6 | 1.17 | 1.82 |
| small | 4 | medium | 26 | 21 | 2 | 3 | 1.35 | 1.85 |
| small | 4 | fast | 25 | 20 | 1 | 4 | 1.47 | 1.96 |
| large | 3 | slow | 8 | 5 | 1 | 2 | 1.85 | 3.45 |
| large | 3 | medium | 8 | 5 | 2 | 1 | 1.30 | 2.46 |
| large | 3 | fast | 8 | 8 | 0 | 0 | 2.40 | 5.18 |
| large | 4 | slow | 10 | 9 | 1 | 0 | 1.42 | 2.96 |
| large | 4 | medium | 9 | 6 | 3 | 0 | 1.52 | 3.47 |
| large | 4 | fast | 10 | 9 | 0 | 1 | 2.82 | 4.83 |
| H3N2 fit points | 3 | real | 25 | 20 | 5 | 0 | 3.65 | 15.30 |
| H3N2 posterior states (post burn-in) | 3 | real | 361 | 222 | 100 | 39 | 1.59 | 5.40 |
| AIV host-order fit points | 3 | real | 8 | 1 | 7 | 0 | 0.24 | 7.12 |

Pooled simulated trees n=292: median abs MASCOT 0.097, LISCO 0.168, SISCO 0.279; MASCOT closest on 78%, |SISCO| > |MASCOT| on 86%, |LISCO| > |MASCOT| on 80%
  slow: n=98 median abs M/L/S 0.290/0.509/0.805; 90th pct 1.81/2.49/3.97
  medium: n=97 median abs M/L/S 0.210/0.284/0.393; 90th pct 0.51/0.73/1.08
  fast: n=97 median abs M/L/S 0.024/0.035/0.046; 90th pct 0.07/0.15/0.23

## Notes
- LISCO helps only partly. In the slow regime its median |error| is 1.2–3× MASCOT's. In the fast regime it is
  1.25–2.8×.
- In the slow regime with d ≥ 3, the sign is negative for all three, and most strongly for SISCO: the closure gives the
  tree too little likelihood. MASCOT's real-data errors are mostly **positive** (medians: H3N2 fit points +0.25, posterior
  states +0.35, AIV +6.5). There SISCO is strongly negative and LISCO is close to zero on average. Read this as MASCOT's κ = 1 conditioning
  overshooting on those trees. It is not evidence that LISCO is generally preferable, because the points are MASCOT/exact
  optima.
- Costs are the same as MASCOT (O(k·d) per ODE step). SISCO is cheaper because its migration-only ODE is linear.

## Note 2026-10-11: same tree set as the error map (281 trees)
The "small" set above has 239 trees: the 228 with a local dense value (bench/dense_vs_mascot.csv, used by the paper's
error map and Fig. 4) plus 11 four-deme trees with 13–14 lineages whose dense values come from kaggle/dense_big.csv. For a
single set in the paper, `bench/sisco_subset281.py` drops the 11: n = 281, median |error| MASCOT 0.100, LISCO 0.167,
SISCO 0.279, MASCOT closest on 77.2% (292-tree version: 0.097 / 0.168 / 0.279, 77.7%). Per regime (median abs M/L/S;
90th pct): slow 95 trees 0.294/0.504/0.764 (1.73/2.36/3.84), medium 92 0.201/0.285/0.386 (0.51/0.75/1.13), fast 94
0.024/0.035/0.045 (0.07/0.15/0.24). Slow d ≥ 3 SISCO medians −1.05 to −1.67 (worst −11.69). Conclusions unchanged.
