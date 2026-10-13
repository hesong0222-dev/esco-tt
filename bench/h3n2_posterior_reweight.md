# H3N2 posterior: reweighting the MASCOT posterior sample to the exact-ESCO posterior

Answers review/AUDIT_DESIGN.md M2 ("Bayesian-level comparison", line 68; action 3).
Exact values: `bench/h3n2_posterior_eval.py` → `bench/h3n2_posterior_eval.csv`, one row per logged state with exact,
converged MASCOT, the run's logged density, SISCO and LISCO log-likelihoods and both root distributions. Analysis: `bench/h3n2_posterior_reweight.py`
→ `bench/h3n2_posterior_reweight.csv` + `bench/h3n2_posterior_reweight_tables.txt`. Local Mac, OpenBLAS .venv, one
process, nice 10, about 2.3 h.

## Bottom line (for the paper)
1. We computed the exact likelihood at every one of the 401 logged states of the Taming-the-BEAST H3N2 MASCOT run (tree
   and parameters change together). MASCOT's model error at these states is larger than at the ML points. The exact value
   is lower at 92% of states, with ℓ_exact − ℓ_MASCOT median −0.35 and range −2.92 … +0.40. It is nearly a constant
   offset (SD 0.44), so the importance weights are benign: weight ESS 309 of 361 states (86%) after 10% burn-in, and the
   largest normalised weight is 0.008.
2. The exact-likelihood posterior moves the means of Ne and the six migration rates by −6% to +7% relative to the
   converged-MASCOT posterior. That is ≤ 0.09 posterior SD, while the 95% HPD intervals span one to four orders of
   magnitude. Tree height rises 1.4% and clock rate falls 0.8%. The root-deme probability of Hong Kong goes from 0.50 to
   0.53. No conclusion of the tutorial analysis changes.
3. Caveat: reweighting can only move mass among states the MASCOT chain visited. It cannot recover regions that MASCOT's
   posterior misses. With weights this flat (ESS 86%) such regions are unlikely here, but this is not a proof. The tree
   varies across states, so this reweights the joint (tree, parameter) posterior. The priors and the sequence
   likelihood cancel.

## Design
- Data: `data/raw/H3N2.log` and `H3N2.trees` (tutorial precooked run, chain length 2e6, logged every 5000 → 401 states;
  provenance in data/README.md). Burn-in is 10% (states with Sample ≥ 200000, n = 361). Sensitivity checks use 0% and 25%.
- Per state: ℓ_S is the structured-coalescent density logged by the run itself (column `Mascot`), i.e. the density
  the chain targeted. ℓ_ours is the converged `approx/mascot.py` value. ℓ_exact is dense for peak ≤ 15 lineages
  (396 states) and TT eps 1e-6 for peak 16–17 (5 states; eps 1e-7 re-runs of two of them differ by 1.6e-4 and 2.1e-4).
- The chain samples π_S ∝ prior · P(seq | T) · exp(ℓ_S). We compare two posteriors on the same states:
  - **C**, the converged MASCOT model (w ∝ exp(ℓ_ours − ℓ_S));
  - **E**, exact (w ∝ exp(ℓ_exact − ℓ_S)).
  The approximation-error step is C → E, with weights ∝ exp(ℓ_exact − ℓ_ours).
- Root deme: the posterior mean of the per-state root distribution. For C this is the MASCOT root recursion
  (bench/realdata_fit.mascot_root). For E it is the exact root distribution (dense / TT).

## Weights
| step | log-weight median | log-weight SD | Kish ESS (n = 361) |
|---|---|---|---|
| C → E (MASCOT model → exact) | −0.350 (range −2.92 … +0.40) | 0.438 | 320.5 |

Kish ESS of the weights from the sampled states: 352.9 to C, 309.1 to E (bootstrap 301–316), largest normalised
weight to E 0.008.

corr(ℓ_exact − ℓ_ours, tree height) = +0.24. MASCOT overestimates least on the tallest trees. ESS from the sampled states to E is
340/401 with no burn-in and 257/301 with 25% burn-in.

## Posterior summaries (10% burn-in)
Rates are backward-time per year (m_a_to_b = lineage in a comes from b); Ne in years.

| quantity | C mean | E mean | E 95% HPD | C → E shift | 95% CI (bootstrap over states) | shift / posterior SD |
|---|---|---|---|---|---|---|
| Ne HK | 1.404 | 1.410 | 0.165 – 4.27 | +0.5% | −3.2 … +4.0% | +0.004 |
| Ne NY | 1.961 | 1.962 | 0.179 – 5.80 | +0.1% | −3.1 … +3.0% | +0.001 |
| Ne NZ | 0.868 | 0.895 | 0.133 – 2.19 | +3.1% | +1.0 … +5.4% | +0.042 |
| m HK→NY | 1.150 | 1.177 | 0.019 – 3.88 | +2.3% | −1.0 … +5.5% | +0.023 |
| m HK→NZ | 0.497 | 0.468 | 0.0002 – 1.50 | −5.9% | −9.7 … −1.6% | −0.050 |
| m NY→HK | 1.357 | 1.447 | 0.0075 – 3.35 | +6.6% | +4.2 … +8.9% | +0.089 |
| m NY→NZ | 0.600 | 0.567 | 0.0008 – 1.74 | −5.4% | −8.8 … −2.4% | −0.054 |
| m NZ→HK | 0.739 | 0.774 | 0.0030 – 2.13 | +4.8% | +2.0 … +7.6% | +0.053 |
| m NZ→NY | 0.825 | 0.830 | 0.0063 – 2.18 | +0.6% | −2.1 … +3.3% | +0.006 |
| tree height (y) | 3.614 | 3.664 | 2.63 – 4.63 | +1.4% | +0.8 … +2.0% | +0.095 |
| clock rate | 3.995e-3 | 3.962e-3 | 2.74e-3 – 5.02e-3 | −0.8% | −1.3 … −0.3% | −0.054 |

The HPDs of C are listed in `h3n2_posterior_reweight_tables.txt`. They differ from E's mainly at the lower end of
the rates, where the chain has few states. The bootstrap CI resamples states as if they were independent. It ignores chain
autocorrelation, so it is too narrow. The shifts are paired: the same states are used, only the weights change.

## Root deme (posterior mean probability)
| posterior | root model | HK | NY | NZ |
|---|---|---|---|---|
| C | MASCOT | 0.495 | 0.264 | 0.240 |
| E (weights only) | MASCOT | 0.528 | 0.262 | 0.210 |
| E | exact | 0.531 | 0.263 | 0.206 |

Almost all of the change comes from the weights (+0.033 HK). The per-state root recursion adds +0.003, because the exact
and MASCOT root distributions at the same state differ little (median |ΔP(HK)| 0.012, max 0.152). With 0% / 25% burn-in,
HK goes 0.482 → 0.515 and 0.508 → 0.541.

## Comparison with the fixed-tree results (bench/realdata.md)
At the ML points on the MCC tree, MASCOT − exact was +0.08 to +0.38. Across posterior states the median is +0.35 and the
maximum +2.9. The posterior covers trees and parameter values (smaller Ne, larger rates) where the closure is worse than at
the MLE. Even so, the error is mostly a common offset. It cancels in the weights, so the posterior barely moves. This is the
practical sense in which MASCOT is adequate for this dataset.

## Caveats
- Support: an importance sampler from π_B cannot reach parameter or tree regions π_B does not visit. Its accuracy is
  limited by the 361 states (thinned, and autocorrelated in the chain). A full exact-likelihood MCMC was not run.
- One dataset (24 tips, d = 3, medium/fast regime by bench/realdata.md). The AIV host data, where MASCOT errors at fit
  points reach +8, would be the stronger test, but no BEAST2-MASCOT posterior sample exists for that 3-deme host
  coding. Each exact evaluation there costs 10–100 min (TT), so 300 states would need Kaggle.
