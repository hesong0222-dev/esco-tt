# Dense (exact ESCO) vs MASCOT on the sim/ test set

Data: `bench/dense_vs_mascot.csv` (one row per tree; `mascot_minus_dense` = logL_MASCOT − logL_exact).
Reproduce: `.venv/bin/python bench/run_dense_vs_mascot.py && .venv/bin/python bench/summarize.py`.
Dense computed for 228/243 trees (peak states ≤ 4^12 = 16.8M); the 15 skipped are d=4 trees with peak n ≥ 13.
Regimes: per-pair migration = 0.1× (slow), 1× (medium), 10× (fast) the mean per-deme coalescence rate (sim/README.md).

## Headline
- **fast migration**: MASCOT is essentially exact: mean bias −0.004 log units, mean |err| 0.025, worst 0.12, no trend with k or d.
- **medium**: mean −0.049, mean |err| 0.22, worst −0.70 (≈ 0.006 per coalescence on average).
- **slow**: large and skewed errors: mean −0.44, mean |err| 0.67, worst −6.46 (d4_k12_slow_r0). The bias grows with d
  (d=2: +0.06, slightly over; d=3: −0.53; d=4: −0.93): MASCOT **underestimates** the likelihood when demes are
  nearly isolated, because its pairwise-independence closure cannot represent the strong co-location correlation
  between lineages whose coalescence forced them into the same deme.
- Independent check that the slow-regime gap is MASCOT's error and not the dense code: d4_k04_slow_r0 dense −12.721,
  Monte Carlo over migration histories (2e7 paths) −12.752 ± 0.019, MASCOT −14.673; d3_k04_slow_r1 dense −14.294,
  MC −14.285 ± 0.018, MASCOT −15.893.

### By regime
| group | n | mean | median | mean abs | min | max |
|---|---|---|---|---|---|---|
| slow |  77 | -0.443 | +0.011 | 0.671 | -6.459 | +0.802 |
| medium |  75 | -0.049 | +0.002 | 0.220 | -0.700 | +0.528 |
| fast |  76 | -0.004 | -0.005 | 0.025 | -0.116 | +0.057 |

### By regime x d
| group | n | mean | median | mean abs | min | max |
|---|---|---|---|---|---|---|
| slow, d=2 |  27 | +0.061 | +0.085 | 0.107 | -0.190 | +0.296 |
| slow, d=3 |  27 | -0.534 | -0.161 | 0.678 | -2.708 | +0.555 |
| slow, d=4 |  23 | -0.928 | -0.294 | 1.326 | -6.459 | +0.802 |
| medium, d=2 |  27 | -0.034 | +0.028 | 0.226 | -0.700 | +0.528 |
| medium, d=3 |  27 | -0.109 | -0.083 | 0.238 | -0.692 | +0.405 |
| medium, d=4 |  21 | +0.007 | +0.086 | 0.189 | -0.682 | +0.342 |
| fast, d=2 |  27 | -0.002 | +0.002 | 0.020 | -0.070 | +0.038 |
| fast, d=3 |  27 | -0.005 | -0.004 | 0.033 | -0.116 | +0.057 |
| fast, d=4 |  22 | -0.008 | -0.012 | 0.022 | -0.054 | +0.054 |

### By regime x k (mean MASCOT - dense; all d pooled)

| regime | k=4 | k=6 | k=8 | k=10 | k=12 | k=14 | k=16 | k=18 | k=20 |
|---|---|---|---|---|---|---|---|---|---|
| slow | -0.415 | +0.069 | -0.001 | -0.787 | -0.754 | -1.049 | -0.052 | -0.348 | -0.640 |
| medium | -0.085 | -0.010 | +0.013 | -0.032 | +0.015 | -0.046 | -0.024 | -0.339 | -0.027 |
| fast | -0.004 | -0.001 | -0.010 | -0.000 | +0.007 | -0.007 | -0.020 | +0.004 | -0.007 |

### Bias per coalescent event, (MASCOT - dense)/(k-1), by regime
| group | n | mean | median | mean abs | min | max |
|---|---|---|---|---|---|---|
| slow |  77 | -0.048 | +0.001 | 0.078 | -0.651 | +0.243 |
| medium |  75 | -0.006 | +0.000 | 0.025 | -0.169 | +0.087 |
| fast |  76 | -0.001 | -0.000 | 0.003 | -0.011 | +0.008 |

(dense skipped for 15 trees whose peak state count exceeds the dense cap 4^12.)
