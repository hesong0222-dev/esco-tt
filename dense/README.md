# dense/ — exact ESCO reference likelihood (full joint vector)

`esco_dense.py` → `loglik(tree_json, method="auto") -> float` (contract interface; CLI:
`.venv/bin/python dense/esco_dense.py file.json ...` prints logL, peak lineages, time).

## Method
- State: tensor `P` with one axis per live lineage (shape `(d,)*n`), renormalized after every
  event; log normalizers summed into logL (contract).
- Between events: `dP/dt = Σ_i (Q^T on axis i) P − D ⊙ P`, `Q = M − diag(rowsum M)`,
  `D(l) = Σ_a C(n_a(l),2)/Ne[a]`. Convention `M[a][b]` = backward rate a→b (see sim/README.md).
- Two independent propagators:
  - `expm`: explicit sparse Kronecker-sum generator + `scipy.sparse.linalg.expm_multiply` (used for d^n ≤ 4096).
  - `unif`: matrix-free uniformization: `exp(Gh) = Σ_m Pois(m; Λh) B^m`, `B = I + G/Λ ≥ 0`,
    `Λ = max(E + D)`; substeps with `Λh ≤ 40`, `Dmax·h ≤ 20`, renormalized each substep; Poisson series
    truncated when the tail < 1e-15·exp(−Dmax h) (≤ 1e-15 relative to the surviving mass). Only
    nonnegative arithmetic, so no cancellation. Axis action is a batched `matmul` on a `(d^i, d, d^{n−i−1})` view.
  - single lineage: `expm(Q^T dt)` on the d-vector.
- Coalescence (a,b): `np.diagonal` over the two axes (merged axis appended last) × `1/Ne`, sum = event factor.
- Sample: outer product with the delta vector.

## Validation (`validate.py`, `validate_mc.py`; outputs in `bench/validate_*.txt`)
| check | result |
|---|---|
| (a) d=1 vs Kingman heterochronous closed form (k=2..20; also d=2 with M=0) | max abs. logL error 1.8e-15 |
| (b) k=2: analytic 2-state (same/different deme) solution, 5 parameter sets; d=3 asymmetric vs direct 9×9 `expm` | max error 5.3e-15 |
| (d) `expm_multiply` vs uniformization on all 130 test trees with d^n ≤ 4096 | max error 2.1e-13 |
| (c) Monte Carlo over migration histories (simulate migrations only, weight = exp(−∫D)·Π 1[same deme]/Ne; no master equation, no matrix exponential) | see below |

(c) details: 8 trees × 40k paths (scalar MC): z = +2.12, +0.62, −0.85, +0.79, +1.53, +0.49, −0.12, −1.77.
One slow-regime tree gave z = −2.96 with 200k paths; the cause is the heavy-tailed weight (rare migrations
dominate L, so modest samples underestimate). With 1e7 paths ×3 seeds (vectorized MC): z = +1.67, +0.35, −0.46.
Vectorized MC 4e6 paths over the k=4,6 trees: see `bench/validate_mc_vec.txt` (stopped after 27 trees, d=2,3,4: max |z| = 1.76, mean z = −0.02, sd 1.02, Σz² = 27.0 on 27 df, p = 0.46).
MC also confirms that MASCOT's large slow-regime errors are real (e.g. d4_k04_slow_r0: dense −12.721,
MC −12.752 ± 0.019, MASCOT −14.673).

## Size limit on this Mac (Apple M4, 16 GB, 10 cores; measured under heavy load from other jobs, load avg 200–400)
- Cost is set by the PEAK number of simultaneous lineages n (heterochronous sampling makes n < k).
- Memory ≈ 8–10 state vectors: 16.7M states (4^12) → ~1.1 GB peak (estimate); 4^13 = 67M states → ~5 GB (a single 4^13 matvec test alone peaked at 1.6 GB RSS with 3 vectors).
- One generator application: 4^10: 0.09 s, 4^11: 0.3 s, 4^12: 1.1 s (measured at lighter load); 4^13: ~40 s here (memory pressure).
- Whole-tree times: d=2 (n ≤ 15, ≤ 33k states): < 1 s; d=3 n=14 (4.8M): 16–70 s; d=3 n=15 (14.3M): 140–160 s;
  d=4 n=11 (4.2M): 35–90 s; d=4 n=12 (16.8M): 120–1460 s (fast migration needs more uniformization steps).
- Practical limit: **d^n ≈ 1.7e7 states (4^12, 3^15, 2^24)**. 4^13 is out of reach in reasonable time here;
  the 24 trees in `sim/data/large/` (d=4, k=30–60) and test-set trees with d=4, n ≥ 13 are dense-infeasible.
