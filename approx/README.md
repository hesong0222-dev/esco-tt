# approx/ — MASCOT approximation

`mascot.py` → `loglik(tree_json) -> float` (contract interface; also `python approx/mascot.py file.json ...`).

Equations from Müller, Rasmussen & Stadler 2018 (Bioinformatics 34(22):3843, "MASCOT"), checked against the paper text:

- per-lineage state probabilities `p_i(a)` evolve independently (pairwise-independence closure):
  `dp_i(a)/dt = Σ_b [m_ba p_i(b) − m_ab p_i(a)] + p_i(a) Σ_b λ_b p_i(b) Σ_{k≠i} p_k(b) − p_i(a) λ_a Σ_{k≠i} p_k(a)`
- interval: `d log P(T)/dt = − Σ_a λ_a Σ_{i<j} p_i(a) p_j(a)`
- coalescence of i,j: `P(T) *= Σ_a λ_a p_i(a) p_j(a)`, parent `p(a) ∝ λ_a p_i(a) p_j(a)`
- sampling: new lineage with `p = δ(deme)`.
- `λ_a = 1/Ne[a]`, `m_ab = M[a][b]` (backward rate a→b, same convention as sim/ and dense/).

Integrated with `solve_ivp(DOP853, rtol=1e-10, atol=1e-12)`; O(k·d) per step, milliseconds per tree.

Checks: d=1 equals the Kingman closed form exactly (MASCOT is exact there). Comparison with the exact
dense likelihood: `bench/dense_vs_mascot.md`.

## independence.py — SISCO and LISCO (Müller, Rasmussen & Stadler 2017, MBE)
`loglik(tree, method="sisco"|"lisco"|"mascot")`. Same events and interval rate as MASCOT; the lineage ODE keeps a fraction
κ of MASCOT's no-coalescence conditioning terms: κ=1 MASCOT, κ=1/2 LISCO (derived from the upstream
`IndependentStructuredCoalescent.java`), κ=0 SISCO (paper eq. 4: migration only). Derivation, checks and results:
`bench/sisco.md`, `bench/sisco_validate.py`.
