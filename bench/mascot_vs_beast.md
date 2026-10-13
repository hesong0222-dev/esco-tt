# Our MASCOT vs BEAST2-MASCOT logged density (H3N2 tutorial): cause of the 0.003–0.46 gap

**Bottom line.** The gap comes from how BEAST's MASCOT integrates the ODE. It is not a model or convention difference.
I re-implemented MASCOT v3.0.0's integrator in Python (`bench/mascot_vs_beast.py`, function `loglik_beast`) and
reproduced BEAST's logged `Mascot` density on **all 401 logged states** of the tutorial run to within **2.1e-14**
(floating-point noise). Two numerical features of that integrator make up the whole gap:
1. a coarse step size (2nd-order Taylor steps chosen from epsilon = 1e-3), and
2. a bookkeeping error: the second-derivative term of the log-likelihood carries over from one step to the next.

Our `approx/mascot.py` (DOP853, rtol 1e-10) is the converged solution of the MASCOT equations. When I correct the
carried term and shrink the step (eps = 1e-9), the BEAST scheme agrees with ours to ≤ 1.8e-5 on all 401 states.
No paper claim changes, because every comparison in the paper is exact vs our converged MASCOT. Only the
"residual is unexplained" sentence in Methods needs replacing (text below).

## Source read
github.com/nicfel/Mascot, tag **v3.0.0** (commit 7943662, 2022-11-17). The tutorial XML requires `Mascot v3.0.0`
(`data/raw/H3N2.xml`, and the model block in the `H3N2.log` header). I re-checked the relevant lines at HEAD b62d9ad
(2026-07-14) and they are unchanged.

- `src/mascot/distribution/Mascot.java`
  - Inputs: `epsilon` default 0.001 and `maxStep` default +∞. The tutorial XML sets neither.
  - `implementation` defaults to `allnative`, which uses JNI `jni/Mascot.cpp` + `Euler2ndOrderCPU` if `libmascot`
    loads. Otherwise it falls back to Java `Euler2ndOrder3` (3 states).
  - `useCache` defaults to false, so there is no history-dependent caching.
  - Events:
    - Sampling appends a one-hot lineage.
    - Coalescence adds `log Σ_a λ_a p_i(a) p_j(a)`, with parent `∝ λ_a p_i p_j`.
    - λ_a = 1/(ploidy·Ne_a) with ploidy 1 (`dynamics/Constant.java`).
    - The backward rate a→b is `m[a*n+b]`, enumerated a-major over a≠b, which is the same order as the log column
      labels.
  - There is no conditioning, no root term and no extra tree-prior term. The root simply ends the loop. The model and
    conventions are identical to ours.
- `src/mascot/ode/Euler2ndOrder3.java` and `jni/Euler2ndOrderCPU.cpp` are the same algorithm. Per step:
  - Compute p′ and p″ exactly (closure ODE).
  - Approximate the third derivative as p‴ ≈ (C_i − T_ia)·p″ + migration(p″).
  - Step: h = min(∛(6·eps / max|p‴|), remaining interval, maxStep). Shrink h by 0.9 while any updated p would leave
    [0,1] or move by > 0.2.
  - Update p += h p′ + h²/2 p″. The log-likelihood slot gets the same second-order update.
  - Renormalise every lineage after each step.
  - There is no error estimate and no step rejection, so eps does not control the error in the log-likelihood.
- **Carried log-slot second derivative.** `pDotDot` is zeroed only once per inter-event interval
  (`calculateValues`). In each step, `computeDerivatives` overwrites the p entries of `pDotDot`, but
  `computeSecondDerivate` does `pDotDot[log] -= Σ pCoalRate; pDotDot[log] /= 2` on the old value. So the value used is
  ℓ″_n = ℓ″_{n−1}/2 + ℓ″_n(true) instead of ℓ″_n(true). For a slowly varying interval this is about 2× the true
  second-order term. `pDot` is cleared every step, so the first-order term is correct. The same pattern is in the Java
  and the native code, and it is still present at HEAD.

## Reproduction (all 401 states of `data/raw/H3N2.log` and `H3N2.trees`, same STATE_n tree and parameters)

`.venv/bin/python bench/mascot_vs_beast.py` → `bench/mascot_vs_beast.csv` (about 6 min on a laptop).
Parameters: M[a][b] = `b_migrationConstant.a_to_b`, Ne = `NeConstant.a`. Trees are converted with `data/convert.py`.

| quantity (log units) | max abs | mean abs | median | range |
|---|---|---|---|---|
| ours (DOP853) − BEAST logged | 0.86 | 0.132 | −0.062 | [−0.86, +0.15] |
| **emulated BEAST (eps 1e-3, carry) − BEAST logged** | **2.1e-14** | 3.9e-15 | 0 | — |
| step error only: emulation without carry, eps 1e-3 − ours | 0.175 | 0.026 | +0.008 | [−0.058, +0.175] |
| carry effect: emulation with − without carry, eps 1e-3 | 0.72 | 0.113 | +0.057 | [−0.13, +0.72] |
| emulation with carry, eps 1e-9 − ours | 0.016 | 0.004 | +0.004 | [−0.002, +0.016] |
| emulation without carry, eps 1e-9 − ours | 1.8e-5 | 3.4e-6 | — | — |

- Steps per tree at eps 1e-3: 55–123 (median 75).
- BEAST's value is higher than ours in 73% of states. The carry term is the larger of the two contributions. It does
  not vanish quickly as eps → 0 (still up to 0.016 at eps = 1e-9), because the carried term is O(h) per unit time.
- The earlier 10-sample check (0.003–0.46) was a subset. Over all 401 states the gap reaches 0.86.

## Effect on the paper
- **No claim changes.** All exact-vs-MASCOT numbers (simulations, large trees, H3N2 fits) use `approx/mascot.py` at
  rtol 1e-10. That code agrees with the BEAST scheme in the limit (corrected carry, eps → 0) to 1.8e-5, so "MASCOT"
  in the paper means the converged MASCOT model. The convention check still holds: the emulation reproduces BEAST
  exactly with our M orientation and λ = 1/Ne.
- **Worth one sentence.** In BEAST2 as run in the tutorial, the numerical error of the MASCOT implementation (up to
  0.86, mean 0.13 log units here) is of the same size as MASCOT's model error on this dataset (+0.08 to +0.38,
  `bench/realdata.md`). We did not test whether this shifts BEAST posteriors. It is a fixed-tree, fixed-parameter
  statement only.
- Suggested replacement for the Methods sentence ("The remaining residual is unexplained; …"):
  "The remaining differences (up to 0.86 log units over all 401 logged states) are numerical error in BEAST's MASCOT
  integrator (adaptive second-order Taylor steps with tolerance 10⁻³, and a log-likelihood second-derivative term
  that is not reset between steps); re-implementing that integrator reproduces every logged value to 10⁻¹³
  (bench/mascot_vs_beast.md)."

## Limits
- I could not tell from the files which path (JNI or Java) produced the tutorial log. Both have the same step rule
  and the same carry, and the emulation matches to 1e-14, so it does not matter here.
- One dataset, 3 demes, `Constant` dynamics. Other dynamics (skyline rate shifts, indicators) use the same integrator,
  but I did not test them.
