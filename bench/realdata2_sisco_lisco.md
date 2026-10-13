# Real data 2 on the sisco and lisco MCC trees: does the masco result replicate?

Same 133 AIV H7 HA tips, same 3 bird-order demes (ans 93 / cha 32 / gal 8). The trees are the other two MCC trees
of Müller et al. (2017): `data/aivorder_sisco_mcc.json` and `data/aivorder_lisco_mcc.json`. The masco reference is
`realdata2_masco_refined.md`, the refined version that replaces the ranking claims of `realdata2.md`.

**Verdict: partially replicated.**

| claim from masco | sisco and lisco | status |
|---|---|---|
| MASCOT's error depends strongly on the peak | yes: about 0 at the ans-root hub, about 1.4 at the gal root, about 5.5–6 at the cha root, about 7 for the symmetric fits | **replicated** |
| asym vs sym LRT is much stronger under the exact likelihood | exact 2Δ 30.0 ± 0.05 (sisco), ≤ 30.9 (lisco) vs MASCOT 18.8 / 20.4 (masco: 29.6 ± 1.3 vs 14.1) | **replicated** |
| MASCOT and exact pick the same best peak | **no**: MASCOT's best is the cha root on both trees, exact's best is the gal root, ahead by 3.79 on lisco (converged) and by ≤ 4.56 on sisco (cha root still a lower bound; ≥ ≈3.77 with the 14× factor) | **not replicated: the rankings genuinely flip** |

So the result `realdata2.md` first claimed for masco (it did not survive refinement there) does happen on the other
two trees, this time with every competing peak refined. On sisco and lisco, MASCOT's two best peaks are within 0.04
and 0.25 of each other. The exact likelihood puts the gal root first, by ≤ 4.56 (sisco, cha root a lower bound) and 3.79 (lisco, converged) over the cha root. That changes
the inferred root (cha 0.95 → gal 0.999) and the story:
- MASCOT's best: the cha root, with ducks seeded from both shorebirds and poultry.
- Exact best: the gal root; poultry is the largest population and seeds the ducks.

The best exact peak also differs between trees: ans-root hub on masco, gal root on sisco and lisco. On masco the
ans-root hub wins by 4.7. On sisco the hub is 0.56 behind the gal root, on lisco 3.3 behind.

## What is converged and what is a bound (final, 2026-10-11 22:40)
- **Converged** (predicted gain < 0.01 or no accepted step left, KKT probes negative where rates sit at a bound, h/2
  error checked):
  - asymmetric: gal root and ans-root hub on both trees, and the lisco cha root (−262.365, h/2 error 3.0e-3)
  - symmetric: the sisco regular (−281.047, h/2 2.6e-2), the sisco hub (−283.237, h/2 3.5e-2) and the lisco hub
    (−277.873, h/2 9.9e-2)
- **Still a bound, and why:**
  - **sisco cha root ≥ −270.625.** Cut by Kaggle's 12 h limit twice (3 + 2 iterations). The last step gained only
    0.005; the last predicted gain was 0.057. The predicted gain is a quadratic-model estimate, not a guaranteed
    bound; on masco it underestimated the remaining gain by about 14× (0.044 predicted, 0.61 gained from another
    start). Even 14 × 0.057 = 0.8 leaves the gal root ahead by ≥ 3.77, so no further kernel was pushed.
  - **lisco sym regular ≥ −274.044.** Cut by the 12 h limit twice (2 + 2 iterations). The last predicted gain was
    0.028; the last step gained 0.033. Its LRT is an upper bound (≤ 30.94); with the 14× factor it is ≥ about 30.2.
    No conclusion depends on the difference.
  - **masco cha root ≥ −292.239.** Every step length was rejected, see `realdata2_masco_refined.md`. It is 6.1
    below the gal root.
- **Exact LRT** = 2·(best asym − best sym). Its TT error is taken as 2·(h/2 error of each end):
  - masco 29.59 ± 1.30 (the symmetric-hub TT error of 0.65 dominates)
  - sisco 29.97 ± 0.05
  - lisco ≤ 30.94 (h/2 at the unfinished symmetric point was not run)

  All three are well above MASCOT's 14.1 / 18.8 / 20.4, which are converged MASCOT optima.

## Cross-tree synthesis (asymmetric peaks by root type, plus the LRT)
Notation: "max−max" = MASCOT's maximum of a peak minus the exact maximum of the same peak (refined).
**≤** marks a value where the exact side is a lower bound.

| | masco | sisco | lisco |
|---|---|---|---|
| MASCOT best asym peak | ans-root hub −281.478 | cha root −264.670 | cha root −256.886 |
| MASCOT 2nd | old hub (mode A) −284.408 | gal root −264.705 (Δ 0.035) | gal root −257.131 (Δ 0.245) |
| exact best asym peak | **ans-root hub −281.507** | **gal root −266.060** | **gal root −258.572** |
| exact 2nd | gal root −286.180 | ans-root hub −266.621 | ans-root hub −261.906 |
| exact cha root | ≥ −292.239 (bound) | ≥ −270.625 (bound) | −262.365 |
| same best peak? | yes | **no** (gap ≤ 4.56; ≥ ≈3.77 with 14× factor) | **no** (gap 3.79) |
| root of best: MASCOT → exact | ans 1.000 → ans 1.000 | cha 0.951 → gal 0.999 | cha 0.947 → gal 0.999 |
| max−max, ans-root hub | +0.03 | +0.02 | +0.06 |
| max−max, gal root | +1.61 | +1.36 | +1.44 |
| max−max, cha root | ≤ +7.20 | ≤ +5.95 | +5.48 |
| max−max, sym regular | +7.31 | +6.98 | ≤ +6.96 |
| max−max, sym hub | +7.77 | +7.14 | +6.87 |
| LRT 2Δ, MASCOT (p, df 3) | 14.1 (0.003) | 18.8 (3e-4) | 20.4 (1.4e-4) |
| LRT 2Δ, exact ± TT error (p) | 29.6 ± 1.3 (1.7e-6) | 30.0 ± 0.05 (1.4e-6) | ≤ 30.9 (≥ 8.7e-7) |
| best sym peak (both engines) | hub | regular | regular |

The mechanism is the same on all three trees. MASCOT is almost exact at the ans-root hub (Ne gal ≈ 0.3, cha←gal
16–86/y, every other rate except gal←ans at the bound). Its error is about 1.4 at the gal-root peak and 5.5–7 at the
cha-root peak and the symmetric fits. Whichever peak wins exactly therefore depends on how the true heights compare,
and MASCOT, adding peak-specific offsets of 0 to 7, can rank them wrongly. On masco it ranked the top correctly (but
had to be found by refinement), on sisco and lisco it does not.

## Asymmetric peaks per tree (eps 1e-6)

**sisco**

| peak | MASCOT starts | MASCOT logL | exact at MASCOT opt. | refined exact | root exact (ans/cha/gal) | h/2 error |
|---|---|---|---|---|---|---|
| cha root | 7 | −264.670 | −271.448 | ≥ −270.625 (3 + 2 it, 12 h limit twice, pred. 0.057) | 0.000 / 0.937 / 0.063 | — |
| **gal root** | 5 | −264.705 | −266.240 | **−266.060** (5 it, conv., KKT < 0) | 0.000 / 0.001 / 0.999 | 2.4e-4 |
| ans-root hub (gal Ne 0.27, cha←gal 31/y) | 1 (+ masco seed) | −266.604 | −266.637 | −266.621 (conv., KKT < 0) | 1.000 / 0 / 0 | 4.0e-4 |
| ans root | 1 | −273.091 | −273.086 | not refined | 1.000 / 0 / 0 | — |
| ans root | 1 | −275.319 | −275.583 | not refined | 1.000 / 0 / 0 | — |

**lisco**

| peak | MASCOT starts | MASCOT logL | exact at MASCOT opt. | refined exact | root exact | h/2 error |
|---|---|---|---|---|---|---|
| cha root | 7 | −256.886 | −263.323 | −262.365 (3 + 3 it, conv., KKT < 0) | 0.000 / 0.934 / 0.066 | 3.0e-3 |
| **gal root** | 5 | −257.131 | −258.783 | **−258.572** (5 it, conv., KKT < 0) | 0.000 / 0.001 / 0.999 | 7.2e-4 |
| ans-root hub (gal Ne 0.36, cha←gal 86/y, cha Ne at the upper bound 1097) | 2 (masco/sisco hub seeds only) | −261.846 | −261.906 | −261.906 (conv. at once, KKT < 0) | 1.000 / 0 / 0 | 6.1e-4 |
| ans root (cha Ne 0.49 type) | 1 | −266.754 | −266.740 | not refined | 1.000 / 0 / 0 | — |
| ans root | 1 | −269.092 | −269.363 | not refined | 1.000 / 0 / 0 | — |

Re-fitting MASCOT from the refined exact gal-root optima returns to the same MASCOT gal-root optima on all three
trees. So no hidden better MASCOT peak showed up in that direction, unlike the ans-root hub on masco.

## Symmetric peaks

| tree | peak | MASCOT logL | exact at MASCOT opt. | refined exact | status | h/2 error |
|---|---|---|---|---|---|---|
| sisco | regular (cha root 0.67) | −274.067 | −281.655 | −281.047 | 2 + 2 it, conv. (pred. 0.005) | 2.6e-2 |
| sisco | hub (gal Ne 0.10) | −276.101 | −283.302 | −283.237 | stopped (line search), pred. 0.026, KKT < 0 | 3.5e-2 |
| lisco | regular (cha root 0.74) | −267.082 | −274.688 | ≥ −274.044 | 2 + 2 it, 12 h limit twice, pred. 0.028 | — |
| lisco | hub (gal Ne 0.10) | −271.007 | −277.884 | −277.873 | 2 it, conv. (pred. 0.007), KKT < 0 | 9.9e-2 |

Parameters at the two competing asymmetric peaks (backward rates per year):

| | sisco exact gal root | sisco cha root (exact, last iteration) | lisco exact gal root | lisco cha root (exact, converged) |
|---|---|---|---|---|
| Ne ans / cha / gal | 1.77 / 0.61 / 3.40 | 1.52 / 0.58 / 5.02 | 1.73 / 0.50 / 3.44 | 1.45 / 0.47 / 5.29 |
| rates | m02 1.62, m12 0.46, m21 0.03 | m01 0.23, m02 1.57, m12 0.44, m21 0.76 | m02 1.61, m12 0.48, m21 0.03 | m01 0.09, m02 1.74, m12 0.53, m21 0.78 |

The two peaks differ mainly in whether shorebirds also seed ducks (m01) and how strongly poultry receives from
shorebirds (m21). That is enough to move the root from gal to cha.

## Method
- **MASCOT optima:** 10 random starts (seed 1, as on masco), plus the zero start, plus starts from each masco MASCOT
  optimum and from the masco ans-root hub. Optima are grouped at |ΔlogL| < 1e-3 (`bench/realdata2_sl/extract.py`).
- **Exact evaluation:** TT at eps 1e-6 at every MASCOT optimum.
- **Refinement:** quasi-Newton with the MASCOT Hessian (`bench/realdata2_refine.py`) of every competing peak. Stops
  at predicted gain < 0.01, cap 8 iterations, KKT probe at 0.02. Continuations restart from the last saved iteration.
- **Error check:** h/2 paired error at the refined optima.
- **Where it ran:** TT on the local Mac (serial, cached: `bench/realdata2_sl/ttq.py`, `queue.sh`) until 2026-10-11
  00:40, then on Kaggle, one job per kernel (`kaggle/realdata2sl/push2.sh`, `run_template2.py`). Outputs are merged
  by `kaggle/realdata2sl/collect.sh` into `bench/realdata2_sl/`.
- **Tables:** `bench/realdata2_sl_summarize.py` → `bench/realdata2_sisco_lisco.csv`.

## Limitations
- Fixed MCC trees, no tree uncertainty. A 3-deme ML fit, not the published 5-type Bayesian setup.
- An exact-only peak (one with no MASCOT optimum in its basin) would be missed. Masco showed that even MASCOT's own
  best peak can be missed by 10 random starts.
- Two values remain bounds (sisco cha root, lisco sym regular; see above). Neither can change a conclusion. On the
  lisco hub peak, cha Ne sits at the upper bound (1097, i.e. effectively infinite).
- Rates at the lower bound make the χ² LRT approximate.


**Correction 2026-10-11 (draft v1.5 review).** Earlier versions of this file wrote the gal−cha gap as "≥ 4.59 / ≥ 3.85". The cha-root exact values are lower bounds, so the gap is an upper bound (≤ 4.59 / ≤ 3.85); with the 14× safety factor on the remaining gain it is at least about 4.47 / 3.51. The conclusion (the best peak differs) is unchanged.

**Final values 2026-10-11 22:40.** All continuation kernels were collected. The following changed from the version
above this note:
- lisco cha root −262.417 (bound) → −262.365 (converged); its gal−cha gap ≤ 3.85 → 3.79.
- sisco cha root ≥ −270.648 → ≥ −270.625; gap ≤ 4.59 → ≤ 4.56.
- sisco sym regular ≥ −281.077 → −281.047 (converged); exact LRT ≤ 30.0 → 29.97 ± 0.05.
- lisco sym regular ≥ −274.225 → ≥ −274.044; exact LRT ≤ 31.3 → ≤ 30.94.
- lisco sym hub −277.884 (unrefined) → −277.873 (refined).
- max−max: sisco cha ≤ 5.98 → ≤ 5.95; lisco cha ≤ 5.53 → 5.48; sisco sym regular ≤ 7.01 → 6.98; lisco sym
  regular ≤ 7.14 → ≤ 6.96; lisco sym hub +6.88 (at the MASCOT optimum) → +6.87.

The verdict is unchanged.
