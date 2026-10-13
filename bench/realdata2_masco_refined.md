# Real data 2, masco tree: refined exact optima for all peaks (revises realdata2.md)

**Bottom line.**
1. The ranking flip of `realdata2.md` (MASCOT best = hub, exact best = gal root) does **not** survive. Refining the hub
   basin exactly moved it to an Anseriformes-root hub solution with exact logL −281.51 (converged), 4.7 above the gal root.
   MASCOT, started from that point, finds its own best optimum there (−281.48, ans root 1.000), which the 10-start
   multistart had missed. Both engines now put the best asymmetric fit in the same basin with the same root.
2. What does survive: MASCOT's error depends on the basin (+0.03 at the ans-root hub, +1.5 at the gal root, +7.2 at
   the cha root), so the exact likelihood separates the secondary optima that MASCOT puts within 0.63. And the
   asymmetry LRT gets stronger under the exact likelihood: 2Δ = 29.6 ± 1.3 (exact, TT error) vs 14.1 (MASCOT). Both are now
   significant, at p = 2e-6 and p = 0.003.
3. The "both engines reject an Anseriformes root" statement of `realdata2.md` is also overturned. The best fit under
   both engines has the root in Anseriformes, as in the published analyses.

**Update 2026-10-11.** The Kaggle refinement started from the new MASCOT optimum (`kaggle/esco-rd2sl-masco-asym-hubans`)
converged immediately. The exact logL there is −281.507: predicted gain 0.007, every KKT probe negative, h/2 error
2.4e-4. That is 0.03 below MASCOT's −281.478 at the same point. The capped mode-A run, continued for 4 more
iterations from its last point (`kaggle/esco-rd2sl-masco-asym-modeacont`), climbs towards the same point:
−281.587 at a predicted gain of 0.009. **So the exact and MASCOT maxima of the best basin coincide to within
0.03 in logL; the exact refinement accepted no step away from MASCOT's optimum.**

## Method
Same as `realdata2.md`: tree `data/aivorder_masco_mcc.json`, TT eps 1e-6, quasi-Newton refinement with the MASCOT
Hessian as the curvature model (`bench/realdata2_refine.py`). The stopping rule is a predicted gain < 0.01, the
cap is 8 iterations, and a KKT probe raises each rate at its bound to 0.02. The runs were:
- asym hub (mode A) and sym regular: local Mac, serial and cached (`bench/realdata2_sl/ttq.py`).
- asym cha root (mode C) and sym hub: Kaggle, 4 CPUs (`kaggle/realdata2sl`, outputs in
  `kaggle/out/esco-rd2sl-masco-*`).
- The sym hub run continued from the best point of the stopped earlier run.

The paired error is |TT(eps 1e-6) − TT(eps 1e-6, half time step)| at each refined point.

## Asymmetric model

| peak | MASCOT logL at MASCOT opt. | exact at MASCOT opt. | refined exact | iterations / stop | root exact (ans/cha/gal) | h/2 error |
|---|---|---|---|---|---|---|
| **ans-root hub**, Kaggle run from the new MASCOT opt. | **−281.478** | −281.507 | **−281.507** | converged at once (0.007), KKT all < 0 | 1.000 / 0.000 / 0.000 | 2.4e-4 |
| ans-root hub, reached from mode A | −284.408 (mode A) → **−281.478** (new MASCOT opt.) | −290.300 (mode A) | **−282.118** (fit file); −282.038 at the last accepted step | 8, cap hit (0.044); +4 more: −281.587, converged (0.009) | 0.9996 / 0.0000 / 0.0004 | 8.0e-4 |
| gal root (mode B) | −284.569 | −286.311 | −286.180 | 3, converged (0.007) | 0.000 / 0.001 / 0.999 | 4.5e-4 (eps 1e-7: 8.1e-4) |
| cha root (mode C) | −285.041 | −292.239 | −292.239 (no accepted step) | 1; step 1, ½, ¼ all lower (−359.9, −298.9, −292.29) | 0.000 / 0.974 / 0.026 | 4.1e-4 |
| ans root, m02 → 0 | −295.419 | −295.661 | not refined | — | 1.000 / 0 / 0 | — |

At the refined ans-root hub point (iteration 7), MASCOT gives −282.004 and exact gives −282.118, a difference of +0.11.

Parameters (backward rates per year):

| | ans-root hub, exact (it 7) | ans-root hub, new MASCOT opt. | gal root, exact | mode A, MASCOT (old "best") |
|---|---|---|---|---|
| Ne ans / cha / gal | 5.17 / 3.02 / 0.29 | 5.22 / 3.03 / 0.30 | 2.17 / 0.93 / 3.32 | 5.28 / 2.93 / 0.27 |
| non-zero rates | m12 16.5 (cha←gal), m20 0.57 (gal←ans), m01 0.0095 | m12 16.9, m20 0.51 | m02 1.38, m12 0.39, m21 0.03 | m12 15.9, m20 0.44, m01 0.18 |
| KKT (bound rate → 0.02) | m02 −1.19, m10 −0.03, m21 −0.006 | — | m01 −0.33, m10 −0.06, m20 −0.39 | — |

The refinement from mode A kept the hub structure: poultry is a small population (Ne ≈ 0.3) exchanging lineages with
shorebirds at about 16/y. It mainly turned m01 (ans←cha) from 0.18 down to about 0.01. That moved the root from
gal 0.75 / ans 0.25 to ans 0.9996, and the likelihood rose by 8.2 under the exact engine and by 2.9 under MASCOT.
The MASCOT multistart had not found this better MASCOT optimum, because none of its 10 starts landed in that basin.

**Exact ranking after refinement:** ans-root hub (−281.51) > gal root (−286.18) > cha root (−292.24, a lower
bound: the quasi-Newton step was rejected, see Limitations) > ans root with m02 = 0 (−295.66, unrefined).
**MASCOT ranking with the new optimum:** ans-root hub (−281.48) > mode A (−284.41) > gal root (−284.57) > cha
root (−285.04). Both engines put the same peak on top.

## Symmetric model

| peak | MASCOT logL | exact at MASCOT opt. | refined exact | iterations / stop | root exact | h/2 error |
|---|---|---|---|---|---|---|
| hub (gal Ne 0.10, cha–gal 22/y) | −288.531 | −296.323 | **−296.304** | earlier run 2 + continued run 1, pred. gain 0.024, step ½ and ¼ lower | 0.100 / 0.022 / 0.878 | 0.65 |
| regular | −294.624 | −302.617 | −301.930 | 3, converged (0.004) | 0.200 / 0.611 / 0.190 | 2.1e-3 |

The sym hub is converged to within about 0.02 (the predicted gain). That is well inside its 0.65 TT error. The
regular solution ends 5.6 below the hub, so the hub is the symmetric best under both engines.

## LRT asym vs sym (df 3, boundary effects ignored)

| engine | best asym | best sym | 2ΔlogL | p | realdata2.md had |
|---|---|---|---|---|---|
| MASCOT | −281.478 (ans-root hub) | −288.531 | 14.1 | 0.0028 | 8.25 (p 0.041) |
| exact | −281.507 ± 2e-4 (ans-root hub) | −296.304 ± 0.65 | 29.6 ± 1.3 | 1.7e-6 (range 0.9e-6–3.1e-6) | 20.25 (upper bound, wrong basin) |

The earlier MASCOT LRT used a MASCOT asymmetric optimum that was not MASCOT's best. With the better optimum,
MASCOT also rejects symmetry clearly. The exact LRT is larger again because MASCOT's error at the symmetric hub
(+7.8) is much larger than at the asymmetric best (+0.03).

## What this changes in realdata2.md
- "MASCOT's 2nd = exact best, the ranking changes": **withdrawn.** The comparison mixed one refined exact optimum
  with unrefined ones, and MASCOT's own global optimum had been missed. After refinement both engines agree on the
  best peak, its root (ans) and its parameters (Ne within 2%, m12 within 3%, m20 0.57 vs 0.51).
- "Both engines reject an Anseriformes root": **withdrawn.** Both now prefer it.
- **Still true:** MASCOT's error is strongly basin-dependent (+0.03 / +1.5 / +7.2 / +7.8). The exact likelihood
  therefore re-orders and separates the *secondary* optima (gal root vs cha root vs mode A: within 0.63 under
  MASCOT, 4–10 apart exactly). And it makes the asymmetry test much more decisive.
- **Lesson for method:** a multistart of 10 random starts was not enough on this surface, even for MASCOT. Exact
  refinement from every MASCOT peak, and re-fitting MASCOT from the refined points, was needed to find the best
  basin.

## Limitations
- The ans-root hub exact maximum is found from two starts: −281.507 (converged) and −281.587 (still approaching it).
- Cha root: the MASCOT-Hessian quasi-Newton step was rejected at every step length, even though the exact gradient
  there is large (|g| = 3.4). So −292.24 is a lower bound of that basin, not its maximum. Even the model-predicted
  gain (2.9) would leave it 3.2 below the gal root, so the ranking is not affected, but this is a prediction, not a
  bound.
- Rates at the lower bound make the χ² LRT approximate. One fixed MCC tree. Each exact value is at eps 1e-6.
  Paired errors are ≤ 2e-3 except at the symmetric hub (0.65).

## Files
`bench/realdata2_sl/masco_refine_<peak>{.json,_fit.json,.log}` for peak = asym_modeA, asym_modeC, sym_hubcont,
sym_regular. `masco_pair_h2.json` holds the h/2 checks. `masco_mascot_asym_from_exactA.json` is the new MASCOT
optimum (copied to `masco_mascot_asym_hubans.json` as the Kaggle start). Tables: `bench/realdata2_sl_summarize.py`
→ `bench/realdata2_sisco_lisco.csv` (masco rows included).

**Final status 2026-10-11 22:40.**
- **Converged:** every masco value except the cha root.
- **Still a bound:** the cha root (≥ −292.239). Every quasi-Newton step was rejected, so it was not continued: it is
  6.1 below the gal root and 10.7 below the best peak, so no conclusion depends on it.
- **MASCOT max − exact max per peak:**
  - ans-root hub +0.03
  - gal root +1.61
  - cha root ≤ +7.20
  - sym hub +7.77
  - sym regular +7.31
- **Exact LRT:** 2Δ = 29.59 ± 1.30. The TT error is 2·(h/2 errors at both ends), dominated by the symmetric hub
  (0.65).

Cross-tree comparison: `realdata2_sisco_lisco.md`.
