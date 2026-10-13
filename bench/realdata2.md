# Real data 2: avian influenza H7 HA across bird hosts (133 tips, 3 demes) — exact (TT) vs MASCOT

**Bottom line.** On this dataset the exact likelihood *does* change substantive conclusions, unlike the H3N2 dataset
of `realdata.md`. Under the asymmetric model MASCOT and the exact likelihood pick different optima, and the two give
different stories about which host population is large and which direction the virus moves. The asymmetric-vs-symmetric
LRT is borderline under MASCOT (p = 0.04) and clear-cut under the exact likelihood (p = 0.0002). The error at fixed
parameters is +0.2 to +8.0 log units (MASCOT above exact), 20× larger than on H3N2.

The dataset is, however, *not* in the slow-migration regime of the simulations: at the exact asymmetric MLE the mean
migration rate over the mean pairwise coalescence rate is 0.49 (simulated "slow" = 0.1, "medium" = 1). It is a
few-tips-per-deme, strongly unbalanced dataset instead, and those are the features that seem to matter here.

## Data
AIV H7 HA, 133 sequences 2001–2012 from North America. Fixed tree = the MCC tree of Müller, Rasmussen & Stadler
(2017) `masco` analysis (CC BY 3.0); host labels = the `typeTraitSet` of the BASTA example XML
`AIV_HA_BASTA_host_asymmetric.xml` (GPL-3). No tree was re-estimated. Provenance, hashes and the conversion script
are in `../data/README.md` ("Real data 2").

Demes = bird orders: 0 Anseriformes (ducks/geese, 93 tips), 1 Charadriiformes (shorebirds, 32), 2 Galliformes
(poultry, 8). The BASTA grouping has 5 types; its `pas` type is 2 sanderlings (which are Charadriiformes) and its
`MexicoGal` type is 2 chickens of the 2012 Mexican outbreak (Galliformes), so we merged them into cha and gal
(`data/aivorder_masco_mcc.json`). The 5-deme version was dropped: MASCOT's optimum there depends on the start
(8 symmetric starts gave 6 different optima spanning 27 log units, 8 asymmetric starts gave 8 different ones) and
several optima put the root in a 2-tip deme, while one exact evaluation would cost hours (360k TT roundings, rank 42
at 20 lineages; see `realdata2/aivhost_tt_cost_probe.log`).

Peak concurrent lineages 46, so 3^46 ≈ 9 × 10^21 states: **dense is impossible here and TT is the only exact
engine**. This is the first real dataset in this project where that is true. Root height 12.3 y; 128 distinct
sampling times; Fitch parsimony needs 8 host changes.

## Method
Same as `realdata.md` (`bench/realdata_fit.py`, now deme-count-agnostic): asym = 6 backward rates + 3 Ne, sym = 3
rates + 3 Ne, log parameters, L-BFGS-B, bounds log ∈ [−12, 7]. Engines: MASCOT (DOP853, rtol 1e-10) and exact TT
(`tt/esco_tt.py`, eps 1e-6).

Because one exact evaluation takes 10–50 min, the exact MLE was not found by the same optimiser. Instead
`bench/realdata2_refine.py` runs quasi-Newton steps from a MASCOT optimum, using the MASCOT Hessian as the curvature
model and the exact gradient written as g_exact = g_MASCOT + ∇(logL_TT − logL_MASCOT), where the second term uses a
coarse step (0.1 in log space) because the difference surface is much flatter than the likelihood. Rates at the lower
bound were kept there and checked afterwards with a KKT probe (raise each to 0.02 and re-evaluate). This finds the
exact optimum in the basin of a MASCOT optimum; it does not search globally.

Local optima were mapped with MASCOT first (`bench/realdata_multistart.py`, 10 random starts per model), then each
distinct optimum was evaluated exactly.

**TT error.** Paired estimates at four points (eps 1e-6 vs the same eps with half the time step, and once vs
eps 1e-7):

| point | TT eps 1e-6 | h/2 | eps 1e-7 | error estimate |
|---|---|---|---|---|
| exact asym optimum (gal root) | −286.17996 | −286.17951 | −286.17915 | 8.1e-4 |
| MASCOT asym hub optimum | −290.30014 | −290.34570 | — | 4.6e-2 |
| exact sym hub optimum | −296.30429 | −296.95221 | — | 6.5e-1 |

The largest, 0.65, is at the hub point (a deme with Ne ≈ 0.10, i.e. very fast coalescence, which is the expensive
corner of the step rule). All three are far below the MASCOT−exact differences of 1.5–8.0 that the conclusions rest
on, but the hub value is big enough that the exact hub logL should be read as −296.3 ± 0.7, not to three decimals.

## Asymmetric model: the two engines rank the optima differently

| optimum (named by its root) | MASCOT logL | exact logL | MASCOT − exact | root P(ans/cha/gal), MASCOT | root, exact |
|---|---|---|---|---|---|
| **gal root** (mode B) | −284.569 | **−286.311** | +1.74 | 0.000 / 0.001 / 0.999 | 0.000 / 0.001 / 0.999 |
| hub (mode A) | **−284.408** | −290.300 | +5.89 | 0.250 / 0.000 / 0.750 | 0.594 / 0.000 / 0.406 |
| cha root (mode C) | −285.041 | −292.239 | +7.20 | 0.000 / 0.977 / 0.023 | 0.000 / 0.974 / 0.026 |
| ans root (m02 → 0) | −295.419 | −295.661 | +0.24 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 |
| exact optimum near mode B | −284.646 | **−286.180** | +1.47 | — | 0.000 / 0.001 / 0.999 |

MASCOT's three best optima lie within 0.63 log units of each other, so MASCOT cannot separate them; its best is the
hub. Under the exact likelihood they are 4.1 and 6.1 log units apart and the ordering changes: the best is the gal-root
mode. The reason is that MASCOT's error is not constant across the parameter space — it is +1.7 in the gal-root basin
and +5.9 to +7.2 in the hub and cha-root basins. **This is the mechanism that the simulations predicted (error growing
with the amount of structure in the lineage-state distribution) acting on a real dataset.**

The two modes say different things biologically:

| | MASCOT best (hub) | exact best (gal root) |
|---|---|---|
| Ne ans / cha / gal | 5.28 / 2.93 / 0.27 | 2.17 / 0.93 / 3.32 |
| dominant rates (per year, backward) | m12 = 15.9 (cha←gal), m20 = 0.44 | m02 = 1.38 (ans←gal), m12 = 0.39 |
| story | poultry is a tiny population exchanging lineages with shorebirds extremely fast | poultry is the largest population and seeds ducks |
| root | gal 0.75, ans 0.25 | gal 0.999 |

Both engines agree that the root is **not** Anseriformes: forcing the ans-root mode (m02 → 0) costs 10.9 (MASCOT) or
9.5 (exact) log units. That is the one place where this analysis disagrees with the published analyses of these data,
which put the root in Anseriformes (Roberts et al. 2025; Müller et al. 2017 for the location version). Those are
Bayesian analyses with priors on a 5-deme model and they integrate over trees; ours is a 3-deme ML fit on one fixed
MCC tree with 8 and 32 tips in the two non-duck demes, so this is not a contradiction of their result, only a warning
that the ML surface on a fixed tree is not where their conclusion comes from.

At the exact optimum near mode B the parameter shift from the MASCOT optimum in the same basin is small:
m02 +2.1%, m12 −3.8%, m21 −31.1%, Ne +0.6 to +3.9%. So *within* a basin MASCOT's parameter estimates are fine; what
changes is *which basin wins*.

KKT probe at the exact optimum (each zero rate raised to 0.02): m01 −0.33, m10 −0.06, m20 −0.39, i.e. all three stay
at the bound.

## Symmetric model and the LRT

| optimum | MASCOT logL | exact logL | MASCOT − exact | root, MASCOT | root, exact |
|---|---|---|---|---|---|
| hub (gal Ne 0.10, cha–gal 22/y) | **−288.531** | −296.323 | +7.79 | 0.070 / 0.020 / 0.910 | 0.093 / 0.022 / 0.885 |
| regular (all rates 0.16–0.34) | −294.624 | −302.617 | +7.99 | 0.166 / 0.738 / 0.096 | 0.263 / 0.626 / 0.111 |
| exact optimum near the hub | −288.549 | −296.304 | +7.76 | — | 0.100 / 0.022 / 0.878 |

The symmetric best is the hub under both engines (8/10 MASCOT starts land on the regular optimum, but the hub is 6.1
log units better). The exact refinement from the regular optimum was stopped after one iteration (local machine
load), so −302.617 is an upper bound for that basin, not its optimum; it does not affect the comparison since the hub
wins by 6.3.

LRT asym vs sym, df = 3 (boundary effects ignored, so p-values are indicative only):

| engine | best asym | best sym | 2ΔlogL | p | verdict |
|---|---|---|---|---|---|
| MASCOT | −284.408 | −288.531 | 8.25 | 0.041 | just significant |
| exact | −286.180 | −296.304 | 20.25 | 0.00015 | strongly significant |

So the LRT verdict is formally the same (both reject symmetry at 0.05) but the strength changes by three orders of
magnitude in p. MASCOT would be reported as "borderline"; the exact likelihood as "decisive". The whole gap comes from
the symmetric fit, where MASCOT's error (+7.8) is much larger than at the exact asymmetric optimum (+1.5).

## What changed and what did not
- **Changed:** the ML ranking of the asymmetric optima (hub → gal root), hence the estimated source population, the
  deme with the largest Ne, and the fastest migration route; the strength of the asymmetry LRT (p 0.04 → 0.0002).
- **Not changed:** the root deme of the best MASCOT optimum vs the exact optimum *within* a basin (both gal 0.999 in
  mode B, both cha 0.97 in mode C); the rejection of an Anseriformes root; the fact that the symmetric best is the hub;
  the parameter values within a basin (≤ 4%, except the smallest rate at −31%).
- **Direction of the error:** MASCOT above exact at all six points, as on H3N2 (+0.08 to +0.38 there) but 20× larger.
  In the simulations the slow-regime errors were mostly *negative*; the sign here matches the medium/fast regime while
  the magnitude matches the slow one.

## Limitations
- One fixed MCC tree, no tree uncertainty, no posterior-tree replicates (one exact evaluation is 10–50 min, and the
  posterior trees of this analysis have 40–46 peak lineages too).
- The exact optimum is found only within the basin of a MASCOT optimum (quasi-Newton from there). Four basins were
  located with MASCOT and all four were evaluated exactly, but an exact-only basin would have been missed.
- The 3-deme merge is ours, not the original grouping; the published analyses use 5 host types or 7 locations. We did
  not reproduce their Bayesian setup.
- `realdata2_refine.py` uses the MASCOT Hessian as its curvature model; the stopping rule is a predicted gain of
  < 0.01, and the hub run was stopped by hand at a predicted gain of 0.023, so the reported hub logL may be ~0.02
  below its basin's optimum (within the 0.65 TT error estimate there anyway).
- Rates at the lower bound make the χ² LRT approximate, as in `realdata.md`.

## Files
`bench/realdata2.csv` (table), `bench/realdata2/` (all fits, evals, logs), `bench/realdata2_summarize.py` (rebuilds
the CSV and prints the summary), `bench/realdata2_eval.py` (MASCOT + TT at fixed parameters, `1e-6h2` = half step),
`bench/realdata2_refine.py` (exact quasi-Newton), `data/convert_aiv.py`. All numbers above are from the local Mac
(OpenBLAS venv, `tt/` as of this session); nothing was run on Kaggle for this dataset.
