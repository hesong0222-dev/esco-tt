# Real data: exact structured coalescent (ESCO) vs MASCOT on the MASCOT-tutorial H3N2 tree

**Bottom line:** on this public dataset, the exact likelihood does **not** change any substantive conclusion compared with MASCOT.
The source deme is the same (Hong Kong at the root under the asymmetric model, P ≈ 1.00 under both). The LRT for
asymmetric vs symmetric migration gives the same answer (not significant under either). The same three migration
routes are estimated as zero. The asymmetric migration-rate MLEs move by 5–10%. The one change in ranking happens in
the symmetric model, where the most likely root deme switches from Hong Kong (0.49 vs NZ 0.47) to New Zealand (0.52 vs
HK 0.46). Both splits are close to 50/50, so I don't count this as a real change of conclusion. MASCOT's
log-likelihood error at fixed parameters is +0.08 to +0.38 (it overestimates). The 4 posterior trees show the same
picture. So this is a negative but useful result: it is the first real-data check of MASCOT against an exact,
error-controlled likelihood, and on a 3-deme, 24-tip dataset in the "medium/fast migration" regime MASCOT is adequate.
In simulations (bench/dense_vs_mascot.md) the large MASCOT errors appear only for slow migration and d ≥ 3, and this
dataset is not in that regime.

## Data
Taming the BEAST MASCOT tutorial (CC BY 4.0): 24 H3N2 HA sequences from 2000–2001, demes Hong Kong (HK, 10 tips),
New York (NY, 6), New Zealand (NZ, 8). The fixed tree is the tutorial's precooked MCC tree (median heights). The
robustness check also uses 4 of the tutorial's posterior trees. Provenance, hashes and conversion are in `../data/README.md`.
Peak concurrent lineages on the MCC tree = 12, so 3^12 = 531k states and the **dense exact** likelihood is cheap (≈3.7 s
per evaluation on a Kaggle CPU). Dense is the reference everywhere. TT is only a cross-check.
Convention check: our MASCOT reproduces BEAST2-MASCOT's logged density on posterior samples within 0.003–0.46 (see
data/README.md). Transposed M or doubled Ne miss by much more.

## Method
- Models: **asym** = 6 backward migration rates m_ab (a→b backward in time; m10 = NY lineages coming from HK) + 3 Ne
  (9 parameters). **sym** = 3 symmetric rates + 3 Ne (6 parameters).
- Maximum likelihood in log-parameters with L-BFGS-B and forward-difference gradients (h = 1e-5, evaluated in parallel).
  Bounds are log ∈ [−12, 7], so a rate at e^−12 ≈ 6e-6 means "zero" and Ne at e^7 = 1097 means "no coalescence there".
- Engines: MASCOT (`approx/mascot.py`, DOP853 rtol 1e-10) and exact dense (`dense/esco_dense.py`). Root-deme
  probabilities come from the same recursions (`bench/realdata_fit.py: dense_root / mascot_root`).
- Local optima: MASCOT was fitted from 12 (asym) or 8 (sym) random starts on the MCC tree. The best optimum was found 6/12
  and 7/8 times, and it equals the fit reported below. Other optima: −31.845, −31.861 and a "root in NY" mode at −33.403.
  Dense asym was fitted from 2 starts (MASCOT optimum and dense-sym optimum), which converged to the same point
  (−31.647356 both). On posterior trees, asym was fitted from 2 starts (MASCOT-MCC optimum and the tree's own sym
  optimum) and the best is kept.
- TT cross-check at every optimum (eps 1e-6, 1e-7, and 1e-7 with h/2): |TT − dense| ≤ 8.6e-5 (1e-6) and ≤ 1.3e-5
  (1e-7). The h/2 run agrees to ≤ 3.8e-5. Max TT rank is 24 (1e-7). TT root probabilities match dense to ~1e-6.
- All fits ran on Kaggle CPU (Linux, OpenBLAS): kernels kaggle/esco-realdata-k1…k5. Raw JSON is in kaggle/out/realdata_k*/.
  Table: `bench/realdata.csv`. Reproduce the tables with `.venv/bin/python bench/realdata_summarize.py`.

## MCC tree: ML estimates
Units are per year (rates) and years (Ne, coalescent rate 1/Ne per pair). "0" means at the lower bound (< 1e-5).

| model | engine | logL at own MLE | m01 HK←NY | m02 HK←NZ | m10 NY←HK | m12 NY←NZ | m20 NZ←HK | m21 NZ←NY | Ne HK | Ne NY | Ne NZ | root P(HK/NY/NZ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| asym | MASCOT | −31.5406 | 0.649 | 0 | 2.985 | 0 | 0 | 1.212 | 0.838 | ∞ (bound) | 0.327 | 1.00 / 0.00 / 0.00 |
| asym | exact  | −31.6474 | 0.615 | 0 | 3.220 | 0 | 0 | 1.099 | 0.876 | ∞ (bound) | 0.368 | 1.00 / 0.00 / 0.00 |
| sym  | MASCOT | −33.6228 | 1.809 (HK–NY) | 0 (HK–NZ) | = | 0.508 (NY–NZ) | = | = | 0.414 | 8.15 | 0.322 | 0.49 / 0.04 / **0.47** |
| sym  | exact  | −33.9618 | 1.964 | 0 | = | 0.557 | = | = | 0.399 | 17.7 | 0.346 | 0.46 / 0.02 / **0.52** |

Relative change of the exact MLE vs MASCOT for the nonzero asym rates: m10 +7.9%, m21 −9.3%, m01 −5.3%; Ne HK +4.5%, Ne NZ +12.5%.
Sym: m01 +8.5%, m12 +9.7%. Ne NY is effectively unidentified (it runs to the bound in asym, 8 vs 18 in sym). Both
engines say the same thing: there is almost no coalescence in New York, so NY is a sink that is seeded from elsewhere.

**MASCOT error at fixed parameters** (MASCOT − exact log-likelihood at the same parameter values):
+0.135 at the MASCOT asym MLE, +0.082 at the exact asym MLE, +0.379 at the MASCOT sym MLE, +0.292 at the exact sym MLE.
MASCOT overestimates here. That is the opposite sign to the slow-migration simulations, and of the size we saw in the
"medium" regime.

**Root at the same parameters:** at the MASCOT sym MLE, MASCOT gives a root of HK 0.49 / NZ 0.47 and exact gives HK 0.45 /
NZ 0.51. The ranking flip comes from MASCOT's root marginal itself, not from the parameter shift.

## Model comparison (LRT asym vs sym, df = 3; boundary effects ignored, so the p-values are only indicative)

| tree | MASCOT 2ΔlogL (p) | exact 2ΔlogL (p) | most likely root, sym: MASCOT → exact | asym: MASCOT → exact |
|---|---|---|---|---|
| MCC | 4.17 (0.24) | 4.63 (0.20) | HK 0.49 → **NZ 0.52** | HK 1.00 → HK 1.00 |
| post150 | 0.48 (0.92) | 0.50 (0.92) | HK 0.63 → HK 0.66 | HK 0.86 → HK 0.91 |
| post205 | 2.98 (0.39) | 2.31 (0.51) | HK 0.54 → HK 0.51 | HK 0.76 → HK 0.74 |
| post280 | 6.61 (0.085) | 7.42 (0.060) | HK 0.63 → HK 0.55 | HK 1.00 → HK 1.00 |
| post390 | 1.58 (0.66) | 1.18 (0.76) | HK 0.72 → HK 0.65 | HK 0.97 → HK 1.00 |

The 4 posterior trees were chosen from samples 150–400 of 401 (burn-in excluded), evenly spaced among those with peak
lineages ≤ 12. That restriction was only to keep dense cheap, and it selects 19 of 51 thinned samples. The LRT verdict is
the same under both engines on all 5 trees. post280 comes closest to the 0.05 line (0.085 vs 0.060), and neither
crosses it. On the smallest tree (post205), the exact m10 is 3.35 vs MASCOT 2.35 (+42%), but Ne NY goes to the bound
under exact (12 under MASCOT). This is the poorly identified, few-lineage end, and logL differs by ≈ 1.2 between the
engines there.

## Profile of m10 (NY←HK backward rate, i.e. HK→NY forward seeding), asym model, MCC tree
ΔlogL = profile logL − the engine's own maximum. Each point is the best of two warm-started chains: one from the main
(HK-root) optimum and one from MASCOT's NY-root local optimum. With a single chain the low-m10 end was badly
underestimated (MASCOT −6.2 at m10 = 0.1), so the second chain is necessary.

| m10 | MASCOT ΔlogL (root) | exact ΔlogL (root) |
|---|---|---|
| 0.1 | −1.92 (NY 1.00) | −1.74 (NY 1.00) |
| 0.3 | −2.02 (NY 1.00) | −1.84 (NY 1.00) |
| 0.6 | −1.89 (HK 0.97) | −1.96 (NY 0.99) |
| 1.0 | −0.56 (HK 0.99) | −0.96 (HK 0.93) |
| 1.5 | −0.32 | −0.52 |
| 2.0 | −0.21 | −0.34 |
| MLE | 0 at 2.99 | 0 at 3.22 |
| 4.0 | −0.13 | −0.06 |
| 6.0 | −0.47 | −0.30 |
| 10  | −0.92 | −0.70 |

Under both engines the m10 profile has two modes. The main mode (root HK, HK seeds NY) peaks at m10 ≈ 3. The
alternative mode (m10 ≈ 0, root NY, NY seeds HK with m01 ≈ 10) is only ≈ 1.7–2.0 log units worse, right at the χ²₁
95% threshold (1.92). The exact likelihood favours the NY-root mode slightly more (−1.74 vs −1.92), which puts it
inside the approximate 95% region, while under MASCOT it sits on the boundary. So the honest reading is the same under
both: the data do not firmly exclude "NY is the source". The exact model makes that ambiguity a little more visible,
but the effect is small (≈ 0.2 log units). The upper side (m10 up to ≥ 10 within ΔlogL > −1) is equally flat in both.

## Limitations
- One small dataset (24 tips, 3 demes, peak 12 lineages), so dense suffices and the TT itself is not needed here.
  The TT advantage is for d^k ≫ 1e8. Finding a real dataset in the slow-migration, d ≥ 4, high-peak regime where
  MASCOT was badly off in simulation remains open.
- The trees are fixed (MCC + 4 posterior samples). Tree uncertainty was not integrated over, and no joint tree +
  parameter inference was done.
- Several MLEs sit on parameter bounds (zero rates, infinite Ne NY), so the χ² LRT p-values are approximate.
- MASCOT here always means our converged solution (approx/mascot.py, DOP853, rtol 1e-10). BEAST 2 MASCOT's logged
  density differs from it because of the package's integrator: bench/mascot_vs_beast.md, CompEvol/Mascot#27.
