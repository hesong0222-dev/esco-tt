# MultiTypeTree (MTT) vs exact vs MASCOT posteriors on fixed trees

**Question.** On fixed trees where the exact ESCO likelihood is cheap, does MTT's posterior over (migration rates, Ne)
match the exact posterior? How does MASCOT's posterior compare, and what does each method cost (ESS per CPU-hour)?

**Status (2026-10-09 17:00).** Design, convention mapping, XMLs, kernel and analysis scripts are done. The local
exact (importance-sampling) and MASCOT posteriors are in §4. The Kaggle smoke kernel passed: MTT 8.3.0 is installed,
and the convention check gives max |Δ| = 7e-15 on 202 logged states. The full kernel `kaggle/esco-mtt-full` has
been RUNNING since 16:37 KST (~8–9 h), and its results will be collected automatically (§5). **The MTT result
(the main question) is not in yet.**

## 1. Design

| item | choice |
|---|---|
| trees (fixed topology and node times, from `sim/data/`) | `d2_k16_slow_r1` (peak 10 lineages), `d2_k16_medium_r0` (11), `d3_k10_slow_r1` (7), `d3_k10_medium_r2` (8) |
| why these | dense likelihood takes 0.04–0.1 s per evaluation on an idle M4 (`bench/tt_vs_dense.csv`: 0.1–1.0 s under load), so MCMC with the exact likelihood is feasible. MASCOT − exact at the generating values: +0.119, +0.305, −0.155, +0.405 |
| model | the paper's model: per-deme Ne, full asymmetric backward migration matrix. d = 2: 4 parameters, d = 3: 9 parameters |
| priors (all methods) | Ne_a ~ LogNormal(0, 1); M_ab ~ LogNormal(−1, 1.5) (meanlog, sdlog; median 0.37, 95% 0.02–7) |
| start | all chains start at the generating parameters |
| exact posterior | (a) local: MASCOT adaptive-Metropolis chain, then importance reweighting of 1500 thinned draws with w = exp(ℓ_exact − ℓ_MASCOT) (same prior, so this targets the exact posterior); (b) Kaggle: 2 adaptive-Metropolis chains with the dense likelihood, 1.25 CPU-h each |
| MASCOT posterior | adaptive Metropolis with `approx/mascot.py` (DOP853, rtol 1e-10): local chain, plus 1 Kaggle chain of 0.5 CPU-h (same hardware as MTT, for ESS/h) |
| MTT | BEAST 2.7.7 + MultiTypeTree 8.3.0, 2 chains per tree, 2.5 CPU-h each (chain length set from an in-kernel pilot), 10% burn-in |
| MTT operators | migration history only: NodeRetype (w 10), TypeMergeSplit incl. root (w 10), TypePairBirthDeath (w 10), TypeBirthDeath (w 10, d = 3 only); parameters: ScaleOperator on rateMatrix (w 3d(d−1)) and popSizes (w 3d), UpDown(popSizes up, rateMatrix down; w 2). Excluded: MultiTypeUniform, NodeShiftRetype, MultiTypeTreeScale, TypedSubtreeExchange, TypedWilsonBalding (they change node heights or topology). Initial history: `MultiTypeTreeFromUntypedNewick` (simulated on the fixed tree with the generating rates) |
| adaptive Metropolis | Haario-type random walk in log space; covariance adapts during the first 30% (iterations or wall time), which is discarded; the rest is a fixed-kernel Metropolis chain |
| reported | posterior mean and 95% interval per parameter; KS distance and overlap coefficient to the exact posterior; KS_ref = 1.36·sqrt(1/ESS_method + 1/ESS_exact) (what Monte Carlo error alone would give); ESS (Geyer initial monotone sequence, same estimator for all methods); ESS per CPU-hour (sum over chains / total CPU time incl. burn-in); MTT split-R-hat across the 2 chains, target ESS ≥ 200 |

## 2. Convention mapping (ours ↔ MTT 8.3.0 source, tag v8.3.0)

| quantity | ours (`sim/README.md`, `dense/`, `approx/`) | MTT | mapping |
|---|---|---|---|
| migration | `M[a][b]`: backward-time rate at which one lineage in deme a jumps to b | `SCMigrationModel.getBackwardRate(i,j)` = `rateMatrixScaleFactor`·`rateMatrix[offset(i,j)]` when `useForwardMigrationRateMatrix` = false (the default); `offset = i(n−1) + j'` with `j' = j − 1` if `j > i` for a matrix of dimension n(n−1) | `rateMatrix` = `M01 M02 M10 M12 M20 M21` (a-major off-diagonal), no scale factor, forward-matrix flag left false |
| where M enters the density | master equation `Q = M − diag(rowsum M)` acting per lineage | `StructuredCoalescentTreeDensity`: interval rate Σ_c k_c Σ_{c'} m(c,c'); a migration event on a branch contributes log m(type below, type above) | identical (type below = tip-ward = start of the backward jump) |
| coalescence | each pair in deme a coalesces at rate 1/Ne[a] | interval rate Σ_c k_c(k_c−1)/(2 N_c), event factor 1/N_c, `N_c = popSizesScaleFactor·popSizes[c]` | `popSizes` = Ne, no scale factor, no ploidy factor |
| time | event times `t` (present → past), rates per unit of t | node heights from Newick branch lengths (`TreeParser`, `adjustTipHeights=false`), youngest tip at height 0 | branch length = t(parent) − t(child); all JSON trees have a tip at t = 0; coalescence times reproduced to 7e-15 |
| root | no root term, no conditioning | no root term; root type is sampled (`TypeMergeSplit includeRoot`, `NodeRetype` on the root) | MTT's density summed over histories and root type = ESCO likelihood |
| tip demes | `deme` field | `TraitSet` type = `d<deme>`; `TypeSet value="d0,d1[,d2]"` (sorted set → index = deme) | identical indices |
| priors | Normal(μ, s) on log x | `LogNormalDistributionModel M=μ S=s meanInRealSpace=false` | identical |
| logged | — | `popSizes.k`, `rateMatrix.k` (k = 1-based offset), `treePrior` = complete-data density | analysis reads these columns |

Check: `bench/mtt/typed_density.py` recomputes MTT's complete-data density with our convention. It gives −16.5283137 on
MTT's own hand-computed test tree (`StructuredCoalescentTreeDensity.main`, expected −16.52831). The smoke kernel
recomputes it on every logged typed tree and compares with the logged `treePrior` (pass: max |Δ| < 1e-6). The full
kernel repeats the check for every chain.

## 3. Files

- `bench/mtt/common.py` — parameterisation, priors, adaptive Metropolis, ESS, R-hat, KS/overlap.
- `bench/mtt/make_xml.py` → `bench/mtt/xml/<tree>.xml` (placeholders CHAINLENGTH, LOGEVERY, TREEEVERY).
- `bench/mtt/run_local.py`, `bench/mtt/run_local_all.sh` — local MASCOT chain + IS-exact → `bench/mtt/out/<tree>_local.npz`.
- `bench/mtt/typed_density.py` — convention check on MTT logs.
- `bench/mtt/kernel_main.py` — Kaggle driver (smoke | full); `kaggle/mtt/build.py` bundles it into `run.py`.
- `kaggle/mtt/push.sh`, `autopush.sh` (nohup, log `kaggle/mtt/autopush.log`), `pull.sh` (download + `bench/mtt/analyze.py`).
- `bench/mtt/analyze.py` → `bench/mtt_compare.csv`, `bench/mtt/results.md`.

## 4. Local results (exact by importance sampling vs MASCOT)

Run on the Mac with OpenBLAS `.venv`, BLAS threads 1, nice 10, one process, under load avg 20–100. Commands:
`bench/mtt/run_local_all.sh` (log `bench/mtt/out/local.log`), then `bench/mtt/analyze.py`. The full per-parameter
tables are in `bench/mtt/results.md` and `bench/mtt_compare.csv`.

| tree | MASCOT chain (it, s, accept) | IS weight-ESS / 1500 | sd(log w) | max KS (MASCOT vs exact) | min overlap | largest posterior-mean shift MASCOT/exact − 1 | min MASCOT ESS |
|---|---|---|---|---|---|---|---|
| d2_k16_slow_r1 | 12000, 598, 0.30 | 1468 | 0.18 | 0.039 | 0.948 | M10 +7.8% | 451 |
| d2_k16_medium_r0 | 12000, 520, 0.30 | 1457 | 0.20 | 0.053 | 0.944 | M10 −4.5% | 442 |
| d3_k10_slow_r1 | 15000, 780, 0.26 | 975 | 0.48 | 0.095 | 0.901 | Ne0 +17.1% (4.41 vs 3.77) | 152 |
| d3_k10_medium_r2 | 15000, 402, 0.23 | 1406 | 0.28 | 0.053 | 0.941 | M01 −12.3% | 193 |

Reading:
- On these small trees, MASCOT's posterior is close to the exact one. sd(log w) is the spread of
  ℓ_exact − ℓ_MASCOT over the posterior. It is 0.18–0.48, so most of MASCOT's error at fixed parameters
  (+0.12 to +0.41 at the truth) is a near-constant offset that cancels in the posterior.
- The largest effect is in the three-deme slow tree. It has the smallest weight-ESS (975) and the largest KS (0.095,
  M02). There, MASCOT puts the Ne of deme 0 17% higher in posterior mean. The 95% intervals still overlap almost
  entirely ([0.44, 14.2] vs [0.36, 12.2]). This matches the direction of the paper's MLE result: in slow d ≥ 3 the
  approximation moves estimates, but by less than the posterior width.
- The posteriors are wide. d = 3 with 10 tips is dominated by the prior for most rates (intervals span two orders of
  magnitude). So this is a test of *correctness* of the posterior, not of power.
- Caveats:
  - The IS proposal is the MASCOT posterior itself. KS and overlap between "exact_IS" and "mascot_local" therefore
    measure only the reweighting. The Monte Carlo errors are correlated, so KS_ref in `results.md` is conservative.
  - IS cannot find exact-posterior mass where MASCOT has none. The independent exact chains in the Kaggle kernel
    check this.
  - The overlap coefficient (60-bin histograms, ~1500 draws) is about 0.95 even for identical distributions. Use KS.
  - ESS/CPU-h here (700–3700 for MASCOT) is Mac-under-load time and is not comparable to Kaggle times. Use the
    Kaggle numbers for cost.

## 5. Kaggle status and how to collect

- Smoke `kaggle/esco-mtt-smoke`: COMPLETE, `SMOKE_OK`. Output is in `kaggle/out/mtt-smoke-kaggle/`.
  - kaggle allows internet. BEAST 2.7.7 (bundled JRE) and MTT 8.3.0 are installed; `packagemanager -list` shows
    `MultiTypeTree | 8.3.0 | 8.3.0`.
  - Two 20000-step chains on d2_k16_slow_r1 ran at about 1 min per 10⁶ steps (≈16k steps/s).
  - Convention check: 101 + 101 logged typed trees, max |ours − MTT treePrior| = 7.1e-15, coalescence times
    within 1.6e-15.
  - Operator acceptance: NodeRetype 0.47, TypeMergeSplit 0.006, TypePairBirthDeath 0.009, scale moves 0.73–0.83
    (auto-tuned in the long runs).
  - Kaggle dense likelihood ≈ 0.2 s per evaluation (exact AM: 200 iterations in 42 s); MASCOT ≈ 0.045 s.
- Full `kaggle/esco-mtt-full`: pushed 16:37 KST, RUNNING. Plan:
  - a pilot per tree (1000 vs 200000 steps), then 8 MTT chains (4 trees × 2) of 2.5 CPU-h each;
  - 8 exact chains of 1.25 CPU-h; 4 MASCOT chains of 0.5 CPU-h;
  - all in a 4-worker pool, ≈ 8.5 h in total;
  - every MTT chain repeats the convention check (`full_summary.json` → `conv`).
- Collection runs automatically: `kaggle/mtt/wait_pull.sh` (nohup) waits for the kernel to end, then runs
  `kaggle/mtt/pull.sh` and sends a notification. By hand: `bash kaggle/mtt/pull.sh`
  → `kaggle/out/mtt-full-kaggle/` → `bench/mtt/analyze.py` → `bench/mtt_compare.csv`, `bench/mtt/results.md`
  (reference becomes the pooled exact chains).
- After collection, check:
  - MTT R-hat < 1.01 and ESS ≥ 200 per parameter (sum over 2 chains);
  - KS(MTT vs exact) against KS_ref;
  - ESS per CPU-h for MTT, exact and MASCOT on the same hardware;
  - `conv.max_abs_density_diff` < 1e-6 for each chain.
  If an MTT chain falls short of ESS 200, rerun with a longer budget: `B_MTT` in `bench/mtt/kernel_main.py`, then
  `kaggle/mtt/push.sh <acct> full` after renaming the kernel id.
