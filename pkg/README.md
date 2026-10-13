# escott

**Exact structured-coalescent likelihoods for a fixed time tree, and a way to check MASCOT on your own data.**

MASCOT (Müller, Rasmussen & Stadler 2018) makes the structured coalescent fast by approximating it. escott computes
the same likelihood without that approximation: with the full joint vector of lineage locations when it is small,
and with tensor trains (TT) when it is not. The point is not to replace MASCOT, but to tell you whether MASCOT is
adequate for *your* tree and parameters, and what changes when it is not.

| command | question it answers | typical cost (H3N2 example, 24 tips, 3 demes) |
|---|---|---|
| `escott estimate` | Can I afford an exact evaluation on this tree? Which engine, eps, how long? | 1 s |
| `escott check` | How far is MASCOT from the exact likelihood at these parameters? | 10 s |
| `escott reweight` | Does my BEAST 2 MASCOT posterior change under the exact likelihood? | about 1 h (all 361 states, 3 processes) |
| `escott fit` | Where are the maximum-likelihood estimates under both models, and do they differ? | 6–10 min (3 processes) |
| `escott compare` | Does the symmetric-vs-asymmetric migration test give the same answer? | 27 min (3 processes) |

Times are from an Apple M4 laptop shared with other work; see [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Install

```
pip install "escott @ git+<repository URL>"     # or, from a checkout: pip install -e .
escott --version
```

Python ≥ 3.10, numpy and scipy, nothing else.

**Apple Silicon:** use numpy/scipy builds linked to OpenBLAS, not Apple Accelerate. With Accelerate's LAPACK we saw
heap corruption in long tensor-train runs: segmentation faults and, worse, silently wrong log-likelihoods. escott
prints a warning at start-up when it detects Accelerate. Check with
`python -c "import numpy; numpy.show_config()"` (look for `lapack: scipy-openblas`). For Python 3.13 the
`numpy-2.5.3-cp313-cp313-macosx_11_0_arm64` wheel is an OpenBLAS build, while the newer `macosx_14_0` wheel links
Accelerate; one way to get the OpenBLAS wheels is

```
pip download numpy==2.5.3 --only-binary=:all: --platform macosx_11_0_arm64 --python-version 3.13 --no-deps -d wheels
pip download scipy==1.18.1 --only-binary=:all: --platform macosx_12_0_arm64 --python-version 3.13 --no-deps -d wheels
pip install wheels/*.whl && pip install -e .
```

Linux wheels are not affected. Run single-threaded (`export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) and use
`--jobs` for parallelism instead; escott's worker processes are single-threaded by default.

## Three-minute tutorial: H3N2

`examples/` holds the Taming the BEAST MASCOT tutorial data (24 H3N2 sequences from Hong Kong, New York and New
Zealand; CC BY 4.0, see `examples/LICENSE-H3N2-data.txt`): the MCC tree `H3N2.mcc.tree`, the posterior trees
`H3N2.trees` and trace log `H3N2.log` of the tutorial's MASCOT run, a tip-location CSV and a parameter file (the exact
maximum-likelihood estimate on the MCC tree). [docs/TUTORIAL.md](docs/TUTORIAL.md) walks through the same steps with
complete output.

**1. Is an exact evaluation affordable?**

```
$ escott estimate examples/H3N2.mcc.tree --params examples/params_h3n2.json
state space: d^k = 3^12 = 531,441 joint deme states
full-vector (dense) engine: feasible (limit 4,782,969 states); about 34 MB per process, about 4.0 s per evaluation ...
...
recommended eps: 1e-06 (largest of 1e-6, 1e-7 with a bound <= 0.02; do not use 1e-5 for inference)
```

k is the largest number of lineages present at once. Up to 3^14 joint states escott uses the full vector, which is
exact to floating point. Beyond that it uses the tensor train, and `estimate` predicts its rank and run time from
benchmark runs.

**2. How far is MASCOT from the exact likelihood here?**

```
$ escott check examples/H3N2.mcc.tree --locations examples/tips_h3n2.csv --params examples/params_h3n2.json
exact  log L = -31.647356   (full vector, floating-point accuracy; 6.1 s)
MASCOT log L = -31.565174   (MASCOT ODE solved to rtol 1e-10; 0.1 s)
MASCOT - exact = +0.082181

MASCOT is 0.0822 log units higher than the exact likelihood here; ... The gap is small; MASCOT is very likely adequate
for this tree. ...
```

`--log run.log` takes the parameters from a row of a BEAST 2 MASCOT log instead (`--log-row`, default the last).

**3. Does the posterior change?** Every logged state of the BEAST run (tree and parameters) is evaluated under both
models; per-state results are written to a CSV as they finish, and `--resume` continues an interrupted run.

```
$ escott reweight examples/H3N2.trees examples/H3N2.log --jobs 3
states: 401 in the log; burn-in 40 dropped; thin 1 -> 361 used (exact engine: 348 dense, 13 tensor train)
weights = exact likelihood / the chain's target density (log column 'Mascot')
log-weight l_exact - l_denominator: median -0.448, range -3.559 .. +0.549, SD 0.533
weight ESS: 309.1 of 361 states (86%); largest normalised weight 0.008
...
parameter                          sample mean        sample 95% HPD  exact mean         exact 95% HPD   shift shift/SD
m.Hong_Kong_to_New_Zealand              0.5088       0.000242 - 1.67      0.4679        0.000242 - 1.5   -8.0%   -0.070
Tree.height                              3.609           2.68 - 4.56       3.664           2.63 - 4.63   +1.5%   +0.103
...
root deme probability (posterior mean of the per-state root distribution):
                                           Hong_Kong    New_York New_Zealand
  BEAST sample, MASCOT root                    0.503       0.250       0.247
  exact posterior, exact root                  0.531       0.263       0.206

The weights are even: the exact posterior is well represented by the BEAST sample, and the reweighted summaries can
be trusted within the support caveat below.
```

The log-weights are nearly a constant offset (SD 0.53), so the weights are even and the posterior barely moves: the
means shift by at most 8%, about 0.1 posterior SD. `--thin N` uses every N-th state for a quicker look.

**4. Do the point estimates and the model test change?**

```
$ escott fit examples/H3N2.mcc.tree --locations examples/tips_h3n2.csv --model asym --jobs 3
MASCOT multistart: 10 starts -> 5 distinct optima; best reached by 5 start(s)
   #      origin  MASCOT opt.     exact at    exact max    error        exact root      MASCOT root  status
                               MASCOT opt.
   1  multistart     -31.5406     -31.6757     -31.6474        0    1.00/0.00/0.00   1.00/0.00/0.00  converged
   2  multistart     -31.8613     -32.2922     -32.2761        0    1.00/0.00/0.00   1.00/0.00/0.00  lower bound (...)
...
cross-check: re-fitting MASCOT from each refined exact optimum found no better MASCOT optimum.
best exact fit: log L = -31.6474 (error 0; converged); MASCOT at the same parameters: -31.5654
  parameter                           MASCOT MLE   exact MLE   change
  m.New_York_to_Hong_Kong                  2.985       3.223    +7.9%
  Ne.New_Zealand                          0.3273      0.3682   +12.5%
...

$ escott compare examples/H3N2.mcc.tree --locations examples/tips_h3n2.csv --jobs 3
...
  engine      best asym     best sym          2 Delta                      p
  MASCOT       -31.5406     -33.6228            4.165                  0.244
  exact        -31.6474     -33.9619            4.629                  0.201
MASCOT does not reject symmetry at 5%; the exact likelihood does not reject symmetry at 5%.
```

On this dataset the exact likelihood changes no conclusion: the estimates move by 4–13%, the root of the asymmetric
model stays in Hong Kong, and the model test gives the same answer.

## When to trust the results, and when not

**`check`.** With the full vector the exact value is exact to floating point. With the tensor train escott reports an
error estimate: the larger of (a) the difference to a paired run at eps/10 and half the time step and (b) the
empirical bound 0.81 · eps · N_round, where N_round is the number of TT roundings. (b) comes from 96 trees with an
exact reference; it is an envelope, not a proof, and it is loose on fast-migration trees. A MASCOT–exact gap is
meaningful only when it is clearly larger than this error, and `check` says so. A gap at one parameter value says
nothing by itself about bias in MASCOT's estimates; use `fit`/`compare` or `reweight` for that.

**`reweight`.** The weights use the chain's target density as denominator: exp(ℓ_exact − ℓ_logged), where ℓ_logged
is the MASCOT tree density in the log (a column named `Mascot…`, found automatically). If the log has no such
column, escott uses its own converged MASCOT likelihood instead and warns that this is an approximate reweighting;
`--denominator converged` asks for that explicitly. The per-state CSV records both. Trust the reweighted summaries when the weight ESS is a large
fraction of the states used (≥ 50%: escott says so). Below 10% (or ESS < 50) escott prints *reweighting not
reliable — run an exact-likelihood MCMC*, and it means it. Even with even weights, reweighting cannot find
regions of tree or parameter space that the MASCOT chain never visited, and the ESS does not account for chain
autocorrelation. Burn-in is a fraction of the logged states (default 10%).

**`fit` and `compare`.** Each MASCOT optimum ("peak") is refined on the exact likelihood and reported as *converged*
(predicted gain below 1e-4 for the full vector, 0.01 for the tensor train, and every parameter at a bound passes a
KKT probe) or as a *lower bound* of that basin. Compare MASCOT and exact only at refined optima: on a 133-tip avian
influenza tree, comparing an unrefined optimum with a refined one, and a MASCOT multistart that had missed MASCOT's
own best optimum, together produced a model-ranking flip that did not survive refinement. escott therefore also
re-fits MASCOT from every exact optimum and adds any better MASCOT optimum it finds. The exact search starts only
from MASCOT's peaks: an exact optimum with no MASCOT peak nearby would be missed, so use enough `--starts`. Rates
at the lower bound (6e-6) make the χ² reference of the likelihood-ratio test approximate; `compare` carries the exact
side's numerical error into 2ΔlogL and p.

## Limitations

- **Fixed tree.** `check`, `fit` and `compare` condition on one dated tree (for example an MCC tree); tree
  uncertainty enters only through `reweight`, which uses the posterior trees.
- **Constant parameters.** Ne and the migration rates are constant through time; only constant-rate MASCOT logs
  (`NeConstant.*`, `b_migrationConstant.*_to_*`) are read.
- **Number of demes.** Calibrated for d ≤ 4. `estimate` warns for d = 5 and says that d ≥ 6 is not feasible with the
  tensor train. The full vector works for any d while d^k ≤ 3^14.
- **Time per evaluation.** Seconds for the full vector on the H3N2 trees; with the tensor train it grows with the
  rank, which depends on the migration regime. On the 133-tip avian influenza tree (3 demes, 46 lineages at once) one
  evaluation took 2–60 min (median 3.5 min) on a laptop, so a full `fit` there takes hours to days and `reweight` of a
  posterior sample needs a cluster.
- **Likelihood.** The density of the tree given the sampling times, with no root or conditioning term — the quantity
  MASCOT computes. Priors and the sequence likelihood are not part of it.

## Model and conventions

d demes; backward migration rates `M[a][b]` (one lineage moving from deme a to deme b, backwards in time; BEAST's
`b_migration` convention); effective sizes `Ne[a]` (two lineages in deme a coalesce at rate 1/Ne[a]). Times are in
the tree's units. Optimisation is in log parameters within [6.1e-6, 1097].

Tip locations, in order of precedence: `--locations tips.csv` (two columns: tip label, deme), `--location-key KEY`
(a tip annotation such as `max` in a MASCOT summary tree), a `type` or `location` annotation present on every tip, or
a field of the tip label (`--label-field`, `--sep`; default: the last `|` field). The exact engine is chosen
automatically (full vector up to 3^14 states, else tensor train); `check --method dense|tt` overrides it.

## Library

```python
import escott

tree = escott.load_tree("examples/H3N2.mcc.tree", "examples/params_h3n2.json")  # NEXUS/Newick or escott JSON
r = escott.loglik(tree, method="tt", eps=1e-6)        # migration=..., Ne=... override the values in `tree`
r.loglik, r.error, r.info["root_dist"]
escott.loglik(tree, method="dense").loglik             # -31.647356..., exact
escott.loglik(tree, method="mascot").loglik            # -31.565174...
```

`loglik(tree, migration=None, Ne=None, method="tt"|"dense"|"auto"|"mascot", eps=1e-6, paired=True)` returns
`Result(loglik, error, method, info)`; `info["root_dist"]` is the deme distribution of the root lineage.
`escott.fit.fit`, `escott.reweight` and `escott.estimate.estimate` are the functions behind the commands.

The tree dict is `{"demes": d, "M": [[...]], "Ne": [...], "events": [...]}` with events
`{"t", "type": "sample", "id", "deme"}` or `{"t", "type": "coal", "a", "b", "new"}`, t measured backwards from the
most recent tip.

## Tests

```
pip install -e ".[test]" && pytest
```

The tests (about a minute) check that the full vector and the tensor train agree on small trees, that both engines reduce
to Kingman's coalescent with one deme, that the root recursions reproduce the engines exactly, that the per-state
values and the weight ESS of `reweight` match the research run on the H3N2 posterior, that `fit` reaches the
maximum, and that every command runs.

## Source

The engines in `src/escott/_engines/` are unchanged copies of the research code that produced the paper's numbers;
`sync_engines.py` refreshes them and records hashes in `_engines/SOURCES.txt`.

## Citation

[placeholder: manuscript in preparation, Molecular Biology and Evolution (Methods); see `CITATION.cff`]. If you use
the MASCOT comparison, also cite Müller, Rasmussen & Stadler (2018) Bioinformatics 34(22):3843.

## License

MIT (see `LICENSE`). The example data in `examples/` are from the Taming the BEAST MASCOT tutorial, CC BY 4.0.
