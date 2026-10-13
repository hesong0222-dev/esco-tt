# escott tutorial: H3N2 from start to finish

This walk-through uses the data of the Taming the BEAST MASCOT tutorial that ship with escott in `examples/`
(24 H3N2 sequences, 2000–2001, from Hong Kong, New York and New Zealand; CC BY 4.0):

| file | what it is |
|---|---|
| `H3N2.mcc.tree` | the tutorial's MCC tree (TreeAnnotator, median heights) |
| `H3N2.trees`, `H3N2.log` | posterior trees and trace log of the tutorial's BEAST 2 MASCOT run (401 states) |
| `tips_h3n2.csv` | tip label → deme (the same as the last `\|` field of the labels) |
| `params_h3n2.json` | the exact-likelihood ML estimate on the MCC tree (asymmetric model) |

All output below is pasted from runs on an Apple M4 laptop (Python 3.13, OpenBLAS numpy/scipy, one BLAS
thread per process) that was shared with other jobs, so the times are on the slow side. Progress messages go to
stderr and are mostly omitted; the per-state CSV of `reweight` is shown under its default name.

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd escott          # the package directory, so that examples/ is found
```

## 1. Before computing anything: `estimate`

```
$ escott estimate examples/H3N2.mcc.tree --params examples/params_h3n2.json --jobs 3
tree: examples/H3N2.mcc.tree  (24 tips, 3 demes, at most 12 lineages at once)
demes: 0=Hong_Kong, 1=New_York, 2=New_Zealand   (tip locations from the field -1 of the tip label split at '|')
state space: d^k = 3^12 = 531,441 joint deme states
full-vector (dense) engine: feasible (limit 4,782,969 states); about 34 MB per process, about 4.0 s per evaluation (range x0.8-x1.4); exact to floating point. 'auto' uses it.
tensor train: within the calibrated range (d <= 4)
migration regime at these parameters: medium (mean migration rate / mean coalescence rate = 0.639; simulations: 0.1 slow, 1 medium, 10 fast)
N_round (number of TT roundings per evaluation) = 8,991
expected max TT rank at eps 1e-6: about 17 (range 8-38; simulated trees, d = 3, rates are uneven, so between the slow and the medium regime)
expected TT time per evaluation at eps 1e-6: about 21 s (plausible range 2.3 s-4 min); the paired check (eps/10, h/2) adds about x2.0
empirical TT error bound 0.81*eps*N_round: 0.007 at eps 1e-6, 0.0007 at eps 1e-7
recommended eps: 1e-06 (largest of 1e-6, 1e-7 with a bound <= 0.02; do not use 1e-5 for inference)
command time (rough, jobs = 3): check about 4.0 s; reweight about 4.0 s per posterior state / jobs; fit or compare about 30-60 exact evaluations per likelihood peak and model (about 61 s per peak)
```

The tree never has more than 12 lineages at once, so the full vector has 3^12 = 531,441 entries and escott uses it:
exact to floating point and a few seconds per evaluation. The tensor-train lines matter for larger trees; here they
only show what the alternative would cost. With uneven rates (some near zero, some large) the expected rank spans
the slow and medium regimes of the benchmark, hence the wide range.

The posterior trees differ from the MCC tree. The last logged state, for example:

```
$ escott estimate examples/H3N2.trees --tree-name -1 --log examples/H3N2.log --log-row -1 --jobs 3
tree: examples/H3N2.trees  (24 tips, 3 demes, at most 13 lineages at once)
demes: 0=Hong_Kong, 1=New_York, 2=New_Zealand   (tip locations from the field -1 of the tip label split at '|')
parameters: examples/H3N2.log row Sample=2000000
state space: d^k = 3^13 = 1,594,323 joint deme states
full-vector (dense) engine: feasible (limit 4,782,969 states); about 102 MB per process, about 17 s per evaluation (range x0.8-x1.4); exact to floating point. 'auto' uses it.
tensor train: within the calibrated range (d <= 4)
migration regime at these parameters: medium (mean migration rate / mean coalescence rate = 0.643; simulations: 0.1 slow, 1 medium, 10 fast)
N_round (number of TT roundings per evaluation) = 12,676
expected max TT rank at eps 1e-6: about 18 (range 8-41; simulated trees, d = 3, rates are uneven, so between the slow and the medium regime)
expected TT time per evaluation at eps 1e-6: about 27 s (plausible range 2.6 s-6 min); the paired check (eps/10, h/2) adds about x2.0
empirical TT error bound 0.81*eps*N_round: 0.01 at eps 1e-6, 0.001 at eps 1e-7
recommended eps: 1e-06 (largest of 1e-6, 1e-7 with a bound <= 0.02; do not use 1e-5 for inference)
command time (rough, jobs = 3): check about 17 s; reweight about 17 s per posterior state / jobs; fit or compare about 30-60 exact evaluations per likelihood peak and model (about 4 min per peak)
```

## 2. One parameter value: `check`

```
$ escott check examples/H3N2.mcc.tree --locations examples/tips_h3n2.csv --params examples/params_h3n2.json
tree: examples/H3N2.mcc.tree  (24 tips, 3 demes, at most 12 lineages at once)
demes: 0=Hong_Kong, 1=New_York, 2=New_Zealand   (tip locations from the locations file)
Ne: 0.8758  1097  0.3682
M (backward rate row->col):
       Hong_Kong: 0  0.6145  6.144e-06
        New_York: 3.22  0  6.144e-06
     New_Zealand: 6.144e-06  1.099  0
exact  log L = -31.647356   (full vector, floating-point accuracy; 6.1 s)
MASCOT log L = -31.565174   (MASCOT ODE solved to rtol 1e-10; 0.1 s)
MASCOT - exact = +0.082181

MASCOT is 0.0822 log units higher than the exact likelihood here; this is larger than the exact method's estimated numerical error (floating-point level), so the gap is due to MASCOT's approximation. The gap is small; MASCOT is very likely adequate for this tree. Note: this compares the two likelihoods at one parameter value; it does not by itself show that MASCOT's parameter estimates are biased.
```

The same with the tensor train, to see its error estimate at work:

```
$ escott check examples/H3N2.mcc.tree --params examples/params_h3n2.json --method tt
tree: examples/H3N2.mcc.tree  (24 tips, 3 demes, at most 12 lineages at once)
demes: 0=Hong_Kong, 1=New_York, 2=New_Zealand   (tip locations from the field -1 of the tip label split at '|')
Ne: 0.8758  1097  0.3682
M (backward rate row->col):
       Hong_Kong: 0  0.6145  6.144e-06
        New_York: 3.22  0  6.144e-06
     New_Zealand: 6.144e-06  1.099  0
exact  log L = -31.647384   (tensor train, eps 1e-06, max rank 11; est. error 0.007; 33.1 s)
         error estimated from the paired run (eps/10, h/2): 1e-05; empirical bound 0.81*eps*N_round: 0.007 (N_round 8,991); the larger is used
MASCOT log L = -31.565174   (MASCOT ODE solved to rtol 1e-10; 0.1 s)
MASCOT - exact = +0.082210

MASCOT is 0.0822 log units higher than the exact likelihood here; this is larger than the exact method's estimated numerical error (0.007), so the gap is due to MASCOT's approximation. The gap is small; MASCOT is very likely adequate for this tree. Note: this compares the two likelihoods at one parameter value; it does not by itself show that MASCOT's parameter estimates are biased.
```

The tensor-train value differs from the exact one by 2.8e-5. The paired run (eps/10, half time step) estimates
1e-5, the empirical bound 0.007; escott reports the larger. Both are far below the 0.082 gap, so the gap is
MASCOT's.

## 3. The posterior: `reweight`

`reweight` pairs every logged state of `H3N2.log` with tree `STATE_<n>` of `H3N2.trees`, drops the first 10% as
burn-in, and computes at each state the exact log-likelihood (full vector up to 3^14 states, tensor train above), the
converged MASCOT log-likelihood and both root distributions. The weight of a state is the exact likelihood divided by
the density the chain sampled from, the MASCOT tree density in the log's `Mascot` column. Per-state results go to
`examples/H3N2.escott-reweight.csv` (or `--out`) as they finish, with both MASCOT values; if the run is
interrupted, the same command with `--resume` continues where it stopped. Our run was interrupted after 190 of 361
states and resumed.

```
$ escott reweight examples/H3N2.trees examples/H3N2.log --jobs 3
[    0.0s] 361 states to evaluate, 0 already in examples/H3N2.escott-reweight.csv, 361 to go (3 processes)
[    2.5s] [1/361] state 205000: 12 lineages, dense, 1.96 s, exact -33.0141, MASCOT - exact +0.2738; about 15 min left
[   10.5s] [2/361] state 215000: 13 lineages, dense, 8.03 s, exact -41.8219, MASCOT - exact +0.2356; about 31 min left
[   18.4s] [3/361] state 210000: 12 lineages, dense, 17.79 s, exact -35.2922, MASCOT - exact -0.0326; about 36 min left
...
```

and after `--resume` finished the remaining 171 states:

```
trees: examples/H3N2.trees; log: examples/H3N2.log; demes: Hong_Kong, New_York, New_Zealand (tip locations from the field -1 of the tip label split at '|')
per-state results: examples/H3N2.escott-reweight.csv
states: 401 in the log; burn-in 40 dropped; thin 1 -> 361 used (exact engine: 348 dense, 13 tensor train)
weights = exact likelihood / the chain's target density (log column 'Mascot')
log-weight l_exact - l_denominator: median -0.448, range -3.559 .. +0.549, SD 0.533
weight ESS: 309.1 of 361 states (86%); largest normalised weight 0.008
largest per-state error estimate of the exact value: 0.02 log units (moves a weight by about 2% at most)

parameter                          sample mean        sample 95% HPD  exact mean         exact 95% HPD   shift shift/SD
Ne.Hong_Kong                             1.372          0.119 - 4.09        1.41          0.165 - 4.27   +2.8%   +0.025
Ne.New_York                              2.046          0.179 - 5.86       1.962           0.179 - 5.8   -4.1%   -0.042
Ne.New_Zealand                          0.8499          0.122 - 2.18      0.8949          0.133 - 2.19   +5.3%   +0.068
m.Hong_Kong_to_New_York                  1.139        0.00539 - 3.78       1.177         0.0194 - 3.88   +3.3%   +0.033
m.Hong_Kong_to_New_Zealand              0.5088       0.000242 - 1.67      0.4679        0.000242 - 1.5   -8.0%   -0.070
m.New_York_to_Hong_Kong                  1.382        0.00747 - 3.14       1.447        0.00747 - 3.35   +4.7%   +0.065
m.New_York_to_New_Zealand               0.6162       0.000623 - 1.92      0.5673       0.000844 - 1.74   -7.9%   -0.079
m.New_Zealand_to_Hong_Kong              0.7493        0.00296 - 2.12      0.7743        0.00296 - 2.13   +3.3%   +0.037
m.New_Zealand_to_New_York               0.8433        0.00627 - 2.18        0.83        0.00627 - 2.18   -1.6%   -0.018
Tree.height                              3.609           2.68 - 4.56       3.664           2.63 - 4.63   +1.5%   +0.103
clockRate                             0.004002     0.00288 - 0.00508    0.003962     0.00274 - 0.00502   -1.0%   -0.065

root deme probability (posterior mean of the per-state root distribution):
                                           Hong_Kong    New_York New_Zealand
  BEAST sample, MASCOT root                    0.503       0.250       0.247
  exact weights, MASCOT root                   0.528       0.262       0.210
  exact posterior, exact root                  0.531       0.263       0.206

The weights are even: the exact posterior is well represented by the BEAST sample, and the reweighted summaries can be trusted within the support caveat below.
Support caveat: reweighting only moves weight among the states the MASCOT chain visited; it cannot find regions the MASCOT posterior misses. The ESS above counts weight unevenness only, not chain autocorrelation.
```

How to read it:

- The log-weights are mostly a common offset (median −0.45, SD 0.53). A common offset cancels in the weights, so the
  weights are even: ESS 309 of 361 (86%), largest weight 0.8%.
- The posterior means move by −8% to +5%, at most about 0.1 posterior SD; the 95% HPD intervals barely change. The
  probability that the root is in Hong Kong rises from 0.50 to 0.53, almost all of it through the weights rather than
  through the per-state root distribution.
- 13 states have 15–17 lineages at once and use the tensor train with a paired check; their largest error estimate
  is 0.02 log units, which changes a weight by at most 2%.
- The verdict line: the weights are even, so these summaries can be trusted, with the caveat that reweighting cannot
  find regions the MASCOT chain did not visit.

If the log has no MASCOT density column, escott divides by its own converged MASCOT likelihood instead and says that
this is an approximate reweighting; `--denominator converged` does the same on request.

A quick look with fewer states, for example `--thin 20` (19 states), gives shifts of the same size but noisier ones,
and escott warns that 19 states are too few:

```
$ escott reweight examples/H3N2.trees examples/H3N2.log --thin 20 --jobs 3
trees: examples/H3N2.trees; log: examples/H3N2.log; demes: Hong_Kong, New_York, New_Zealand (tip locations from the field -1 of the tip label split at '|')
per-state results: examples/H3N2.escott-reweight.csv
states: 401 in the log; burn-in 40 dropped; thin 20 -> 19 used (exact engine: 18 dense, 1 tensor train)
weights = exact likelihood / the chain's target density (log column 'Mascot')
log-weight l_exact - l_denominator: median -0.688, range -1.486 .. -0.095, SD 0.393
weight ESS: 16.7 of 19 states (88%); largest normalised weight 0.085
largest per-state error estimate of the exact value: 0.004 log units (moves a weight by about 0.4% at most)

parameter                          sample mean        sample 95% HPD  exact mean         exact 95% HPD   shift shift/SD
Ne.Hong_Kong                             1.433          0.285 - 5.03       1.465          0.285 - 3.56   +2.2%   +0.026
Ne.New_York                              2.446          0.327 - 6.96       2.475          0.469 - 6.96   +1.2%   +0.016
Ne.New_Zealand                          0.7769           0.19 - 3.48      0.8876          0.206 - 3.48  +14.2%   +0.151
m.Hong_Kong_to_New_York                 0.9799          0.062 - 4.09      0.8789          0.062 - 2.28  -10.3%   -0.106
m.Hong_Kong_to_New_Zealand              0.2203         0.008 - 0.655      0.2286         0.008 - 0.605   +3.8%   +0.046
m.New_York_to_Hong_Kong                  1.156         0.0211 - 4.17       1.296         0.0211 - 4.17  +12.1%   +0.141
m.New_York_to_New_Zealand               0.5601         0.0565 - 2.77      0.5425        0.0565 - 0.925   -3.1%   -0.030
m.New_Zealand_to_Hong_Kong              0.7006         0.0439 - 1.49      0.7703         0.0439 - 1.49   +9.9%   +0.138
m.New_Zealand_to_New_York               0.8132         0.0776 - 2.88       0.903           0.17 - 2.88  +11.0%   +0.125
Tree.height                              3.858           2.63 - 6.64       3.829            2.63 - 4.5   -0.8%   -0.036
clockRate                             0.003739     0.00238 - 0.00524    0.003718       0.003 - 0.00524   -0.6%   -0.028

root deme probability (posterior mean of the per-state root distribution):
                                           Hong_Kong    New_York New_Zealand
  BEAST sample, MASCOT root                    0.497       0.265       0.238
  exact weights, MASCOT root                   0.549       0.264       0.187
  exact posterior, exact root                  0.552       0.267       0.182

WARNING: too few effective states (ESS 17 < 50) for reliable summaries. Use more states (less thinning); if the ESS stays low, run an exact-likelihood MCMC.
Support caveat: reweighting only moves weight among the states the MASCOT chain visited; it cannot find regions the MASCOT posterior misses. The ESS above counts weight unevenness only, not chain autocorrelation.
```

## 4. Point estimates: `fit`

```
$ escott fit examples/H3N2.mcc.tree --locations examples/tips_h3n2.csv --model asym --jobs 3
tree: examples/H3N2.mcc.tree  (24 tips, 3 demes, at most 12 lineages at once)
demes: 0=Hong_Kong, 1=New_York, 2=New_Zealand   (tip locations from the locations file)
model asym: 6 migration rates + 3 Ne = 9 parameters; log-space bounds [6.1e-06, 1097]
MASCOT multistart: 10 starts -> 5 distinct optima; best reached by 5 start(s)
exact engine: dense

peaks, sorted by exact log L ('exact at MASCOT opt.' = exact value at MASCOT's optimum; root = probability of Hong_Kong / New_York / New_Zealand):
   #      origin  MASCOT opt.     exact at    exact max    error        exact root      MASCOT root  status
                               MASCOT opt.
   1  multistart     -31.5406     -31.6757     -31.6474        0    1.00/0.00/0.00   1.00/0.00/0.00  converged
   2  multistart     -31.8613     -32.2922     -32.2761        0    1.00/0.00/0.00   1.00/0.00/0.00  lower bound (moving m.Hong_Kong_to_New_York off its bound raises log L by 0.00205)
   3  multistart     -31.8452     -32.3412     -32.2763        0    1.00/0.00/0.00   1.00/0.00/0.00  lower bound (moving m.Hong_Kong_to_New_York off its bound raises log L by 0.00197)
   4  multistart     -33.4030     -33.3463     -33.3314        0    0.00/1.00/0.00   0.00/1.00/0.00  converged
   5  multistart     -36.2349     -36.7855     -36.7485        0    0.70/0.00/0.30   0.62/0.00/0.38  converged
cross-check: re-fitting MASCOT from each refined exact optimum found no better MASCOT optimum.

best exact fit: log L = -31.6474 (error 0; converged); MASCOT at the same parameters: -31.5654
best MASCOT fit: log L = -31.5406
  parameter                           MASCOT MLE   exact MLE   change
  m.Hong_Kong_to_New_York                 0.6491      0.6133    -5.5%
  m.Hong_Kong_to_New_Zealand           6.144e-06   6.144e-06 at bound
  m.New_York_to_Hong_Kong                  2.985       3.223    +7.9%
  m.New_York_to_New_Zealand            6.144e-06   6.144e-06 at bound
  m.New_Zealand_to_Hong_Kong           6.144e-06   6.144e-06 at bound
  m.New_Zealand_to_New_York                1.212         1.1    -9.2%
  Ne.Hong_Kong                             0.838       0.876    +4.5%
  Ne.New_York                               1097        1097 at bound
  Ne.New_Zealand                          0.3273      0.3682   +12.5%
```

How to read it:

- Ten random starts of MASCOT find five distinct optima. Each is refined on the exact likelihood (here with the full
  vector, so "error 0").
- Peak 1 is the best under both models and converged: exact log L −31.6474 at the exact MLE, while MASCOT gives
  −31.5654 at the same parameters (+0.08, the gap `check` showed). The root is in Hong Kong with probability 1.00 under
  both.
- Peaks 2 and 3 are lower bounds: a rate at its lower bound (≈ 0) would raise the exact log-likelihood a little
  (0.002) if moved off the bound, so those basins are not fully explored. They are 0.63 below peak 1 and do not
  matter here, but a lower bound near the top would.
- Re-fitting MASCOT from each refined exact optimum found nothing better than the multistart's best, so the MASCOT
  side of the comparison is not an artefact of a missed MASCOT optimum.
- The estimates move by 4–13% between MASCOT and the exact likelihood. Three rates and Ne of New York sit at bounds
  under both (no detectable migration on those routes; almost no coalescence in New York).

## 5. Model comparison: `compare`

```
$ escott compare examples/H3N2.mcc.tree --locations examples/tips_h3n2.csv --jobs 3
tree: examples/H3N2.mcc.tree  (24 tips, 3 demes, at most 12 lineages at once)
demes: 0=Hong_Kong, 1=New_York, 2=New_Zealand   (tip locations from the locations file)
model sym: 3 migration rates + 3 Ne = 6 parameters; log-space bounds [6.1e-06, 1097]
MASCOT multistart: 10 starts -> 3 distinct optima; best reached by 8 start(s); 1 start(s) stopped before converging
exact engine: dense

peaks, sorted by exact log L ('exact at MASCOT opt.' = exact value at MASCOT's optimum; root = probability of Hong_Kong / New_York / New_Zealand):
   #      origin  MASCOT opt.     exact at    exact max    error        exact root      MASCOT root  status
                               MASCOT opt.
   1  multistart     -33.6228     -34.0019     -33.9619        0    0.46/0.02/0.51   0.49/0.04/0.47  converged
   2  multistart     -36.8419     -37.8749     -33.9907        0    0.47/0.01/0.52   0.24/0.11/0.65  lower bound (iteration cap)
   3  multistart     -35.8670     -36.0147     -35.9829        0    0.21/0.44/0.35   0.23/0.42/0.35  lower bound (moving m.Hong_Kong<->New_York off its bound raises log L by 0.00676)
cross-check: re-fitting MASCOT from each refined exact optimum found no better MASCOT optimum.

best exact fit: log L = -33.9619 (error 0; converged); MASCOT at the same parameters: -33.6665
best MASCOT fit: log L = -33.6228
  parameter                           MASCOT MLE   exact MLE   change
  m.Hong_Kong<->New_York                   1.809       1.958    +8.2%
  m.Hong_Kong<->New_Zealand            6.144e-06   6.144e-06 at bound
  m.New_York<->New_Zealand                0.5076      0.5554    +9.4%
  Ne.Hong_Kong                            0.4142      0.3993    -3.6%
  Ne.New_York                              8.155       16.39  +101.0%
  Ne.New_Zealand                           0.322      0.3464    +7.6%

model asym: 6 migration rates + 3 Ne = 9 parameters; log-space bounds [6.1e-06, 1097]
MASCOT multistart: 11 starts -> 5 distinct optima; best reached by 6 start(s)
exact engine: dense

peaks, sorted by exact log L ('exact at MASCOT opt.' = exact value at MASCOT's optimum; root = probability of Hong_Kong / New_York / New_Zealand):
   #      origin  MASCOT opt.     exact at    exact max    error        exact root      MASCOT root  status
                               MASCOT opt.
   1  multistart     -31.5406     -31.6757     -31.6474        0    1.00/0.00/0.00   1.00/0.00/0.00  converged
   2  multistart     -31.8613     -32.2922     -32.2761        0    1.00/0.00/0.00   1.00/0.00/0.00  lower bound (moving m.Hong_Kong_to_New_York off its bound raises log L by 0.00205)
   3  multistart     -31.8452     -32.3412     -32.2763        0    1.00/0.00/0.00   1.00/0.00/0.00  lower bound (moving m.Hong_Kong_to_New_York off its bound raises log L by 0.00197)
   4  multistart     -33.4030     -33.3463     -33.3314        0    0.00/1.00/0.00   0.00/1.00/0.00  converged
   5  multistart     -36.2349     -36.7855     -36.7485        0    0.70/0.00/0.30   0.62/0.00/0.38  converged
cross-check: re-fitting MASCOT from each refined exact optimum found no better MASCOT optimum.

best exact fit: log L = -31.6474 (error 0; converged); MASCOT at the same parameters: -31.5654
best MASCOT fit: log L = -31.5406
  parameter                           MASCOT MLE   exact MLE   change
  m.Hong_Kong_to_New_York                 0.6491      0.6133    -5.5%
  m.Hong_Kong_to_New_Zealand           6.144e-06   6.144e-06 at bound
  m.New_York_to_Hong_Kong                  2.985       3.223    +7.9%
  m.New_York_to_New_Zealand            6.144e-06   6.144e-06 at bound
  m.New_Zealand_to_Hong_Kong           6.144e-06   6.144e-06 at bound
  m.New_Zealand_to_New_York                1.212         1.1    -9.2%
  Ne.Hong_Kong                             0.838       0.876    +4.5%
  Ne.New_York                               1097        1097 at bound
  Ne.New_Zealand                          0.3273      0.3682   +12.5%

likelihood-ratio test, asymmetric vs symmetric migration (3 demes): df = 3; rates at a bound make the chi-square reference approximate
  engine      best asym     best sym          2 Delta                      p
  MASCOT       -31.5406     -33.6228            4.165                  0.244
  exact        -31.6474     -33.9619            4.629                  0.201
MASCOT does not reject symmetry at 5%; the exact likelihood does not reject symmetry at 5%.
```

How to read it:

- `compare` fits both models with the procedure of `fit`. The asymmetric multistart has 11 starts: the best
  symmetric MASCOT fit is added as a start, so that the asymmetric optimum cannot end below the symmetric one.
- The verdict is the same under both engines: symmetric migration is not rejected (MASCOT 2ΔlogL = 4.17, p = 0.24;
  exact 4.63, p = 0.20; df = 3). With the full vector the exact side has no numerical error to carry; with the tensor
  train, 2ΔlogL and p would be given with the error of both optima.
- What does change is the root of the symmetric model: MASCOT puts it in Hong Kong (0.49 vs New Zealand 0.47), the
  exact likelihood in New Zealand (0.51 vs 0.46). Both are close to 50/50, so neither should be reported as a
  conclusion.
- Ne of New York is poorly identified: 8 (MASCOT) vs 16 (exact) in the symmetric model, at the upper bound in the
  asymmetric one. Read it as "almost no coalescence in New York".
- Symmetric peak 2 comes from a MASCOT start that stopped before converging. Its exact refinement climbs towards
  peak 1 (−33.99 vs −33.96) and is marked as a lower bound because it ran out of iterations; it is the same basin,
  not a second optimum. More `--max-iter` would merge it.

## What to do on your own data

1. `estimate` with your tree and a plausible parameter set (for example `--log run.log`). If the full vector is
   feasible, everything is cheap. If not, look at the predicted time per evaluation: `check` needs one (plus the
   paired run), `reweight` one per posterior state, `fit`/`compare` a few dozen per peak.
2. `check` at a few posterior states or at your MCC/ML estimate. A gap below about 0.1 log units at every state you
   look at means MASCOT is adequate for likelihood comparisons on this tree; gaps of a unit or more mean it may not
   be.
3. If you have the BEAST 2 run, `reweight` answers the Bayesian question directly. Thin the sample to what you can
   afford, but keep at least ~100 states.
4. If you report ML estimates or likelihood-ratio tests, run `fit`/`compare` and check that the optima you rely on are
   "converged", not "lower bound".
