# Run times

What the commands cost, as printed by `escott estimate` beforehand and as measured.

**Machine.** Apple M4 laptop (10 cores, 16 GB), macOS, Python 3.13, numpy 2.5.3 / scipy 1.18.1 OpenBLAS wheels, one
BLAS thread per process, `nice -n 10`. The machine was shared with other jobs during these runs (load average 5–45 on
10 cores), so the measured times are pessimistic, by up to about 2× compared with an idle machine. "jobs" is the
`--jobs` value (worker processes).

## H3N2 (24 tips, 3 demes; full vector)

MCC tree: at most 12 lineages at once (3^12 = 531,441 states). Posterior trees: 10–17 lineages, mostly 13.

| command | settings | `estimate` said | measured (wall) |
|---|---|---|---|
| `estimate` | | | 1–4 s |
| `check` (full vector) | ML parameters | 4.0 s per evaluation | 8 s (exact 6.1 s, MASCOT 0.1 s) |
| `check --method tt` | eps 1e-6 + paired check | 21 s (2.3 s – 4 min) per run, paired ×2 | 35 s for both runs (max rank 11) |
| `reweight --thin 20` | 19 states, jobs 3 | 17 s per state (last posterior state) | 8.0 min (sum over states 15 min) |
| `reweight` | all 361 states after burn-in, jobs 3 | 17 s per state → about 35 min | 68 min (interrupted after 190 states and resumed); sum over states 201 min |
| `fit --model asym` | 10 starts, jobs 3 | 30–60 evaluations per peak, about 1 min per peak | 5.9 min (9.7 min in an earlier run under heavier load); MASCOT multistart 1.1 min, 5 peaks refined |
| `compare` | 10 starts per model, jobs 3 | as fit, for both models | 27 min (symmetric 22 min, of which MASCOT multistart 5 min; asymmetric 5 min) |

Per-state times of the full `reweight` run: 348 full-vector states, median 12.9 s (max 222 s); 13 tensor-train states
(15–17 lineages at once), median 321 s (max 1608 s). The tensor-train states are expensive mainly because of the
paired check, which repeats each run at eps/10 and half the time step; `--no-paired-check` skips it and reports only
the empirical bound. The research run of the same evaluations (one process, no paired runs, full vector up to 15
lineages) took 91 min of computing, 2.3 h wall.

Each MASCOT evaluation takes 0.03–0.05 s on this tree, but a start that wanders into extreme parameter values (very
large rates or very small Ne) makes the MASCOT ODE stiff and slow, and the slowest start sets the wall time of the
multistart: one symmetric start took 11 min when 200 L-BFGS-B iterations were allowed; with the cap of 100 the whole
symmetric multistart took 5 min, and `fit` reports "1 start(s) stopped before converging". Exact refinements are
slow for the same reason when a basin lies at large rates: one symmetric peak needed about 90 s per full-vector evaluation instead of 5 s.

## Avian influenza H7 (133 tips, 3 host demes; tensor train) — `estimate` only

The masco MCC tree of Müller, Rasmussen & Stadler (2017) with the BASTA host coding merged to three orders
(Anseriformes, Charadriiformes, Galliformes), at the best asymmetric MASCOT estimate. At most 46 lineages at once,
so 3^46 ≈ 9 × 10^21 states: only the tensor train is possible. No new evaluation was run for this table; the
measured column is from the research runs on this tree (81 tensor-train evaluations at eps 1e-6 at different
parameter values, the same laptop, one process each).

```
$ escott estimate aiv_masco_mcc.trees --locations aiv_order_tips.csv --params aiv_params.json --jobs 4
tree: aiv_masco_mcc.trees  (133 tips, 3 demes, at most 46 lineages at once)
demes: 0=ans, 1=cha, 2=gal   (tip locations from the locations file)
state space: d^k = 3^46 = 8,862,938,119,652,501,095,929 joint deme states
full-vector (dense) engine: not feasible (8.86e+21 states > 4,782,969); 'auto' uses the tensor train
tensor train: within the calibrated range (d <= 4)
migration regime at these parameters: medium (mean migration rate / mean coalescence rate = 2.23; simulations: 0.1 slow, 1 medium, 10 fast)
N_round (number of TT roundings per evaluation) = 135,521
expected max TT rank at eps 1e-6: about 23 (range 9-123; simulated trees, d = 3, rates are uneven, so between the slow and the medium regime); extrapolated beyond the largest simulated trees
expected TT time per evaluation at eps 1e-6: about 12 min (plausible range 56 s-8.5 h); the paired check (eps/10, h/2) adds about x2.0
empirical TT error bound 0.81*eps*N_round: 0.1 at eps 1e-6, 0.01 at eps 1e-7
recommended eps: 1e-07 (largest of 1e-6, 1e-7 with a bound <= 0.02; do not use 1e-5 for inference)
command time (rough, jobs = 4): check about 35 min; reweight about 12 min per posterior state / jobs; fit or compare about 30-60 exact evaluations per likelihood peak and model (about 2.2 h per peak)
```

| quantity | `estimate` said | measured in the research runs |
|---|---|---|
| N_round | 135,521 (at these parameters) | 43,000–154,000 over the parameter values visited |
| max TT rank at eps 1e-6 | about 23 (9–123) | 14–73, median 15 |
| time per evaluation | about 12 min (56 s – 8.5 h) | 2.2–62 min, median 3.5 min |
| paired check | ×2 | h/2 runs: 3.8 and 44 min |
| `check` | about 35 min | one evaluation + one paired run: about 6–100 min from the times above |
| `fit` / `compare` | about 2.2 h per peak with 4 jobs | exact refinement of one peak (research code, same method): 0.7–13.8 h, 1–8 iterations (local laptop or 4 Kaggle CPUs) |

`estimate` recommends eps 1e-7 here because the empirical bound 0.81 · eps · N_round is 0.1 at eps 1e-6. That bound
is an envelope over all benchmark trees; on this tree the paired differences at eps 1e-6 were 2e-4 to 8e-4 at the
best peaks (0.65 at one symmetric peak), so eps 1e-6 with a paired check is a defensible cheaper choice.

## How `estimate` predicts

- Full vector: work = Σ over intervals with n lineages and more than 4096 states of d^n · n · (number of
  uniformisation terms), times 6.5e-9 s; within ×0.8–1.4 on the H3N2 MCC and posterior trees.
- Tensor-train rank: the max rank at eps 1e-6 on simulated trees (3 seeds per cell; d = 2, 3, 4; slow, medium and
  fast migration), interpolated in the number of lineages present at once. The regime is the ratio of the mean
  migration rate to the mean pairwise coalescence rate (0.1 slow, 1 medium, 10 fast in the simulations). When the
  rates are uneven (largest/smallest > 100), the rank is taken to lie between the slow and the classified regime.
  Real trees can fall outside the simulated range (the H3N2 MCC tree has rank 11 at eps 1e-6; the prediction was
  8–38).
- Tensor-train time: a least-squares fit over 428 runs (342 simulated trees, 81 avian-influenza and 5 H3N2
  evaluations): t = exp(−10.687) · N_round^0.055 · k^2.141 · rank^1.524 · d^3.265 s, with 90% of runs within
  ×0.34–3.5 of the fit when the rank is known.
