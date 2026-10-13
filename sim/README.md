# sim/ — structured coalescent simulator + test set

`simulate.py`: exact Gillespie simulation of the structured coalescent backward in time.
Regenerate the set (deterministic, fixed seeds): `.venv/bin/python sim/simulate.py`.

## Conventions (all of dense/, approx/ use these; tt/ should too)
- `M[a][b]` (a != b): backward-time rate at which ONE lineage currently in deme `a` jumps to deme `b`.
  Diagonal is 0 in the files; total out-rate of deme a is `sum_b M[a][b]`.
- Within deme `a`, each pair of lineages coalesces at rate `1/Ne[a]`.
- Events sorted by `t` (present -> past). Several samples share `t = 0.0`.
  Sample ids are `0..k-1`, internal node ids `k, k+1, ...`.
- Files carry an extra `"meta"` key `{d, k, regime, rep, seed}` (ignore it if you like).

## Parameters
- `Ne[a] = 1 + 0.5 a`  (d=2: [1,1.5]; d=3: [1,1.5,2]; d=4: [1,1.5,2,2.5]).
- Per-pair migration `M[a][b] = factor * cbar * U(0.5,1.5)`, `cbar = mean_a(1/Ne[a])`,
  factor = 0.1 (`slow`), 1 (`medium`), 10 (`fast`).
- Sampling: each sample at t=0 with prob 1/2 (sample 0 always at 0), else `U(0,1)` (heterochronous).
  Demes uniform; when k >= d every deme gets at least one sample.

## Test set (`data/`, ~1.2 MB, list in `data/MANIFEST.txt`)
- `data/d{d}_k{kk}_{regime}_r{rep}.json`: d in {2,3,4}, k in {4,6,...,20}, regime in {slow,medium,fast}, rep 0..2
  -> 243 trees. Seed = 20261003 + 100000 d + 1000 k + 10 regime_idx + rep.
- `data/large/d4_k{30,40,50,60}_{regime}_r{0,1}.json`: 24 trees for the TT headline run
  (dense cannot do these; see dense/README.md for the dense size limit).
