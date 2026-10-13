# Changelog

## 0.2.0 (2026-10-13)

New commands:
- `escott estimate`: state space, dense feasibility, N_round, expected tensor-train rank and time per evaluation, and
  a recommended `eps`, before anything is computed.
- `escott reweight`: importance reweighting of a BEAST 2 MASCOT posterior sample (`.trees` + `.log`) to the exact
  structured-coalescent posterior (weights: exact likelihood over the chain's target density, the log's MASCOT
  column; `--denominator converged` for escott's converged MASCOT likelihood), with weight ESS, posterior means and 95% HPD intervals under both models, root-deme
  probabilities and a reliability warning. Per-state results are written to a CSV as they finish (`--resume`), and
  states are evaluated in parallel (`--jobs`).
- `escott fit`: maximum likelihood under MASCOT and the exact likelihood (MASCOT multistart, exact refinement of every
  peak, MASCOT re-fitted from every exact optimum as a cross-check), with a table of peaks and their convergence status.
- `escott compare`: likelihood-ratio test of symmetric vs asymmetric migration under both engines, with the exact
  side's numerical error carried into 2ΔlogL and p.

Other changes:
- `check`: the paired error run (eps/10, h/2) can be switched off with `--no-paired-check`; both error estimates are
  printed.
- Tip locations are also read from a `type` or `location` annotation present on every tip; clearer error messages for
  missing tips, columns and implausible deme assignments.
- `loglik(..., paired=False)` runs the tensor train once; every method now returns the root-deme distribution in
  `info["root_dist"]`.
- The full-vector engine is used automatically up to 3^14 states (was 3^13).
- A warning is printed when numpy or scipy links Apple Accelerate on macOS.
- Requires Python >= 3.10.

## 0.1.0 (2026-10-06)

- `escott check`: exact vs MASCOT log-likelihood of one tree at given parameters.
- Library: `loglik(tree, migration, Ne, method="tt"|"dense"|"auto"|"mascot")`, NEXUS/Newick and BEAST log readers.
