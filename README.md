# TT-ESCO: exact structured-coalescent likelihoods with tensor trains

Code, benchmark results and input data for

> Song H. *Exact structured-coalescent likelihoods beyond the full state space with tensor trains.* (preprint;
> link and DOI to be added)

The structured coalescent's exact likelihood on a fixed time tree (ESCO) needs the joint distribution of all lineage
locations, d^k states for k lineages in d demes. `tt/` represents that state as a tensor train with one core per
lineage in the tree's planar leaf order, so every sampling, migration and coalescence step is a local operation.
The repository also holds an independent full-vector implementation (`dense/`), a converged MASCOT implementation
(`approx/`), simulators, every benchmark behind the paper's tables and figures, and a small user package (`pkg/`,
`escott`) that compares MASCOT with the exact likelihood on your own tree.

## Layout

| path | content |
|---|---|
| `tt/` | TT-ESCO (`esco_tt.py`, `loglik(tree)`), TT toolkit (`ttcore.py`), brute-force check (`brute.py`) |
| `dense/` | full-vector exact likelihood (`esco_dense.py`) and its validation (closed forms, Monte Carlo) |
| `approx/` | MASCOT (DOP853, rtol 1e-10) and SISCO/LISCO (`independence.py`, if present) |
| `sim/` | structured-coalescent simulator and the small test set (`sim/data/`) |
| `pkg/` | `escott` package: `escott.loglik(...)`, `escott check TREE ...` (see `pkg/README.md`) |
| `bench/` | benchmark scripts, result CSVs and write-ups (`*.md`), one per study |
| `kaggle/` | generic drivers of the long cloud runs and aggregated outputs (`kaggle/README.md`) |
| `data/` | real data (H3N2 tutorial; AIV H7) with sources, hashes and licences (`data/README.md`) |
| `paper/figures/` | figure scripts (`fig1`–`fig5`) and `stats.py`, which prints every derived number in the text |
| `PREREGISTRATION.md` | pre-registration of the maximum-likelihood study, with its timestamp |

Tree input format (all engines): JSON `{"M": [[...]], "Ne": [...], "events": [...]}`, events sorted by time from the
present backwards, `{"t", "type": "sample", "id", "deme"}` or `{"t", "type": "coal", "a", "b", "new"}`.
`M[a][b]` is the backward-time rate of one lineage moving from deme a to deme b (BEAST's `b_migration`
convention); two lineages in deme a coalesce at rate `1/Ne[a]`. See `sim/README.md`.

## Install

Python ≥ 3.11 (tested with 3.13.12).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # numpy 2.5.3, scipy 1.18.1, matplotlib, pandas (exact pins)
.venv/bin/pip install -e pkg                       # optional: the escott package and CLI
```

### Compute notes (read before long runs)

- **Apple Silicon / Apple Accelerate.** Do not run long TT jobs with a NumPy/SciPy build linked to Apple
  Accelerate's LAPACK. On macOS 26 (M4) we saw heap corruption in `qr`/`svd`: segmentation faults and, worse,
  silently wrong log-likelihoods. OpenBLAS builds (`scipy-openblas`) were fine on the same jobs. The default PyPI
  wheels for macOS ≥ 14 link Accelerate. Use the OpenBLAS wheels instead:
  ```bash
  .venv/bin/pip download --only-binary=:all: --platform macosx_12_0_arm64 --python-version 3.13 \
      numpy==2.5.3 scipy==1.18.1 -d wheels/
  .venv/bin/pip install --force-reinstall --no-deps wheels/*.whl
  .venv/bin/python -c "import numpy; numpy.show_config()"   # lapack name must be scipy-openblas
  ```
  Linux wheels are not affected. All results in this repository were produced with OpenBLAS (Linux, or macOS with
  the wheels above); `bench/openblas_rerun.md` documents the re-check of earlier Accelerate-era values.
- **Threads.** Run single-threaded and parallelise over trees: `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
- **Cost.** Small trees take seconds. The large sweep (d = 3–4, k up to 60) took up to ~11 h per tree on 4-vCPU
  Linux machines; those runs used the drivers in `kaggle/`. Wall times measured on the (shared, loaded) development
  Mac are relative only.

## Reproduce

All commands from the repository root, with `PY=.venv/bin/python` and the thread variables above.

**Figures and derived numbers (from the shipped CSVs, minutes):**

```bash
$PY paper/figures/fig1_schematic.py      # Fig. 1  method schematic
$PY paper/figures/fig2_tt_vs_dense.py    # Fig. 2  TT vs full vector   <- bench/tt_vs_dense.csv, kaggle/xcheck.csv, kaggle/dense_big.csv
$PY paper/figures/fig3_rank_scaling.py   # Fig. 3  rank and run time   <- bench/rank_study.csv, bench/large_sweep.csv
$PY paper/figures/fig4_mascot_error.py   # Fig. 4  MASCOT error map    <- bench/dense_vs_mascot.csv, bench/large_sweep.csv
$PY paper/figures/fig5_h3n2.py           # Fig. 5  H3N2                <- bench/realdata.csv
$PY paper/figures/stats.py               # every derived number quoted in the text
```

**Tables (from the shipped per-run outputs):**

| table | command | write-up |
|---|---|---|
| Table 1 (MLE study) | values in `bench/mle_study.csv`, `bench/mle_study_summary.csv`; `bench/mle_summarize.py` rebuilds them from the per-kernel `results.jsonl` files (not shipped, regenerate with the runs below) | `bench/mle_study.md`, `PREREGISTRATION.md` |
| Table 2 (rank) | `$PY bench/robustness_analyze.py` → `bench/robustness/summary_rank_v2.csv` | `bench/robustness.md` §3 |
| Table 3 (AIV H7) | `$PY bench/realdata2_summarize.py` | `bench/realdata2.md` |

**Re-running the computations (hours to days):**

| result | command | write-up |
|---|---|---|
| validation of `dense/` | `$PY dense/validate.py`; `$PY dense/validate_mc.py` | `bench/validate_*.txt` |
| TT vs full vector (small trees) | `$PY bench/crosscheck_tt_dense.py out.csv [max_states] [eps]` | `bench/tt_vs_dense.csv` |
| MASCOT vs exact (small trees) | `$PY bench/run_dense_vs_mascot.py`; `$PY bench/summarize.py` | `bench/dense_vs_mascot.csv` |
| large sweep (one tree) | `$PY bench/headline.py sim/data/large/<tree>.json 1e-6 1 out.json` | `bench/large_sweep.md` (`bench/large_sweep_build.py`) |
| rank study | `$PY bench/rank_study.py out.csv "d,k,m,seed,eps" ...`; v2 trees `bench/robustness_make_rank_trees.py` | `bench/rank_study.md`, `bench/robustness.md` |
| error model / robustness | `$PY bench/robustness.py out.csv TREE.json,eps,steps_mult,mass_fix ...` | `bench/robustness.md` |
| MLE study | trees `$PY bench/mle_make_trees.py`; one tree `$PY bench/mle_fit.py <tree.json> <out.jsonl>` | `bench/mle_study.md` |
| H3N2 fits | `$PY bench/realdata_fit.py ...`, `bench/realdata_multistart.py`, `bench/realdata_ttcheck.py` | `bench/realdata.md` |
| AIV H7 fits | `$PY bench/realdata2_refine.py ...`, `bench/realdata2_eval.py ...` | `bench/realdata2.md` |
| H3N2 posterior reweighting | `$PY bench/h3n2_posterior_eval.py` (~2.3 h), then `$PY bench/h3n2_posterior_reweight.py` | `bench/h3n2_posterior_reweight.md` |
| MultiTypeTree comparison | `bench/mtt/` (`run_local.py`, `kernel_main.py`) | `bench/mtt_compare.md` |

Each script's docstring gives its exact arguments. Simulated test sets are deterministic (`sim/simulate.py`,
`bench/mle_make_trees.py`, fixed seeds).

**Tests:** `.venv/bin/pip install -e "pkg[test]" && .venv/bin/python -m pytest pkg/tests` (about 30 s).

## Data

See `data/README.md` for every file's source, commit, sha256 and licence: the Taming the BEAST MASCOT tutorial
(H3N2, CC BY 4.0), the AIV trees of Müller, Rasmussen & Stadler (2017; CC BY 3.0) and the host assignments of the
BASTA example (GPL-3.0).

## Notes on the write-ups

The `bench/*.md` files are working records written while the study ran; a few are in Korean. They name the
generating scripts and are kept unchanged apart from removing machine-specific paths and account names. Paths of
the form `kaggle/out/...` refer to raw cloud outputs that are not included; the aggregated CSVs are.

## License and citation

Code: MIT (`LICENSE`). Third-party data keep their licences (see `LICENSE` and `data/README.md`); simulated trees
and benchmark outputs: CC BY 4.0. Please cite the paper and this archive (`CITATION.cff`).
