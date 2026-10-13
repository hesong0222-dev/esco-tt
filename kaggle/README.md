# kaggle/ — drivers used for the long runs

Long TT runs (large sweep, robustness, maximum-likelihood study, real-data fits, MultiTypeTree comparison) ran as
private Kaggle CPU kernels (Linux, OpenBLAS, 4 vCPU, ≤ 12 h). This folder keeps the drivers in generic form; account
names, dataset slugs and push/queue scripts specific to the original runs were removed.

| path | used for |
|---|---|
| `kernels/S*, C*, H*, L*/run.py` (+ `jobs.txt`) | large sweep (`bench/large_sweep.md`): runs `bench/headline.py` on the listed trees |
| `kernels/k*/run.py` (+ `trees.txt`) | full-vector (dense) reference values for the largest dense-feasible trees (`dense_big.csv`) |
| `kernels/xob*/run.py` | TT/dense cross-check (`bench/kaggle_xcheck.md`) |
| `kernels/realdata_k*/run.py` | H3N2 fits (`bench/realdata.md`) |
| `mle_run.py`, `mle_run2.py`, `mle_run3.py` | maximum-likelihood study (`bench/mle_study.md`); `__JOBS__`/`__PROF__` are filled by `push_kernel.sh` |
| `fill_make.py` | writes a kernel folder that runs `bench/headline.py` jobs 4 at a time (used for the paired h/2 and eps 1e-7 runs of the large sweep) |
| `templates/robustness_run_template.py` | robustness kernels (`bench/robustness.md`) |
| `templates/realdata2*_run_template.py` | AIV H7 refinement kernels (`bench/realdata2.md`) |
| `templates/mtt_build.py` | bundles `bench/mtt/kernel_main.py` into a kernel (`bench/mtt_compare.md`) |
| `*.csv` | aggregated outputs used by the bench tables (`large*_mascot.csv`, `dense_big.csv`, `xcheck.csv`) |
| `push_kernel.sh` | generic push template (`<ACCOUNT>`, dataset slug as arguments) |

Each driver expects its inputs (the repository's `tt/`, `approx/`, `dense/`, `bench/` code and the tree JSON files)
in a Kaggle dataset mounted under `/kaggle/input/`, and finds them with a recursive glob. Raw per-kernel outputs
(`kaggle/out/...` in the bench write-ups) are not included; the aggregated tables in `bench/` are.
Every run can also be executed locally with the same driver by replacing `/kaggle/input/...` and `/kaggle/working`.
