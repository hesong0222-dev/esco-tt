#!/bin/bash
# Generic Kaggle CPU-kernel push (template). Requires the official `kaggle` CLI configured for your account.
# Usage: push_kernel.sh <ACCOUNT> <kernel-name> <driver.py> <dataset-slug> [JOBS] [PROF]
#   <dataset-slug>: a private Kaggle dataset you uploaded that holds tt/, approx/, dense/, bench/ and the tree JSONs.
#   For mle_run3.py, JOBS (comma-separated tags) and PROF (0|1) replace the __JOBS__ / __PROF__ placeholders.
set -e
acct=$1; name=$2; driver=$3; ds=$4; jobs=${5:-}; prof=${6:-0}
dir=build/$name; mkdir -p "$dir"
sed -e "s/__JOBS__/$jobs/" -e "s/__PROF__/$prof/" "$driver" > "$dir/run.py"
cat > "$dir/kernel-metadata.json" <<J
{"id":"$acct/$name","title":"${name//-/ }","code_file":"run.py","language":"python","kernel_type":"script","is_private":true,"enable_gpu":false,"enable_internet":false,"dataset_sources":["$acct/$ds"],"competition_sources":[],"kernel_sources":[]}
J
kaggle kernels push -p "$dir"
