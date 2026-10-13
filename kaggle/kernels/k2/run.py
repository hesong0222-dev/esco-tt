import sys, json, time, os, csv, glob
src = glob.glob('/kaggle/input/**/esco_dense.py', recursive=True)[0]
sys.path.insert(0, os.path.dirname(src))
import numpy as np, esco_dense
print('cpus', os.cpu_count(), flush=True)
os.system('free -g')
out = '/kaggle/working/results.csv'
with open(out, 'w') as fh: fh.write('file,peak,ll_dense,t_s\n')
for f in "d4_k18_medium_r0.json,d4_k18_medium_r2.json,d4_k20_medium_r2.json,d4_k20_slow_r1.json".split(','):
    path = glob.glob('/kaggle/input/**/' + f, recursive=True)[0]
    tree = json.load(open(path))
    t0 = time.time()
    ll = esco_dense.loglik(tree)
    dt = time.time() - t0
    line = f"{f},{esco_dense.max_lineages(tree)},{ll!r},{dt:.1f}"
    print(line, flush=True)
    with open(out, 'a') as fh: fh.write(line + '\n')
print('ALL_DONE', flush=True)
