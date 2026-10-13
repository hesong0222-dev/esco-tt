# Kaggle kernel driver for the MLE study, v2 (resume + validation). d3_k30_slow_r16:-,d3_k30_slow_r17:-,d3_k30_slow_r18:-,d3_k30_slow_r19:-,d3_k30_slow_r37:-,d3_k30_slow_r38:-,d3_k30_slow_r39:-,d3_k50_slow_r14:-,d3_k50_slow_r17:-,d3_k50_slow_r29:- = "tag:V,tag:-,..." (V = VALIDATE=1).
import glob, os, subprocess, time
JOBS = [j.split(':') for j in "d3_k30_slow_r16:-,d3_k30_slow_r17:-,d3_k30_slow_r18:-,d3_k30_slow_r19:-,d3_k30_slow_r37:-,d3_k30_slow_r38:-,d3_k30_slow_r39:-,d3_k50_slow_r14:-,d3_k50_slow_r17:-,d3_k50_slow_r29:-".split(',')]
PROF_ALL = "1"
src = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/bench/mle_fit.py', recursive=True)[0]))
O = '/kaggle/working'
os.system('nproc; free -g; python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__); numpy.show_config()" 2>&1 | grep -iE "^[0-9]|name|version" | head -20')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NPROC='4', PROF_ALL=PROF_ALL,
           PRIOR=f'{src}/prior.jsonl')
t0 = time.time()
for tag, v in JOBS:
    if time.time() - t0 > 11.0 * 3600:
        print('time guard: not starting', tag, flush=True)
        continue
    with open(f'{O}/{tag}.log', 'a') as log:
        p = subprocess.Popen(['python', '-u', f'{src}/bench/mle_fit.py', f'{src}/trees/{tag}.json', f'{O}/results.jsonl'],
                             stdout=log, stderr=subprocess.STDOUT, env=dict(env, VALIDATE='1' if v == 'V' else '0'))
        while p.poll() is None:
            time.sleep(20)
            if time.time() - t0 > 11.75 * 3600:
                p.kill()
                print('killed', tag, flush=True)
                break
    print(f'{(time.time()-t0)/3600:.2f} h  {tag} {v} rc={p.returncode}', flush=True)
print('ALL_DONE', flush=True)
