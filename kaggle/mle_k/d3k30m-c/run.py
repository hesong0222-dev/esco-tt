# Kaggle kernel driver for the MLE study (bench/mle_fit.py). d3_k30_medium_r14,d3_k30_medium_r15,d3_k30_medium_r16,d3_k30_medium_r17,d3_k30_medium_r18,d3_k30_medium_r19,d3_k30_medium_r20/0 are filled in per kernel.
import glob, os, subprocess, time
JOBS = "d3_k30_medium_r14,d3_k30_medium_r15,d3_k30_medium_r16,d3_k30_medium_r17,d3_k30_medium_r18,d3_k30_medium_r19,d3_k30_medium_r20".split(',')
PROF_ALL = "0"
src = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/bench/mle_fit.py', recursive=True)[0]))
O = '/kaggle/working'
os.system('nproc; free -g')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NPROC='4', PROF_ALL=PROF_ALL)
t0 = time.time()
for tag in JOBS:
    if time.time() - t0 > 11.0 * 3600:
        print('time guard: not starting', tag, flush=True)
        continue
    with open(f'{O}/{tag}.log', 'w') as log:
        p = subprocess.Popen(['python', '-u', f'{src}/bench/mle_fit.py', f'{src}/trees/{tag}.json', f'{O}/results.jsonl'],
                             stdout=log, stderr=subprocess.STDOUT, env=env)
        while p.poll() is None:
            time.sleep(20)
            if time.time() - t0 > 11.75 * 3600:
                p.kill()
                print('killed', tag, flush=True)
                break
    print(f'{(time.time()-t0)/3600:.2f} h  {tag} rc={p.returncode}', flush=True)
print('ALL_DONE', flush=True)
