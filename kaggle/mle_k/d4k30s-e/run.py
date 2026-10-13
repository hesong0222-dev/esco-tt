# Kaggle kernel driver for the MLE study (bench/mle_fit.py). d4_k30_slow_r20,d4_k30_slow_r21,d4_k30_slow_r22,d4_k30_slow_r23,d4_k30_slow_r24/1 are filled in per kernel.
import glob, os, subprocess, time
JOBS = "d4_k30_slow_r20,d4_k30_slow_r21,d4_k30_slow_r22,d4_k30_slow_r23,d4_k30_slow_r24".split(',')
PROF_ALL = "1"
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
