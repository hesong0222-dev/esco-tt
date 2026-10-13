import glob, os, shutil, subprocess, time
src = os.path.dirname(glob.glob('/kaggle/input/**/headline.py', recursive=True)[0])
work = '/kaggle/working/esco'; os.makedirs(work + '/bench', exist_ok=True)
shutil.copytree(src + '/tt', work + '/tt', dirs_exist_ok=True)
shutil.copy(src + '/headline.py', work + '/bench/headline.py')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
queue = "d4_k40_slow_r2,d4_k40_medium_r2,d4_k40_fast_r2".split(','); running = []; t0 = time.time()
while queue or running:
    while queue and len(running) < 4:
        t = queue.pop(0); log = open(f'/kaggle/working/{t}.log', 'w')
        running.append((t, subprocess.Popen(['python', '-u', work + '/bench/headline.py', f'{src}/{t}.json', '1e-6', '1', f'/kaggle/working/{t}.json'], stdout=log, stderr=subprocess.STDOUT, env=env)))
    time.sleep(60)
    for t, p in list(running):
        if p.poll() is not None:
            print(f'{(time.time()-t0)/3600:.2f} h done {t} rc={p.returncode}', flush=True); running.remove((t, p))
print('ALL_DONE', flush=True)
