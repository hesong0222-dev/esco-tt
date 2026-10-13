import glob, os, shutil, subprocess, time
src = os.path.dirname(glob.glob('/kaggle/input/**/headline.py', recursive=True)[0])
work = '/kaggle/working/esco'; os.makedirs(work + '/bench', exist_ok=True)
shutil.copytree(src + '/tt', work + '/tt', dirs_exist_ok=True)
shutil.copy(src + '/headline.py', work + '/bench/headline.py')
os.system('free -g; nproc')
jobs = [('d4_k50_medium_r0', '1e-6', '1'), ('d4_k50_medium_r0', '1e-6', '2'), ('d4_k50_medium_r0', '1e-7', '1'), ('d4_k60_medium_r0', '1e-6', '1')]
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
ps = []
for t, eps, sm in jobs:
    tag = f'{t}_eps{eps}_sm{sm}'
    log = open(f'/kaggle/working/{tag}.log', 'w')
    ps.append(subprocess.Popen(['python', '-u', work + '/bench/headline.py', f'{src}/{t}.json', eps, sm, f'/kaggle/working/{tag}.json'], stdout=log, stderr=subprocess.STDOUT, env=env))
t0 = time.time()
while any(p.poll() is None for p in ps):
    time.sleep(600); print(f'{(time.time()-t0)/3600:.2f} h', [p.poll() for p in ps], flush=True)
    os.system('tail -n 2 /kaggle/working/*.log')
print('ALL_DONE', [p.returncode for p in ps])
