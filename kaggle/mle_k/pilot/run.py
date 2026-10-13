import glob, os, subprocess, time
src = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/bench/mle_fit.py', recursive=True)[0]))
os.system(f'nproc; ls -R {src} | head -30')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NPROC='4')
p = subprocess.Popen(['python', '-u', f'{src}/bench/mle_fit.py', f'{src}/trees/d4_k50_slow_r39.json', '/kaggle/working/results.jsonl'], stdout=open('/kaggle/working/p.log', 'w'), stderr=subprocess.STDOUT, env=env)
t0 = time.time()
while p.poll() is None and time.time() - t0 < 900: time.sleep(5)
p.kill(); os.system('cat /kaggle/working/p.log | tail -40; cat /kaggle/working/results.jsonl')
