# OpenBLAS 재계산: kaggle/xcheck.csv의 Mac(Accelerate) TT 값을 Linux(OpenBLAS)에서 다시 계산. 작업당 1 스레드, 4개 병렬, 3시간 상한.
import glob, os, shutil, subprocess, time, numpy
print(numpy.__version__, numpy.show_config(mode='dicts')['Build Dependencies']['lapack']['name'], flush=True)
src = os.path.dirname(glob.glob('/kaggle/input/**/kaggle_xcheck.py', recursive=True)[0])
w = '/kaggle/working/esco'
for d in ('tt', 'approx'): shutil.copytree(f'{src}/{d}', f'{w}/{d}', dirs_exist_ok=True)
os.makedirs(f'{w}/bench', exist_ok=True); os.makedirs(f'{w}/sim/data', exist_ok=True)
shutil.copy(f'{src}/kaggle_xcheck.py', f'{w}/bench/')
for f in glob.glob(f'{src}/*.json'): shutil.copy(f, f'{w}/sim/data/')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
queue = [j.split() for j in "d4_k14_medium_r1.json 1e-7|d4_k18_medium_r0.json 1e-7|d4_k20_medium_r2.json 1e-7|d4_k14_medium_r1.json 1e-6|d4_k18_slow_r1.json 1e-6".split('|')]; running = []; t0 = time.time()
out = open('/kaggle/working/xcheck_openblas.csv', 'a')
while queue or running:
    while queue and len(running) < 4:
        j = queue.pop(0)
        running.append((j, time.time(), subprocess.Popen(['python', f'{w}/bench/kaggle_xcheck.py', *j], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=w)))
    time.sleep(20)
    for r in list(running):
        j, ts, p = r
        if p.poll() is not None:
            o = p.stdout.read().strip().splitlines(); line = o[-1] if p.returncode == 0 and o else f'{j[0]},{j[1]},ERROR rc={p.returncode} {o[-3:]}'
        elif time.time() - ts > 10800:
            p.kill(); p.wait(); line = f'{j[0]},{j[1]},TIMEOUT_3h,{time.time()-ts:.0f}'
        else:
            continue
        out.write(line + '\n'); out.flush(); print(f'{(time.time()-t0)/3600:.2f} h', line, flush=True); running.remove(r)
print('ALL_DONE', flush=True)
