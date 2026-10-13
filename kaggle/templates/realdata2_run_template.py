import glob, os, shutil, subprocess, time
src = os.path.dirname(glob.glob('/kaggle/input/**/realdata2_eval.py', recursive=True)[0])
W = '/kaggle/working/esco'; O = '/kaggle/working'
for d in ['dense', 'approx', 'tt', 'bench', 'data']: os.makedirs(f'{W}/{d}', exist_ok=True)
shutil.copy(f'{src}/esco_dense.py', f'{W}/dense/'); shutil.copy(f'{src}/mascot.py', f'{W}/approx/')
shutil.copy(f'{src}/esco_tt.py', f'{W}/tt/'); shutil.copy(f'{src}/ttcore.py', f'{W}/tt/'); open(f'{W}/tt/__init__.py', 'w').close()
for f in ['realdata_fit.py', 'realdata2_eval.py', 'realdata2_refine.py']: shutil.copy(f'{src}/{f}', f'{W}/bench/')
for p in glob.glob(f'{src}/aivorder_*.json'): shutil.copy(p, f'{W}/data/')
os.system('nproc; free -g')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
T = f'{W}/data/aivorder_masco_mcc.json'; E = f'{W}/bench/realdata2_eval.py'
JOBS = __JOBS__   # list of (tag, eps spec, fit json name)
procs = []
for tag, spec, fit in JOBS:
    log = open(f'{O}/{tag}.log', 'w')
    procs.append((tag, time.time(), subprocess.Popen(['python', '-u', E, T, f'{O}/{tag}.json', spec, f'{W}/data/{fit}'],
                                                  stdout=log, stderr=subprocess.STDOUT, env=env)))
for tag, t0, p in procs:
    rc = p.wait()
    print(tag, 'rc', rc, f'{(time.time()-t0)/60:.1f} min', flush=True)
    os.system(f'tail -n 2 {O}/{tag}.log')
print('ALL_DONE', flush=True)
