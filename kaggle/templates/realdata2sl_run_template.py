# Kaggle kernel: exact quasi-Newton refinement (bench/realdata2_refine.py, NPROC=4) of one MASCOT optimum on the
# sisco or lisco AIV tree, then the paired h/2 TT check at the refined optimum. Job filled in by push.sh.
import glob, os, shutil, subprocess, time
src = os.path.dirname(glob.glob('/kaggle/input/**/realdata2_refine.py', recursive=True)[0])
W = '/kaggle/working/esco'; O = '/kaggle/working'
for d in ['dense', 'approx', 'tt', 'bench', 'data']: os.makedirs(f'{W}/{d}', exist_ok=True)
shutil.copy(f'{src}/esco_dense.py', f'{W}/dense/'); shutil.copy(f'{src}/mascot.py', f'{W}/approx/')
shutil.copy(f'{src}/esco_tt.py', f'{W}/tt/'); shutil.copy(f'{src}/ttcore.py', f'{W}/tt/'); open(f'{W}/tt/__init__.py', 'w').close()
for f in ['realdata_fit.py', 'realdata2_eval.py', 'realdata2_refine.py']: shutil.copy(f'{src}/{f}', f'{W}/bench/')
for p in glob.glob(f'{src}/*.json'): shutil.copy(p, f'{W}/data/')
os.system('nproc; free -g')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NPROC='4', TT_EPS="1e-6", ITERS="8")
M, TAG = __TREE__, __TAG__
T = f'{W}/data/aivorder_{M}_mcc.json'
out = f'{O}/{M}_refine_{TAG}.json'
t0 = time.time()
with open(f'{O}/refine.log', 'w') as log:
    rc = subprocess.call(['python', '-u', f'{W}/bench/realdata2_refine.py', T, f'{W}/data/{M}_mascot_{TAG}.json', out],
                         stdout=log, stderr=subprocess.STDOUT, env=env)
print('refine rc', rc, f'{(time.time()-t0)/60:.1f} min', flush=True)
os.system(f'tail -n 5 {O}/refine.log')
fit = out.replace('.json', '_fit.json')
if os.path.exists(fit):
    with open(f'{O}/pair.log', 'w') as log:
        rc = subprocess.call(['python', '-u', f'{W}/bench/realdata2_eval.py', T, f'{O}/{M}_pair_h2_{TAG}.json', '1e-6h2', fit],
                             stdout=log, stderr=subprocess.STDOUT, env=env)
    print('pair rc', rc, f'{(time.time()-t0)/60:.1f} min', flush=True)
    os.system(f'tail -n 2 {O}/pair.log')
print('ALL_DONE', flush=True)
