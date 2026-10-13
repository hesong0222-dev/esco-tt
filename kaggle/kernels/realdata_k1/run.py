import glob, os, shutil, subprocess, time
src = os.path.dirname(glob.glob('/kaggle/input/**/realdata_fit.py', recursive=True)[0])
W = '/kaggle/working/esco'; O = '/kaggle/working'
for d in ['dense', 'approx', 'tt', 'bench', 'data']: os.makedirs(f'{W}/{d}', exist_ok=True)
shutil.copy(f'{src}/esco_dense.py', f'{W}/dense/'); shutil.copy(f'{src}/mascot.py', f'{W}/approx/')
shutil.copy(f'{src}/esco_tt.py', f'{W}/tt/'); shutil.copy(f'{src}/ttcore.py', f'{W}/tt/'); open(f'{W}/tt/__init__.py', 'w').close()
shutil.copy(f'{src}/realdata_fit.py', f'{W}/bench/'); shutil.copy(f'{src}/realdata_ttcheck.py', f'{W}/bench/')
shutil.copy(f'{src}/h3n2_mcc.json', f'{W}/data/')
for f in ['mascot_sym.json', 'mascot_asym.json']: shutil.copy(f'{src}/{f}', O)
os.system('nproc; free -g')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NPROC='4')
T = f'{W}/data/h3n2_mcc.json'; F = f'{W}/bench/realdata_fit.py'
def run(args, tag):
    t0 = time.time()
    with open(f'{O}/{tag}.log', 'w') as log:
        rc = subprocess.call(['python', '-u'] + args, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(tag, 'rc', rc, f'{(time.time()-t0)/60:.1f} min', flush=True)
    os.system(f'tail -n 3 {O}/{tag}.log')
run([F, 'fit', 'dense', 'sym', T, f'{O}/dense_sym.json', f'{O}/mascot_sym.json'], 'dense_sym')
run([F, 'fit', 'dense', 'asym', T, f'{O}/dense_asym_a.json', f'{O}/mascot_asym.json'], 'dense_asym_a')
run([F, 'fit', 'dense', 'asym', T, f'{O}/dense_asym_b.json', f'{O}/dense_sym.json'], 'dense_asym_b')
fits = [f'{O}/{x}.json' for x in ['mascot_sym', 'mascot_asym', 'dense_sym', 'dense_asym_a', 'dense_asym_b'] if os.path.exists(f'{O}/{x}.json')]
run([f'{W}/bench/realdata_ttcheck.py', T, f'{O}/ttcheck.json'] + fits, 'ttcheck')
print('ALL_DONE', flush=True)
