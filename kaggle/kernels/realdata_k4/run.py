import glob, os, shutil, subprocess, time
src = os.path.dirname(glob.glob('/kaggle/input/**/realdata_fit.py', recursive=True)[0])
W = '/kaggle/working/esco'; O = '/kaggle/working'
for d in ['dense', 'approx', 'tt', 'bench', 'data']: os.makedirs(f'{W}/{d}', exist_ok=True)
shutil.copy(f'{src}/esco_dense.py', f'{W}/dense/'); shutil.copy(f'{src}/mascot.py', f'{W}/approx/')
shutil.copy(f'{src}/esco_tt.py', f'{W}/tt/'); shutil.copy(f'{src}/ttcore.py', f'{W}/tt/'); open(f'{W}/tt/__init__.py', 'w').close()
shutil.copy(f'{src}/realdata_fit.py', f'{W}/bench/'); shutil.copy(f'{src}/realdata_ttcheck.py', f'{W}/bench/')
[shutil.copy(p, f'{W}/data/') for p in glob.glob(f'{src}/h3n2_*.json')]; shutil.copy(f'{src}/realdata_multistart.py', f'{W}/bench/')
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
for f in glob.glob(f'{src}/post*_sym.json'): shutil.copy(f, O)
for i in [150, 205, 280, 390]:
    P = f'{W}/data/h3n2_post{i}.json'
    for eng in ['mascot', 'dense']:
        run([F, 'fit', eng, 'asym', P, f'{O}/post{i}_{eng}_asymB.json', f'{O}/post{i}_{eng}_sym.json'], f'post{i}_{eng}_asymB')
print('ALL_DONE', flush=True)
