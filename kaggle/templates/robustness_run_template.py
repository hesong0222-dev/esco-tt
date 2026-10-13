import glob, os, subprocess
# dataset v2: plain folder esco/ (Kaggle auto-extracts archives, so v1's esco_rb.tar.gz was never visible as a file)
work = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/bench/robustness.py', recursive=True)[0]))
print('code root', work, flush=True)
os.system('nproc; free -g | head -2; python -c "import numpy; numpy.show_config()" | grep -A1 -i "lapack\\|blas" | head -6')
jobs = """__JOBS__""".split()
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NPROC='4')
subprocess.run(['python', '-u', work + '/bench/robustness.py', '/kaggle/working/out.csv'] + jobs, env=env)
print('ALL_DONE', flush=True)
