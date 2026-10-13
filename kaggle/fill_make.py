"""Make a Kaggle CPU kernel dir that runs headline.py jobs, 4 at a time, longest first.
Usage: python3 fill_make.py <dir> <acct/kernel-slug> <acct/dataset> job [job ...]
job = tree:eps:steps_mult  (e.g. d3_k30_slow_r0:1e-6:2). Output per job: /kaggle/working/<tree>[_sm2|_eps1e-7].json + .log.
Each job's JSON is written as soon as it finishes; at 11.3 h the runner kills what is left and exits normally
so finished outputs and partial logs are saved (Kaggle limit 12 h)."""
import json, os, sys

d, ref, ds, *jobs = sys.argv[1:]
os.makedirs(d, exist_ok=True)
J = []
for j in jobs:
    t, eps, sm = j.split(":")
    tag = t + ("_sm2" if sm == "2" else "") + ("_eps" + eps if eps != "1e-6" else "")
    J.append([t, eps, sm, tag])
run = f'''import glob, os, shutil, subprocess, time
src = os.path.dirname(glob.glob('/kaggle/input/**/headline.py', recursive=True)[0])
work = '/kaggle/working/esco'; os.makedirs(work + '/bench', exist_ok=True)
if os.path.isdir(src + '/tt'): shutil.copytree(src + '/tt', work + '/tt', dirs_exist_ok=True)
else:
    shutil.unpack_archive(src + '/tt.zip', work + '/tt')
    if os.path.isdir(work + '/tt/tt'): shutil.copytree(work + '/tt/tt', work + '/tt', dirs_exist_ok=True)
shutil.copy(src + '/headline.py', work + '/bench/headline.py')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
queue = {json.dumps(J)}; running = []; t0 = time.time()
while queue or running:
    while queue and len(running) < 4 and time.time() - t0 < 10.5 * 3600:
        t, eps, sm, tag = queue.pop(0); log = open(f'/kaggle/working/{{tag}}.log', 'w')
        running.append((tag, subprocess.Popen(['python', '-u', work + '/bench/headline.py', f'{{src}}/{{t}}.json', eps, sm, f'/kaggle/working/{{tag}}.json'], stdout=log, stderr=subprocess.STDOUT, env=env)))
    if time.time() - t0 > 11.3 * 3600:
        for tag, p in running: p.kill(); print(f'KILLED {{tag}} at guard', flush=True)
        break
    if not running and time.time() - t0 >= 10.5 * 3600: break
    time.sleep(30)
    for tag, p in list(running):
        if p.poll() is not None:
            print(f'{{(time.time()-t0)/3600:.2f}} h done {{tag}} rc={{p.returncode}}', flush=True); running.remove((tag, p))
print('ALL_DONE' if not queue and not running else f'STOPPED queue_left={{[q[3] for q in queue]}}', flush=True)
'''
open(f"{d}/run.py", "w").write(run)
meta = dict(id=ref, title=ref.split("/")[1].replace("-", " "), code_file="run.py", language="python",
            kernel_type="script", is_private=True, enable_gpu=False, enable_internet=False,
            dataset_sources=[ds], competition_sources=[], kernel_sources=[])
json.dump(meta, open(f"{d}/kernel-metadata.json", "w"))
open(f"{d}/jobs.txt", "w").write("\n".join(jobs) + "\n")
print(d, ref, len(J), "jobs")
