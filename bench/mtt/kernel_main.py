"""Kaggle kernel driver for the MTT comparison (bundled into kaggle/mtt/k/*/run.py by kaggle/mtt/build.py).

mode smoke: download BEAST 2.7.7 + MTT 8.3.0, install MTT, `packagemanager -list`, 2 short MTT chains on one tree,
            convention check (typed_density), tiny exact/MASCOT chains. Writes out/SMOKE_OK on success.
mode full : per tree: pilot (throughput), 2 MTT chains (time budget), 2 exact adaptive-Metropolis chains (dense
            likelihood), 1 MASCOT chain; all jobs in a process pool of os.cpu_count() workers.
Markers in /kaggle/working/out: NO_INTERNET, INSTALL_FAILED, SMOKE_OK, SMOKE_FAILED, FULL_DONE.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
TREEDIR = os.path.abspath(os.path.join(SRC, "..", "..", "sim", "data"))
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "out")
TMP = "/kaggle/tmp" if os.path.isdir("/kaggle") else "/tmp/mtt"
BEAST_URL = "https://github.com/CompEvol/beast2/releases/download/v2.7.7/BEAST.v2.7.7.Linux.x86.tgz"
MTT_URL = "https://github.com/tgvaughan/MultiTypeTree/releases/download/v8.3.0/MultiTypeTree.v8.3.0.zip"
BEAST = os.path.join(TMP, "beast", "bin", "beast")
T_START = time.time()


def log(*a):
    print(f"[{time.time()-T_START:7.0f}s]", *a, flush=True)


def mark(name, text=""):
    open(os.path.join(OUT, name), "w").write(text)


def install():
    os.makedirs(TMP, exist_ok=True)
    try:
        for url in (BEAST_URL, MTT_URL):
            dst = os.path.join(TMP, os.path.basename(url))
            log("download", url)
            urllib.request.urlretrieve(url, dst)
            log("  size", os.path.getsize(dst))
    except Exception as e:  # noqa: BLE001
        mark("NO_INTERNET", repr(e))
        log("download failed", e)
        return False
    try:
        with tarfile.open(os.path.join(TMP, os.path.basename(BEAST_URL))) as tf:
            tf.extractall(TMP)
        pkg = os.path.expanduser("~/.beast/2.7/MultiTypeTree")
        os.makedirs(pkg, exist_ok=True)
        with zipfile.ZipFile(os.path.join(TMP, os.path.basename(MTT_URL))) as zf:
            zf.extractall(pkg)
        log("MTT package files:", sorted(os.listdir(pkg)))
        pm = subprocess.run([os.path.join(TMP, "beast", "bin", "packagemanager"), "-list"],
                            capture_output=True, text=True, timeout=600)
        open(os.path.join(OUT, "packagemanager_list.txt"), "w").write(pm.stdout + "\n--- stderr ---\n" + pm.stderr)
        lines = [l for l in pm.stdout.splitlines() if "MultiTypeTree" in l]
        log("packagemanager -list MTT rows:", lines)
        jv = subprocess.run([os.path.join(TMP, "beast", "jre", "bin", "java"), "-version"], capture_output=True, text=True)
        log("java:", (jv.stderr or jv.stdout).strip().splitlines()[:1])
        if not any("8.3.0" in l for l in lines):
            mark("INSTALL_FAILED", "MTT 8.3.0 not listed as installed\n" + pm.stdout[-3000:])
            return False
    except Exception as e:  # noqa: BLE001
        mark("INSTALL_FAILED", repr(e))
        log("install failed", e)
        return False
    return True


def run_beast(tree, tag, seed, chain_length, log_every, tree_every, timeout_s):
    d = os.path.join(OUT, f"{tree}_{tag}")
    os.makedirs(d, exist_ok=True)
    x = open(os.path.join(SRC, "xml", tree + ".xml")).read()
    x = x.replace("CHAINLENGTH", str(int(chain_length))).replace("LOGEVERY", str(int(log_every))).replace(
        "TREEEVERY", str(int(tree_every)))
    xp = os.path.join(d, tree + ".xml")
    open(xp, "w").write(x)
    t0 = time.time()
    with open(os.path.join(d, "stdout.txt"), "w") as fo:
        try:
            p = subprocess.run([BEAST, "-seed", str(seed), "-overwrite", "-threads", "1", xp], cwd=d, stdout=fo,
                               stderr=subprocess.STDOUT, timeout=timeout_s)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    el = time.time() - t0
    info = {"tree": tree, "tag": tag, "seed": seed, "chain_length": chain_length, "log_every": log_every,
            "seconds": el, "returncode": rc}
    json.dump(info, open(os.path.join(d, "run_info.json"), "w"), indent=1)
    return info


def conv_check(tree, tag):
    import typed_density as T
    d = os.path.join(OUT, f"{tree}_{tag}")
    try:
        n, w, wt = T.check(os.path.join(d, tree + ".log"), os.path.join(d, tree + ".trees"),
                           os.path.join(TREEDIR, tree + ".json"))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    return {"n_states": n, "max_abs_density_diff": w, "max_coal_time_err": wt}


def am_job(tree, engine, seed, seconds=None, n_iter=None):
    import numpy as np
    import common as C
    t = json.load(open(os.path.join(TREEDIR, tree + ".json")))
    d = len(t["Ne"])
    ll = C.make_loglik(t, engine)
    r = C.adaptive_metropolis(lambda th: ll(th) + C.log_prior(th, d), C.theta_true(t), n_iter=n_iter,
                              seconds=seconds, seed=seed)
    fn = os.path.join(OUT, f"{tree}_{engine}_s{seed}.npz")
    np.savez(fn, samples=r["samples"], logpost=r["logpost"], adapt_end=r["adapt_end"], n_iter=r["n_iter"],
             seconds=r["seconds"], accept=r["accept_after_adapt"], names=np.array(C.param_names(d)))
    return {"tree": tree, "engine": engine, "seed": seed, "n_iter": r["n_iter"], "seconds": r["seconds"],
            "accept": r["accept_after_adapt"]}


def job(spec):
    kind = spec[0]
    try:
        if kind == "mtt":
            _, tree, tag, seed, cl, le, te, to = spec
            info = run_beast(tree, tag, seed, cl, le, te, to)
            info["conv"] = conv_check(tree, tag)
            return info
        _, tree, engine, seed, secs = spec
        return am_job(tree, engine, seed, seconds=secs)
    except Exception as e:  # noqa: BLE001
        return {"spec": list(map(str, spec)), "error": repr(e)}


def pilot_rate(tree, steps=200000):
    """steps/s from two runs (1000 and `steps` steps) so that JVM start-up cancels."""
    a = run_beast(tree, "pilot0", 1, 1000, 100, 1000, 1800)
    b = run_beast(tree, "pilot", 1, steps, max(steps // 1000, 1), steps, 3600)
    return (steps - 1000) / max(b["seconds"] - a["seconds"], 1.0), {"small": a, "big": b}


def main(mode):
    os.makedirs(OUT, exist_ok=True)
    if not install():
        return
    import common as C
    ncpu = os.cpu_count() or 1
    log("cpus", ncpu)
    if mode == "smoke":
        tree = C.TREES[0]
        res = []
        for s in (1, 2):
            info = run_beast(tree, f"smoke_c{s}", s, 20000, 20, 200, 1800)
            info["conv"] = conv_check(tree, f"smoke_c{s}")
            log(info)
            res.append(info)
        res.append(am_job(tree, "exact", 1, n_iter=200))
        res.append(am_job(tree, "mascot", 1, n_iter=200))
        log(res[-2:])
        json.dump(res, open(os.path.join(OUT, "smoke_summary.json"), "w"), indent=1)
        ok = all(r.get("returncode") == 0 and "error" not in r["conv"] and r["conv"]["n_states"] > 10
                 and r["conv"]["max_abs_density_diff"] < 1e-6 and r["conv"]["max_coal_time_err"] < 1e-9
                 for r in res[:2])
        mark("SMOKE_OK" if ok else "SMOKE_FAILED", json.dumps(res, indent=1))
        return
    # full
    scale = min(1.0, ncpu / 4.0)
    B_MTT, B_EX, B_MA = 2.5 * 3600 * scale, 1.25 * 3600 * scale, 0.5 * 3600 * scale
    rates = {}
    for tree in C.TREES:
        r, info = pilot_rate(tree)
        rates[tree] = r
        log("pilot", tree, f"{r:.0f} steps/s", info)
    jobs = []
    for tree in C.TREES:
        cl = int(rates[tree] * B_MTT * 0.85)  # leave slack for load from the other pool workers
        le = max(cl // 20000, 1)
        for s in (1, 2):
            jobs.append(("mtt", tree, f"mtt_c{s}", 100 + s, cl, le, le * 20, int(B_MTT * 1.3)))
    for tree in C.TREES:
        for s in (1, 2):
            jobs.append(("am", tree, "exact", s, B_EX))
        jobs.append(("am", tree, "mascot", 1, B_MA))
    log("jobs", len(jobs), "budgets h", B_MTT / 3600, B_EX / 3600, B_MA / 3600)
    results = []
    with ProcessPoolExecutor(max_workers=ncpu) as ex:
        for r in ex.map(job, jobs):
            log(r)
            results.append(r)
            json.dump({"rates": rates, "results": results}, open(os.path.join(OUT, "full_summary.json"), "w"), indent=1)
    mark("FULL_DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke")
    log("end")
