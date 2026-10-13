"""Local (Mac) posterior runs for the MTT comparison: MASCOT adaptive-Metropolis chain, then the exact
posterior by importance reweighting of thinned MASCOT draws with w = exp(ll_exact - ll_mascot) (same prior).

usage: OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python bench/mtt/run_local.py TREE [n_iter] [n_is]
output: bench/mtt/out/TREE_local.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

name = sys.argv[1]
tree = C.load_tree(name)
d = len(tree["Ne"])
n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else (20000 if d == 2 else 30000)
n_is = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
outdir = os.path.join(C.HERE, "out")
os.makedirs(outdir, exist_ok=True)

llm = C.make_loglik(tree, "mascot")
lle = C.make_loglik(tree, "exact")
th0 = C.theta_true(tree)
print(name, "d", d, "n_iter", n_iter, "start ll_mascot", llm(th0), "ll_exact", lle(th0), flush=True)

r = C.adaptive_metropolis(lambda t: llm(t) + C.log_prior(t, d), th0, n_iter=n_iter, seed=1, progress=5000)
S = r["samples"][r["adapt_end"]:]
print(f"mascot chain {r['seconds']:.0f}s accept {r['accept_after_adapt']:.3f} kept {len(S)}", flush=True)

idx = np.linspace(0, len(S) - 1, n_is).round().astype(int)
D = S[idx]
t0 = time.time()
l_ex = np.array([lle(t) for t in D])
l_ma = np.array([llm(t) for t in D])
t_is = time.time() - t0
lw = l_ex - l_ma
w = np.exp(lw - lw.max())
ess_w = w.sum() ** 2 / (w ** 2).sum()
print(f"IS {n_is} draws {t_is:.0f}s weight-ESS {ess_w:.0f} lw sd {lw.std():.3f}", flush=True)
np.savez(os.path.join(outdir, name + "_local.npz"), mascot=S, mascot_lp=r["logpost"][r["adapt_end"]:],
         is_draws=D, is_ll_exact=l_ex, is_ll_mascot=l_ma, is_w=w, mascot_seconds=r["seconds"],
         mascot_n_iter=r["n_iter"], mascot_adapt_end=r["adapt_end"], mascot_accept=r["accept_after_adapt"],
         is_seconds=t_is, names=np.array(C.param_names(d)), theta_true=th0)
print("saved", flush=True)
