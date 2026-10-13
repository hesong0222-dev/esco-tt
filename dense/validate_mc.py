"""Extended independent Monte Carlo check (vectorized estimator in validate.mc_loglik_vec)."""
import glob, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import esco_dense as ed, validate as v

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
files = sorted(glob.glob(os.path.join(HERE, "..", "sim", "data", "d[234]_k0[46]_*_r[01].json")))
rng = np.random.default_rng(2026)
print(f"N={N} paths per tree")
print(f"{'tree':22s} {'L_dense':>12s} {'L_MC':>12s} {'relSE':>8s} {'z':>6s}")
zs = []
for f in files:
    t = json.load(open(f))
    L = math.exp(ed.loglik(t))
    t0 = time.time()
    m, se = v.mc_loglik_vec(t, N, rng)
    z = (m - L) / se
    zs.append(z)
    print(f"{os.path.basename(f)[:-5]:22s} {L:12.5e} {m:12.5e} {se/L:8.2e} {z:+6.2f}  ({time.time()-t0:.0f}s)", flush=True)
zs = np.array(zs)
print(f"n={len(zs)} mean z={zs.mean():+.3f} sd z={zs.std(ddof=1):.3f} sum z^2={np.sum(zs**2):.1f} max|z|={np.abs(zs).max():.2f}")
