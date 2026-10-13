"""Exact (TT) ML estimate refined from the MASCOT optimum, for trees where a full TT optimisation is too costly.

Each TT evaluation costs minutes on the 133-tip, 5-deme AIV tree, so the exact optimum is found by quasi-Newton
steps that use
  * the MASCOT Hessian at the MASCOT MLE (cheap, finite differences) as the curvature model, and
  * the exact gradient written as  g_exact = g_mascot + grad(Delta),  Delta = logL_TT - logL_MASCOT,
    where g_mascot uses a tiny step (MASCOT is smooth) and grad(Delta) uses forward differences with step H_DELTA.
    Delta is much flatter than logL itself, so a large step is accurate and swamps the TT rounding noise.
Free set: rates whose MASCOT MLE is above the lower bound (log > FREE_LOG) and all Ne. Rates at the bound stay at
the bound; afterwards a KKT check re-evaluates the exact logL with each bound rate raised to M_PROBE.

Usage: realdata2_refine.py <tree.json> <mascot_fit.json> <out.json>
Env: NPROC, TT_EPS (default 1e-6), ITERS (default 4), H_DELTA (default 0.1), M_PROBE (default 0.02), KKT (1/0).
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool

import realdata_fit as rf

LO, HI = -12.0, 7.0
FREE_LOG = -11.0
H_DELTA = float(os.environ.get("H_DELTA", "0.1"))
H_M = 1e-4  # MASCOT gradient step
H_HESS = 1e-2  # MASCOT Hessian step


def ev(args):
    engine, model, theta = args
    return rf.evaluate((engine, model, list(theta)))


def main():
    tree_path, fit_path, out_path = sys.argv[1:4]
    rf._init(tree_path)
    f0 = json.load(open(fit_path))
    model = f0["model"]
    names = rf.names(model)
    th0 = np.array(f0["theta"], float)
    free = [i for i, nm in enumerate(names) if nm.startswith("Ne") or th0[i] > FREE_LOG]
    bound = [i for i in range(len(names)) if i not in free]
    n = len(free)
    log = {"model": model, "tree": tree_path, "mascot_fit": fit_path, "free": [names[i] for i in free],
           "bound": [names[i] for i in bound], "eps": rf.TT_EPS, "h_delta": H_DELTA, "iters": []}
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", os.cpu_count())), initializer=rf._init, initargs=(tree_path,)) as pool:

        def at(z, base=None):
            th = (th0 if base is None else base).copy()
            th[free] = z
            return th

        # MASCOT Hessian on the free set at the MASCOT optimum (central differences)
        z0 = th0[free]
        E = np.eye(n)
        pts = [(i, j, si, sj) for i in range(n) for j in range(i, n) for si in (1, -1) for sj in (1, -1)]
        vals = pool.map(ev, [("mascot", model, at(z0 + H_HESS * (si * E[i] + sj * E[j]))) for i, j, si, sj in pts])
        Hm = np.zeros((n, n))
        for (i, j, si, sj), v in zip(pts, vals):
            Hm[i, j] += si * sj * v[0] / (4 * H_HESS ** 2)
        Hm = np.triu(Hm) + np.triu(Hm, 1).T  # Hessian of logL (negative definite at a max)
        w, V = np.linalg.eigh(-Hm)
        w = np.maximum(w, 1e-3 * max(w.max(), 1.0))
        B = V @ np.diag(w) @ V.T  # positive-definite curvature model of -logL
        log["mascot_hessian_eig"] = sorted(np.linalg.eigvalsh(-Hm).tolist())

        def exact_grad(z):
            """Exact logL, root, gradient (= MASCOT gradient + forward-difference gradient of Delta)."""
            th = at(z)
            jobs = [("tt", model, th)] + [("tt", model, at(z + H_DELTA * E[k])) for k in range(n)]
            jobs += [("mascot", model, th)] + [("mascot", model, at(z + H_DELTA * E[k])) for k in range(n)]
            jobs += [("mascot", model, at(z + H_M * E[k])) for k in range(n)]
            r = pool.map(ev, jobs)
            lt, root = r[0]
            lm = r[n + 1][0]
            dT = np.array([r[1 + k][0] for k in range(n)]) - lt
            dM = np.array([r[n + 2 + k][0] for k in range(n)]) - lm
            gM = (np.array([r[2 * n + 2 + k][0] for k in range(n)]) - lm) / H_M
            gD = (dT - dM) / H_DELTA
            return lt, lm, root, gM + gD, gM, gD

        z = z0.copy()
        best = None
        for it in range(int(os.environ.get("ITERS", "4"))):
            lt, lm, root, g, gM, gD = exact_grad(z)
            rec = {"it": it, "theta": at(z).tolist(), "tt": lt, "mascot": lm, "root": root,
                   "grad_norm": float(np.linalg.norm(g)), "grad_mascot_norm": float(np.linalg.norm(gM)),
                   "grad_delta_norm": float(np.linalg.norm(gD)), "sec": time.time() - t0}
            if best is None or lt > best["tt"]:
                best = rec
            step = np.linalg.solve(B, g)
            pred = float(g @ step) / 2
            rec["pred_gain"] = pred
            print(json.dumps({k: v for k, v in rec.items() if k != "theta"}), flush=True)
            log["iters"].append(rec)
            json.dump(dict(log, best=best), open(out_path, "w"), indent=1)
            if pred < 0.01:
                break
            # backtracking on the exact logL (one TT evaluation per trial)
            for a in (1.0, 0.5, 0.25):
                zn = np.clip(z + a * step, LO, HI)
                ln, rn = rf.evaluate(("tt", model, list(at(zn))))
                print(f"  line a={a}: tt={ln:.5f} (from {lt:.5f})", flush=True)
                if ln > lt:
                    break
            if ln <= lt:
                break
            z = zn

        lt, root = best["tt"], best["root"]
        zb = np.array(best["theta"])[free]
        res = {"engine": "tt", "model": model, "loglik": lt, "theta": best["theta"],
               "params": dict(zip(names, np.exp(best["theta"]).tolist())), "root_probs": root,
               "grad_norm": best["grad_norm"], "msg": "quasi-Newton from MASCOT optimum (realdata2_refine.py)",
               "fixed": None}
        # KKT check: raise each bound rate to M_PROBE
        if bound and os.environ.get("KKT", "1") == "1":
            mp = float(os.environ.get("M_PROBE", "0.02"))
            jobs = []
            for i in bound:
                th = np.array(best["theta"])
                th[i] = math.log(mp)
                jobs.append(("tt", model, th))
            r = pool.map(ev, jobs)
            res["kkt_probe"] = {names[i]: v[0] - lt for i, v in zip(bound, r)}
            print("KKT probe (logL change when the bound rate is set to", mp, "):", res["kkt_probe"], flush=True)
        res["seconds"] = time.time() - t0
        log["result"] = res
        json.dump(dict(log, best=best), open(out_path, "w"), indent=1)
        json.dump(res, open(out_path.replace(".json", "_fit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
