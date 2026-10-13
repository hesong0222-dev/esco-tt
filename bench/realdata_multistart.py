"""Multi-start check of the MASCOT ML optimum (cheap engine) to detect multiple local optima."""
import json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realdata_fit as rf

if __name__ == "__main__":
    tree, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
    engine = sys.argv[4] if len(sys.argv) > 4 else "mascot"
    nstart = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    rf._init(tree)
    npar = rf.npar(model)
    rng = np.random.default_rng(1)
    res = []
    with Pool(int(os.environ.get("NPROC", 6)), initializer=rf._init, initargs=(tree,)) as pool:
        rf._init(tree)
        for s in range(nstart):
            x0 = rng.uniform(-3, 1.5, npar)
            f = rf.fit(engine, model, tree, x0, pool=pool)
            print(s, round(f["loglik"], 5), {k: round(v, 4) for k, v in f["params"].items()}, [round(x, 3) for x in f["root_probs"]], flush=True)
            res.append(f)
    json.dump(res, open(out, "w"), indent=1)
