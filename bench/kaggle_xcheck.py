"""TT (eps 1e-6, 1e-7) and MASCOT vs Kaggle dense on peak-13/14 trees; MASCOT on large headline trees."""
import sys, os, json, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tt import esco_tt
from approx import mascot
f, eps = sys.argv[1], float(sys.argv[2])
p = f if os.path.exists(f) else os.path.join(os.path.dirname(__file__), "..", "sim/data", f)
tree = json.load(open(p)); t0 = time.time()
ll = mascot.loglik(tree) if eps == 0 else esco_tt.loglik_detail(tree, eps=eps)["loglik"]
print(f"{os.path.basename(f)},{eps},{ll!r},{time.time()-t0:.1f}", flush=True)
