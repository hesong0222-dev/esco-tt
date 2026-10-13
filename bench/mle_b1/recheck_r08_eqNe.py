"""d4_k50_slow_r08 eqNe 첫 시도점(로그 f=1057.92) 재현과 재평가 (2026-10-13, B1 점검).
v1 eqNe 시작점 x0 = (log m̂, mean log N̂e)를 TT fit에서 만들고, 중앙차분 h=0.05 기울기(5점)를 다시 계산해 로그 f=97.609105,
|g|=3.31과 맞는지 확인한 뒤, L-BFGS-B 첫 시도점(scipy와 같은 방식: x0에서 −g 방향, 처음 단계 길이 1/‖g‖ → 거리 1)을
현재 코드(B1 수정)와 v1 사본(kaggle/mle_ds/tt) 둘 다로 평가한다. 실제로 scipy를 같은 함수로 돌려 첫 시도점을 얻는다.
실행: OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python bench/mle_b1/recheck_r08_eqNe.py"""
import importlib.util, json, os, sys, time
import numpy as np
from scipy.optimize import minimize
E = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, E); sys.path.insert(0, os.path.join(E, "bench"))
import mle_common as C
from tt import esco_tt as new_tt

def load_old():
    pkg = os.path.join(E, "kaggle", "mle_ds")
    sys.path.insert(0, pkg)
    spec = importlib.util.spec_from_file_location("old_tt.ttcore", os.path.join(pkg, "tt", "ttcore.py"))
    return pkg
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recheck_r08_eqNe.json")
tag = "d4_k50_slow_r08"
tree = json.load(open(os.path.join(E, "bench", "mle", "trees", tag + ".json")))
recs = [json.loads(l) for l in open(os.path.join(E, "kaggle", "out", "mle", "kaggle-d4k50s-c", "results.jsonl")) if l.strip()]
fit = [r for r in recs if r["tree"] == tag and r["engine"] == "tt" and r["stage"] == "fit"][0]
th = np.array(fit["theta"]); d = 4
x0 = np.array([th[0], float(np.mean(th[1:]))])
full = lambda z: np.array([z[0]] + [z[1]] * d)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def ev(z, key):
    if key in res: return res[key]["ll"]
    t0 = time.time(); r = new_tt.loglik_detail(C.with_theta(tree, full(z)), eps=1e-6)
    res[key] = {"z": list(map(float, z)), "ll": r["loglik"], "rank": r["max_rank"], "sec": time.time() - t0}
    json.dump(res, open(OUT, "w"), indent=1); print(key, res[key], flush=True); return r["loglik"]
h = 0.05
seen = []
def fg(z):
    zs = [z] + [z + s * h * np.eye(2)[i] for i in range(2) for s in (1, -1)]
    k = len(seen); seen.append(list(map(float, z)))
    lls = [ev(x, f"call{k}_pt{j}") for j, x in enumerate(zs)]
    f = -np.array(lls); g = (f[1::2] - f[2::2]) / (2 * h)
    print(f"call {k}: f={f[0]:.6f} |g|={np.linalg.norm(g):.3e} z={z}", flush=True)
    if k >= 1: raise StopIteration
    return f[0], g
try:
    minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=[(-9.0, 6.0)] * 2, options={"maxiter": 60, "ftol": 1e-10, "gtol": 0.02})
except StopIteration:
    pass
z1 = np.array(seen[1]); Ne1 = np.exp(z1[1])
print("trial point z1", z1, "Ne", Ne1, "m", np.exp(z1[0]), flush=True)
res["trial"] = {"z": list(map(float, z1)), "minNe": float(Ne1)}
json.dump(res, open(OUT, "w"), indent=1)
# 같은 점을 v1 사본으로 (B1 수정 전)
sys.path.insert(0, os.path.join(E, "kaggle", "mle_ds"))
for m in [k for k in list(sys.modules) if k == "tt" or k.startswith("tt.")]: del sys.modules[m]
import importlib; old_tt = importlib.import_module("tt.esco_tt")
assert "mle_ds" in old_tt.__file__, old_tt.__file__
if "trial_old" not in res:
    t0 = time.time(); r = old_tt.loglik_detail(C.with_theta(tree, full(z1)), eps=1e-6)
    res["trial_old"] = {"ll": r["loglik"], "rank": r["max_rank"], "sec": time.time() - t0, "file": old_tt.__file__}
    json.dump(res, open(OUT, "w"), indent=1)
print("old", res["trial_old"], "new", res.get("call1_pt0"), flush=True)
