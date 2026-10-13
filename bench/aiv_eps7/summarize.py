"""AIV 봉우리 eps 1e-7 확인 요약 (2026-10-13). 입력: kaggle/out/aiveps7/*/eps7_<점>.json, bench/aiv_eps7/points_index.json,
masco gal 뿌리는 기존 bench/realdata2/aivorder_pair_B_e7.json(같은 정제점, 2026-10-06 계산).
출력: bench/aiv_eps7/eps7_results.csv, 표준 출력. 실행: .venv/bin/python bench/aiv_eps7/summarize.py"""
import csv, glob, json, os
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(H); E = os.path.dirname(B)
idx = json.load(open(os.path.join(H, "points_index.json")))
R = {}
for f in glob.glob(os.path.join(E, "kaggle", "out", "aiveps7", "*", "eps7_*.json")):
    for k, v in json.load(open(f)).items():
        R[k.replace(".json", "")] = v
g = json.load(open(os.path.join(B, "realdata2", "aivorder_pair_B_e7.json")))["aivorder_exact_asym_B.json"]
idx["masco_asym_gal"] = {"tree": "masco", "src": "realdata2/aivorder_refine_asym_best_fit.json", "tt_1e-6": g["fit_loglik"]}
R["masco_asym_gal"] = g
rows = []
for k, v in idx.items():
    r = R.get(k, {})
    e7 = r.get("tt_1e-07")
    rows.append(dict(point=k, tree=v["tree"], tt_1e6=v["tt_1e-6"], tt_1e7=e7, diff=(e7 - v["tt_1e-6"]) if e7 is not None else None,
                     rank7=r.get("tt_1e-07_rank"), sec7=r.get("tt_1e-07_s"), mascot=r.get("mascot"),
                     root7=r.get("tt_1e-07_root")))
with open(os.path.join(H, "eps7_results.csv"), "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
D = {r["point"]: r for r in rows}
for r in rows:
    print(r["point"], r["tt_1e6"], r["tt_1e7"], r["diff"], r["rank7"], r["sec7"], [round(x, 4) for x in r["root7"]] if r["root7"] else None)
def v(k, e): return D[k][e]
for t, best, other, sym in [("masco", "masco_asym_hubans", "masco_asym_gal", "masco_sym_hub"),
                            ("sisco", "sisco_asym_gal", "sisco_asym_cha", "sisco_sym_regular"),
                            ("lisco", "lisco_asym_gal", "lisco_asym_cha", "lisco_sym_regular")]:
    for e in ("tt_1e6", "tt_1e7"):
        try:
            print(t, e, "gap", round(v(best, e) - v(other, e), 4), "LRT", round(2 * (v(best, e) - v(sym, e)), 4))
        except TypeError:
            print(t, e, "incomplete")
