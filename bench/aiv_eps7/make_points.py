"""AIV 핵심 봉우리 eps 1e-7 확인용 점 파일 만들기 (2026-10-13, 심사 대응 레인).
정제된 모수 그대로 fit JSON(engine/model/theta/loglik)을 points/<id>.json으로 쓴다.
'최신 점'은 정제 JSON에 저장된 마지막(최고) 반복점. 그 뒤 줄 탐색 시도점은 theta가 저장되지 않아 쓰지 않음.
실행: esco/.venv/bin/python bench/aiv_eps7/make_points.py"""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(H, "..", "realdata2_sl")
os.makedirs(os.path.join(H, "points"), exist_ok=True)
P = {  # id: (tree, source file, kind)
    "masco_asym_hubans": ("masco", "masco_refine_asym_hubans_fit.json", "fit"),
    "masco_sym_hub": ("masco", "masco_refine_sym_hubcont_fit.json", "fit"),
    "masco_sym_regular": ("masco", "masco_refine_sym_regular_fit.json", "fit"),
    "sisco_asym_gal": ("sisco", "sisco_refine_asym_L264705_fit.json", "fit"),
    "sisco_asym_cha": ("sisco", "sisco_refine_asym_L264670cont.json", "refine"),
    "sisco_sym_regular": ("sisco", "sisco_refine_sym_L274067cont_fit.json", "fit"),
    "lisco_asym_gal": ("lisco", "lisco_refine_asym_L257131_fit.json", "fit"),
    "lisco_asym_cha": ("lisco", "lisco_refine_asym_L256886cont_fit.json", "fit"),
    "lisco_sym_regular": ("lisco", "lisco_refine_sym_L267082cont.json", "refine"),
}
idx = {}
for k, (tree, f, kind) in P.items():
    j = json.load(open(os.path.join(S, f)))
    if kind == "refine":
        b = j["best"]
        out = {"engine": "tt", "model": j["model"], "theta": b["theta"], "loglik": b["tt"], "src": f + f" (iter {b['it']})"}
    else:
        out = {"engine": "tt", "model": j["model"], "theta": j["theta"], "loglik": j["loglik"], "src": f}
    json.dump(out, open(os.path.join(H, "points", k + ".json"), "w"), indent=1)
    idx[k] = {"tree": tree, "src": out["src"], "tt_1e-6": out["loglik"]}
json.dump(idx, open(os.path.join(H, "points_index.json"), "w"), indent=1)
print(json.dumps(idx, indent=1))
