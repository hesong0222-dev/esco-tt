"""Collect bench/realdata2_sl/* (AIV H7, 3 bird-order demes, sisco and lisco MCC trees) into
bench/realdata2_sisco_lisco.csv and print the markdown tables of realdata2_sisco_lisco.md.
Rerun any time; unfinished TT work shows as blank cells."""
import csv
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "realdata2_sl")
DEMES = ["ans", "cha", "gal"]
ASYM = ["m01", "m02", "m10", "m12", "m20", "m21", "Ne0", "Ne1", "Ne2"]
SYM = ["m01", "m02", "m12", "Ne0", "Ne1", "Ne2"]


def jget(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def label(params, root):
    """Optimum name: 'hub' = a deme with Ne < 0.5 exchanging lineages at > 5/y; otherwise '<root deme> root'."""
    ne = [params[f"Ne{i}"] for i in range(3)]
    rates = [v for k, v in params.items() if k.startswith("m")]
    if min(ne) < 0.5 and max(rates) > 5:
        return f"hub ({DEMES[ne.index(min(ne))]} Ne {min(ne):.2f})"
    return f"{DEMES[root.index(max(root))]} root"


def rs(v):
    return "/".join(f"{x:.3f}" for x in v) if v else ""


rows = []
TREES = ["masco", "sisco", "lisco"]  # masco: the hub / cha-root / symmetric refinements of realdata2.md
for m in TREES:
    E = jget(os.path.join(R, f"{m}_eval.json"))
    P = jget(os.path.join(R, f"{m}_pair_h2.json"))
    pat = "_mascot_*.json" if m == "masco" else "_mascot_*_L*.json"
    for p in sorted(glob.glob(os.path.join(R, m + pat)), key=lambda p: -json.load(open(p))["loglik"]):
        if "_seed_" in p or "_x0" in p or "from_exact" in p or (
                p.endswith("cont.json") and os.path.exists(p.replace("cont.json", ".json"))):
            continue
        f = json.load(open(p))
        key = os.path.basename(p)
        e = E.get(key, {})
        tag = key.replace(f"{m}_mascot_", "").replace(".json", "")
        r0 = jget(os.path.join(R, f"{m}_refine_{tag}.json"))
        if "tt_1e-06" not in e and r0.get("iters"):  # exact value at the start point = refinement iteration 0
            e = {"tt_1e-06": r0["iters"][0]["tt"], "tt_1e-06_root": r0["iters"][0]["root"]}
        tt = e.get("tt_1e-06")
        rows.append({"tree": m, "kind": "MASCOT optimum", "point": tag, "model": f["model"],
                     "name": label(f["params"], f["root_probs"]), "n_starts": f.get("n_starts"),
                     "mascot_logL": f["loglik"], "exact_logL": tt,
                     "mascot_minus_exact": f["loglik"] - tt if tt is not None else None,
                     "root_mascot": rs(f["root_probs"]), "root_exact": rs(e.get("tt_1e-06_root")),
                     "tt_rank": e.get("tt_1e-06_rank"), "tt_sec": e.get("tt_1e-06_s"),
                     **{k: f["params"].get(k) for k in ASYM}})
        for rtag in [tag, tag + "cont"]:  # "<tag>cont" = continuation of a capped/stopped refinement
          r = jget(os.path.join(R, f"{m}_refine_{rtag}.json"))
          if r.get("iters"):
            b = r["best"]
            res = r.get("result", {})
            names = ASYM if r["model"] == "asym" else SYM
            prm = dict(zip(names, map(math.exp, b["theta"])))
            fitkey = f"{m}_refine_{rtag}_fit.json"
            pe = P.get(fitkey, {})
            rows.append({"tree": m, "kind": "exact optimum" if res else "exact (refinement unfinished)",
                         "point": "exact near " + rtag, "model": r["model"], "name": label(prm, b["root"]),
                         "mascot_logL": b["mascot"], "exact_logL": b["tt"], "mascot_minus_exact": b["mascot"] - b["tt"],
                         "root_exact": rs(b["root"]), "iters": len(r["iters"]),
                         "pred_gain_last": r["iters"][-1].get("pred_gain"),
                         "kkt": json.dumps({k: round(v, 3) for k, v in res.get("kkt_probe", {}).items()}) if res else "",
                         "tt_h2": pe.get("tt_1e-06_h2"),
                         "err_h2": abs(pe["tt_1e-06_h2"] - b["tt"]) if "tt_1e-06_h2" in pe else None,
                         **{k: prm.get(k) for k in ASYM}})

cols = ["tree", "kind", "point", "model", "name", "n_starts", "mascot_logL", "exact_logL", "mascot_minus_exact",
        "root_mascot", "root_exact", "tt_rank", "tt_sec", "iters", "pred_gain_last", "kkt", "tt_h2", "err_h2"] + ASYM
with open(os.path.join(HERE, "realdata2_sisco_lisco.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in cols})


def f3(v):
    return "" if v is None else f"{v:.3f}"


for m in TREES:
    for model in ["asym", "sym"]:
        print(f"\n### {m} {model}\n")
        print("| point | name | starts | MASCOT logL | exact logL | MASCOT − exact | root MASCOT | root exact |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["tree"] == m and r["model"] == model:
                print(f"| {r['point']}{'' if r['kind'] != 'exact (refinement unfinished)' else ' (unfinished)'} | "
                      f"{r['name']} | {r.get('n_starts') or ''} | {f3(r['mascot_logL'])} | {f3(r['exact_logL'])} | "
                      f"{f3(r['mascot_minus_exact'])} | {r.get('root_mascot', '')} | {r.get('root_exact', '')} |")
    print(f"\n### {m} exact optima: parameters, KKT, h/2 check\n")
    for r in rows:
        if r["tree"] == m and r["kind"] != "MASCOT optimum":
            print(r["point"], r["kind"], "iters", r["iters"], "pred_gain_last", r["pred_gain_last"], "KKT", r["kkt"],
                  "h2", r["tt_h2"], "err", r["err_h2"],
                  {k: round(r[k], 4) for k in ASYM if r.get(k) is not None})
