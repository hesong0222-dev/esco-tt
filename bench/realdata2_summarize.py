"""Collect bench/realdata2/* (AIV H7, 3 bird-order demes) into bench/realdata2.csv and print the tables of realdata2.md."""
import csv
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "realdata2")
DEMES = ["ans", "cha", "gal"]

rows = []


def add(point, model, mascot_ll, tt_ll, params, root_m, root_t, source, note=""):
    rows.append({"point": point, "model": model, "mascot_logL": mascot_ll, "exact_logL": tt_ll,
                 "mascot_minus_exact": (mascot_ll - tt_ll) if (mascot_ll is not None and tt_ll is not None) else None,
                 **{k: params.get(k) for k in ["m01", "m02", "m10", "m12", "m20", "m21", "Ne0", "Ne1", "Ne2"]},
                 "root_mascot": "/".join(f"{x:.3f}" for x in root_m) if root_m else "",
                 "root_exact": "/".join(f"{x:.3f}" for x in root_t) if root_t else "",
                 "source": source, "note": note})


def norm(v):
    if v is None:
        return None
    s = sum(max(x, 0) for x in v)
    return [max(x, 0) / s for x in v]


# MASCOT optima evaluated with the exact likelihood
for tag, fit, model in [("sym_hub", "aivorder_mascot_sym_hub.json", "sym"),
                        ("asym_A", "aivorder_mascot_asym_mode_A.json", "asym"),
                        ("asym_C", "aivorder_mascot_asym_mode_C.json", "asym"),
                        ("asym_ansroot", "aivorder_mascot_asym_ansroot.json", "asym")]:
    ev = os.path.join(R, f"aivorder_eval_{tag.split('_', 1)[1] if tag != 'sym_hub' else 'hub'}.json")
    ev = {"sym_hub": "aivorder_eval_hub.json", "asym_A": "aivorder_eval_mode_A.json",
          "asym_C": "aivorder_eval_mode_C.json", "asym_ansroot": "aivorder_eval_ansroot.json"}[tag]
    p = os.path.join(R, ev)
    if not os.path.exists(p):
        continue
    e = json.load(open(p))[fit]
    f = json.load(open(os.path.join(R, fit)))
    add("MASCOT optimum " + tag, model, e["mascot"], e.get("tt_1e-06"), f["params"], e["mascot_root"],
        norm(e.get("tt_1e-06_root")), ev, f"TT rank {e.get('tt_1e-06_rank')}, {e.get('tt_1e-06_s', 0):.0f} s")

# refinements (iteration 0 = MASCOT optimum; best = exact optimum near it)
for tag in ["asym_best", "sym_regular", "sym_hub"]:
    p = os.path.join(R, f"aivorder_refine_{tag}.json")
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    it0 = r["iters"][0]
    names = ["m01", "m02", "m10", "m12", "m20", "m21", "Ne0", "Ne1", "Ne2"] if r["model"] == "asym" else \
        ["m01", "m02", "m12", "Ne0", "Ne1", "Ne2"]
    f0 = json.load(open(r["mascot_fit"])) if os.path.exists(r["mascot_fit"]) else None
    if f0:
        add("MASCOT optimum " + tag, r["model"], f0["loglik"], it0["tt"], f0["params"], f0["root_probs"],
            norm(it0["root"]), os.path.basename(p), "iteration 0 of the refinement")
    b = r["best"]
    res = r.get("result", {})
    if not res and len(r["iters"]) == 1:
        rows[-1]["note"] += " (refinement stopped after it 0)"
        continue
    add("exact optimum near " + tag, r["model"], b["mascot"], b["tt"], dict(zip(names, map(math.exp, b["theta"]))),
        None, norm(b["root"]), os.path.basename(p),
        f"grad_norm {b['grad_norm']:.3f}, iters {len(r['iters'])}, " + ("KKT " + str({k: round(v, 3) for k, v in
                                                                                    res['kkt_probe'].items()}) if res.get('kkt_probe') else "unfinished" if not res else ""))

# paired TT error estimates
for f in sorted(glob.glob(os.path.join(R, "aivorder_pair_*.json"))):
    for key, e in json.load(open(f)).items():
        base = e.get("tt_1e-06")
        for tag, lab in [("tt_1e-06_h2", "h/2"), ("tt_1e-07", "eps 1e-7")]:
            if tag in e:
                rows.append({"point": f"TT error estimate at {key} ({lab})", "model": e["model"],
                             "mascot_logL": e["mascot"], "exact_logL": e[tag], "mascot_minus_exact": None,
                             "root_mascot": "", "root_exact": "", "source": os.path.basename(f),
                             "note": f"rank {e[tag + '_rank']}, {e[tag + '_s']:.0f} s, rmax_cuts {e.get(tag + '_rmax_cuts')}"})

out = os.path.join(HERE, "realdata2.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    if r["mascot_minus_exact"] is None and r["point"].startswith("TT error"):
        print(f"{r['point']:<70} {r['exact_logL']:.5f}  {r['note']}")
        continue
    print(f"{r['point']:<36} {r['model']:<4} MASCOT {r['mascot_logL']:.3f}  exact {r['exact_logL']:.3f}  "
          f"diff {r['mascot_minus_exact']:+.3f}  root M {r['root_mascot']:<18} E {r['root_exact']:<18} {r['note']}")
print("->", out)
