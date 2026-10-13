import csv
import math
import os

import numpy as np
import pytest

import escott
from escott import api, reweight as rw
from escott._engines import esco_dense, mascot
from escott._env import accelerate_warning
from escott.cli import main
from escott.io import load_tree

EX = os.path.join(os.path.dirname(__file__), "..", "examples")
REF = os.path.join(os.path.dirname(__file__), "data", "h3n2_posterior_ref.csv")
H3N2_DEMES = ["Hong_Kong", "New_York", "New_Zealand"]


def random_tree(n, d, seed):
    """Random heterochronous binary tree with random tip demes (any valid tree will do for a likelihood test)."""
    rng = np.random.default_rng(seed)
    ev = [{"t": 0.0 if i < n // 2 else float(rng.uniform(0, 0.5)), "type": "sample", "id": i,
           "deme": int(rng.integers(d))} for i in range(n)]
    ev.sort(key=lambda e: e["t"])
    alive, t, nid, out, pending = [], 0.0, n, [], list(ev)
    while pending or len(alive) > 1:
        t_next = t + float(rng.exponential(0.3))
        if pending and (pending[0]["t"] <= t_next or len(alive) < 2):
            e = pending.pop(0)
            t = e["t"]
            out.append(e)
            alive.append(e["id"])
            continue
        t = t_next
        a, b = rng.choice(len(alive), 2, replace=False)
        out.append({"t": t, "type": "coal", "a": alive[a], "b": alive[b], "new": nid})
        alive = [x for k, x in enumerate(alive) if k not in (a, b)] + [nid]
        nid += 1
    return {"demes": d, "events": out}


def random_params(d, seed):
    rng = np.random.default_rng(seed + 100)
    M = rng.uniform(0.1, 2.0, (d, d))
    np.fill_diagonal(M, 0)
    return M, rng.uniform(0.5, 2.0, d)


# ----------------------------------------------------------------------------- engines


@pytest.mark.parametrize("n,d,seed", [(5, 2, 1), (6, 3, 2), (6, 2, 3)])
def test_dense_and_tt_agree(n, d, seed):
    M, Ne = random_params(d, seed)
    tree = random_tree(n, d, seed)
    exact = escott.loglik(tree, M, Ne, method="dense")
    tt = escott.loglik(tree, M, Ne, method="tt", eps=1e-8)
    diff = abs(tt.loglik - exact.loglik)
    assert diff < 1e-5                      # at these sizes the TT error is time-step error, ~1e-6
    assert diff < 1.5 * tt.error + 1e-9     # and the paired-run estimate tracks it
    assert np.allclose(tt.info["root_dist"], exact.info["root_dist"], atol=1e-5)
    single = escott.loglik(tree, M, Ne, method="tt", eps=1e-8, paired=False)
    assert single.loglik == tt.loglik and "paired_diff" not in single.info


def test_root_recursions_reproduce_engines():
    """dense_root / mascot_root must give exactly the vendored engines' log-likelihoods."""
    M, Ne = random_params(3, 5)
    tree = api._as_tree(random_tree(7, 3, 5), M, Ne)
    ll, root = api.dense_root(tree)
    assert ll == esco_dense.loglik(tree)
    assert abs(sum(root) - 1) < 1e-12
    llm, rootm = api.mascot_root(tree)
    assert llm == mascot.loglik(tree)
    assert abs(sum(rootm) - 1) < 1e-12


def test_mascot_one_deme_is_kingman():
    tree = random_tree(8, 1, 7)
    Ne = 1.7
    ll, k, t = 0.0, 0, 0.0
    for e in tree["events"]:
        ll -= k * (k - 1) / 2 / Ne * (e["t"] - t)
        t = e["t"]
        if e["type"] == "sample":
            k += 1
        else:
            ll += math.log(1 / Ne)
            k -= 1
    m = escott.loglik(tree, [[0.0]], [Ne], method="mascot").loglik
    assert abs(m - ll) < 1e-8
    assert abs(escott.loglik(tree, [[0.0]], [Ne], method="dense").loglik - ll) < 1e-10


def _d1_tree(events):
    return {"demes": 1, "M": [[0.0]], "Ne": [1.0], "events": events}


def test_long_survival_no_underflow():
    """AUDIT B1/B3: survival < 1e-308 between coalescences. d=1, Ne=1: 0-400 one pair, 400-800 three pairs, coalesce,
    800-801 one pair, coalesce -> log L = -400 - 1200 - 1 = -1601 exactly (TT used to return -743.8, dense expm
    raised ValueError)."""
    S = lambda t, i: {"t": t, "type": "sample", "id": i, "deme": 0}
    C = lambda t, a, b, n: {"t": t, "type": "coal", "a": a, "b": b, "new": n}
    tree = _d1_tree([S(0, 0), S(0, 1), S(400, 2), C(800, 0, 1, 3), C(801, 3, 2, 4)])
    assert abs(escott.loglik(tree, method="dense").loglik + 1601) < 1e-8
    assert abs(escott.loglik(tree, method="tt", eps=1e-10).loglik + 1601) < 1e-6


def test_simulator_rejects_reducible_migration():
    """AUDIT B6: with a reducible M the simulator used to loop forever."""
    import sys
    esco = os.path.join(os.path.dirname(__file__), "..", "..")
    if not os.path.exists(os.path.join(esco, "sim", "simulate.py")):
        pytest.skip("research simulator not available")
    sys.path.insert(0, esco)
    from sim.simulate import simulate_tree
    M = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]          # deme 2 isolated
    with pytest.raises(ValueError, match="irreducible"):
        simulate_tree(M, [1, 1, 1], [0, 0], [0, 2], np.random.default_rng(0))
    ev = simulate_tree([[0, 1, 0], [0, 0, 1], [1, 0, 0]], [1, 1, 1], [0, 0], [0, 2], np.random.default_rng(0))
    assert ev[-1]["type"] == "coal"                  # a directed cycle is irreducible and still works


# ----------------------------------------------------------------------------- input


def test_tip_location_sources_agree(tmp_path):
    p = {"demes": H3N2_DEMES, "Ne": [1, 1, 1], "M": [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}
    path = os.path.join(EX, "H3N2.mcc.tree")
    by_label = load_tree(path, p)
    csvf = tmp_path / "tips.csv"
    csvf.write_text("tip,deme\n" + "".join(f"{lab},{lab.split('|')[-1]}\n" for lab in by_label["tip_labels"]))
    by_csv = load_tree(path, p, locations=escott.read_locations_csv(str(csvf)))
    by_annot = load_tree(path, p, location_key="max")
    assert by_label["events"] == by_csv["events"] == by_annot["events"]
    assert sum(e["type"] == "sample" for e in by_label["events"]) == 24
    assert by_label["location_source"].startswith("field")


def test_type_annotation_read_automatically(tmp_path):
    f = tmp_path / "t.tree"
    f.write_text("#NEXUS\nBegin trees;\ntree t1 = ((a[&type=\"X\"]:1,b[&type=\"Y\"]:1):1,c[&type=\"X\"]:2);\nEnd;\n")
    t = load_tree(str(f))
    assert t["deme_names"] == ["X", "Y"] and t["location_source"] == "tip annotation 'type'"
    assert [e["deme"] for e in t["events"] if e["type"] == "sample"] == [0, 1, 0]


def test_friendly_errors(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("Sample\tposterior\tNe.A\n0\t-1\t1\n")
    with pytest.raises(ValueError, match="NeConstant"):
        escott.params_from_beast_log(str(log))
    f = tmp_path / "t.tree"
    f.write_text("((a1:1,a2:1):1,(a3:1,(a4:1,(a5:1,(a6:1,(a7:1,(a8:1,(a9:1,(a10:1,a11:1):1):1):1):1):1):1):1):1);")
    with pytest.raises(ValueError, match="does not look like a deme assignment"):
        load_tree(str(f))
    with pytest.raises(SystemExit, match="not found in the locations file"):
        (tmp_path / "loc.csv").write_text("a1,X\n")
        main(["estimate", str(f), "--locations", str(tmp_path / "loc.csv")])


def test_accelerate_warning():
    assert accelerate_warning({"numpy": "scipy-openblas"}, "darwin") is None
    assert "Accelerate" in accelerate_warning({"numpy": "accelerate", "scipy": "scipy-openblas"}, "darwin")
    assert accelerate_warning({"numpy": "accelerate"}, "linux") is None


# ----------------------------------------------------------------------------- commands


def test_cli_check_h3n2(capsys):
    main(["check", os.path.join(EX, "H3N2.mcc.tree"), "--log", os.path.join(EX, "H3N2.log")])
    out = capsys.readouterr().out
    assert "exact  log L" in out and "MASCOT - exact" in out
    exact = float(out.split("exact  log L = ")[1].split()[0])
    masc = float(out.split("MASCOT log L = ")[1].split()[0])
    assert abs(exact + 38.475743) < 1e-5 and abs(masc + 37.609196) < 1e-5


def test_cli_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert escott.__version__ in capsys.readouterr().out


def test_cli_estimate(capsys, tmp_path):
    main(["estimate", os.path.join(EX, "H3N2.mcc.tree"), "--params", os.path.join(EX, "params_h3n2.json")])
    out = capsys.readouterr().out
    assert "3^12 = 531,441" in out and "dense) engine: feasible" in out and "N_round" in out
    assert "recommended eps: 1e-06" in out
    tree = random_tree(8, 6, 3)
    names = [f"D{i}" for i in range(6)]
    nwk = tmp_path / "t.json"
    import json
    nwk.write_text(json.dumps(dict(tree, deme_names=names)))
    main(["estimate", str(nwk)])
    assert "not feasible with the tensor train" in capsys.readouterr().out


def _states_of(samples):
    names, states, info = rw.load_states(os.path.join(EX, "H3N2.trees"), os.path.join(EX, "H3N2.log"), burnin=0)
    return names, [s for s in states if s["sample"] in samples], info


def test_reweight_states_match_bench():
    """Per-state exact and converged-MASCOT values agree with the research run (bench/h3n2_posterior_eval.csv)."""
    ref = {r["sample"]: r for r in csv.DictReader(open(REF))}
    names, states, _ = _states_of({"0", "45000", "55000"})
    assert len(states) == 3
    for s in states:
        r = rw.evaluate_state(s["sample"], s["tree"])
        b = ref[s["sample"]]
        assert r["engine"] == "dense"
        assert abs(float(r["exact"]) - float(b["exact"])) < 1e-8
        assert abs(float(r["mascot_converged"]) - float(b["mascot"])) < 1e-8
        for a, ab in zip(names, ("HK", "NY", "NZ")):
            assert abs(float(r[f"root_exact_{a}"]) - float(b[f"root_exact_{ab}"])) < 1e-8
            assert abs(float(r[f"root_mascot_{a}"]) - float(b[f"root_mascot_{ab}"])) < 1e-8


def test_reweight_summary_matches_bench():
    """With the research run's per-state values for all 361 post-burn-in states: the default weights (denominator =
    the log's 'Mascot' column) give the exact posterior E of bench/h3n2_posterior_reweight.md (ESS 309.1, the E means
    and root); --denominator converged gives ESS 320.5; a log without the column falls back to it with a warning."""
    ref = [r for r in csv.DictReader(open(REF))]
    names, states, info = rw.load_states(os.path.join(EX, "H3N2.trees"), os.path.join(EX, "H3N2.log"), burnin=0.1)
    by = {r["sample"]: r for r in ref}
    assert len(states) == 361 and info["logged_col"] == "Mascot"
    rows = []
    for s in states:
        b = by[s["sample"]]
        row = {"exact": b["exact"], "mascot_converged": b["mascot"], "mascot_logged": s["logged"],
               "exact_error": "0", "engine": b["engine"][:5]}
        for a, ab in zip(names, ("HK", "NY", "NZ")):
            row[f"root_exact_{a}"], row[f"root_mascot_{a}"] = b[f"root_exact_{ab}"], b[f"root_mascot_{ab}"]
        rows.append(row)

    s = rw.summarize_rows(rows, states, info, names)
    assert s["denominator"] == "logged" and s["denominator_warning"] is None
    assert abs(s["ess"] - 309.1) < 0.05
    mean = {p["name"]: p["exact_mean"] for p in s["params"]}
    md = {"Ne.Hong_Kong": 1.410, "Ne.New_York": 1.962, "Ne.New_Zealand": 0.895, "m.Hong_Kong_to_New_York": 1.177,
          "m.Hong_Kong_to_New_Zealand": 0.468, "m.New_York_to_Hong_Kong": 1.447, "m.New_York_to_New_Zealand": 0.567,
          "m.New_Zealand_to_Hong_Kong": 0.774, "m.New_Zealand_to_New_York": 0.830, "Tree.height": 3.664}
    for k, v in md.items():
        assert abs(mean[k] - v) < 6e-4, k
    assert np.allclose(s["root"]["exact"], [0.531, 0.263, 0.206], atol=6e-4)
    assert np.allclose(s["root"]["exact_weights_mascot_root"], [0.528, 0.262, 0.210], atol=6e-4)
    assert "can be trusted" in s["verdict"]

    c = rw.summarize_rows(rows, states, info, names, "converged")
    assert c["denominator"] == "converged" and abs(c["ess"] - 320.5) < 0.05
    lw = np.array([float(r["exact"]) - float(r["mascot_converged"]) for r in rows])
    assert abs(c["logw_median"] - (-0.350)) < 5e-4 and abs(c["logw_sd"] - 0.438) < 5e-4
    w = np.exp(lw - lw.max())
    x = np.array([float(st["row"]["NeConstant.New_Zealand"]) for st in states])
    p = next(q for q in c["params"] if q["name"] == "Ne.New_Zealand")
    assert abs(p["exact_mean"] - (w * x).sum() / w.sum()) < 1e-12 and abs(p["mascot_mean"] - x.mean()) < 1e-12

    f = rw.summarize_rows(rows, states, dict(info, logged_col=None), names)
    assert f["denominator"] == "converged" and abs(f["ess"] - 320.5) < 0.05 and "no MASCOT density column" in \
        f["denominator_warning"]


def test_reweight_cli_resume(tmp_path, capsys):
    """A 3-state run writes one CSV row per state; --resume then evaluates nothing new."""
    keep = ("0", "45000", "55000")
    src = open(os.path.join(EX, "H3N2.trees")).read().splitlines()
    lines = [ln for ln in src if not ln.strip().startswith("tree ") or any(f"STATE_{k} " in ln for k in keep)]
    (tmp_path / "x.trees").write_text("\n".join(lines) + "\n")
    log = open(os.path.join(EX, "H3N2.log")).read().splitlines()
    hdr = [i for i, ln in enumerate(log) if ln.startswith("Sample")][0]
    rows = [ln for ln in log[hdr + 1:] if ln.split("\t")[0] in keep]
    (tmp_path / "x.log").write_text("\n".join(log[:hdr + 1] + rows) + "\n")
    args = ["reweight", str(tmp_path / "x.trees"), str(tmp_path / "x.log"), "--burnin", "0", "--quiet"]
    main(args)
    out = capsys.readouterr().out
    assert "weight ESS" in out and "root deme probability" in out
    assert "too few effective states" in out             # 3 states: far below the minimum ESS
    csvf = tmp_path / "x.escott-reweight.csv"
    got = list(csv.DictReader(open(csvf)))
    assert len(got) == 3 and all(r["mascot_logged"] and r["mascot_converged"] for r in got)
    with pytest.raises(SystemExit, match="--resume"):
        main(args)
    main(args + ["--resume"])
    assert len(list(csv.DictReader(open(csvf)))) == 3


def _small_tree_file(tmp_path):
    """8 tips, 2 demes; the dense engine is instant."""
    t = random_tree(8, 2, 11)
    import json
    f = tmp_path / "small.json"
    f.write_text(json.dumps(dict(t, deme_names=["A", "B"])))
    return str(f), dict(t, deme_names=["A", "B"])


def test_fit_exact_refinement(tmp_path):
    from escott.fit import Model, fit
    from scipy.optimize import minimize
    path, tree = _small_tree_file(tmp_path)
    res = fit(tree, "asym", starts=2, engine="both", jobs=1)
    best = res["peaks"][0]
    assert best["loglik"] >= best["exact_at_mascot"] - 1e-9
    assert best["status"] in ("converged",) or best["status"].startswith("lower bound")
    m = Model("asym", ["A", "B"])

    def f(th):
        M, Ne = m.unpack(th)
        return -api.dense_root(api._as_tree(tree, M, Ne))[0]

    direct = minimize(f, best["theta"], method="L-BFGS-B", bounds=[(-12, 7)] * m.npar)
    assert -direct.fun - best["loglik"] < 2e-3       # the refinement stopped within its tolerance of the maximum


def test_cli_fit_and_compare(tmp_path, capsys):
    path, _ = _small_tree_file(tmp_path)
    main(["fit", path, "--model", "sym", "--starts", "2", "--engine", "mascot", "--quiet"])
    assert "best MASCOT fit" in capsys.readouterr().out
    main(["compare", path, "--starts", "2", "--quiet", "--out", str(tmp_path / "c.json")])
    out = capsys.readouterr().out
    assert "df = 1" in out and "MASCOT" in out and "exact" in out
    import json
    c = json.load(open(tmp_path / "c.json"))
    assert c["lrt_exact"]["df"] == 1 and c["lrt_exact"]["stat"] > -1e-3
