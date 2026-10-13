"""Prints the next unfinished TT task for queue.sh (or nothing when all are done / waiting).

Priority 0 (coordinator, 2026-10-09): finish the masco-tree exact refinements first, so the masco ranking flip
compares refined exact optima with refined exact optima: asym hub (mode A), asym cha root (mode C), sym hub
(continued from the stopped run's best point), sym regular; then h/2 at each of them. Start files
masco_mascot_<tag>.json; the TT values already computed for these start points are seeded into masco_ttcache.jsonl.

Then, per tree (sisco first), each tree only after <tree>_seeds_done exists:
  1. ev   : TT eps 1e-6 at the competing MASCOT optima: asym within REF_ASYM of the best asym MASCOT logL,
            sym within REF_SYM of the best sym one
  2. ref  : exact quasi-Newton refinement of the top candidates only: exact-best asym, MASCOT-best asym (if
            different), exact-best sym
  3. pair : TT eps 1e-6 with half step (h/2) at every finished exact (refined) optimum
  4. ev   : TT eps 1e-6 at the remaining MASCOT optima within EV_WINDOW (e.g. ans-root modes)
Stage 1 for both trees comes before stage 2 for either (the MASCOT-optimum ranking is the first answer).
Tasks listed in failed.txt or claimed.txt are skipped. `plan.py list` prints all tasks with their status.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV_WINDOW, REF_ASYM, REF_SYM = 12.0, 3.0, 8.0
TREES = ["sisco", "lisco"]
MASCO = ["asym_modeA", "asym_modeC", "sym_hubcont", "sym_regular"]


def optima(m, model):
    out = []
    for p in glob.glob(os.path.join(HERE, f"{m}_mascot_{model}_L*.json")):
        out.append((json.load(open(p))["loglik"], os.path.basename(p)))
    return sorted(out, reverse=True)


def jget(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def tasks():
    ev, ref, pair, ev2 = [], [], [], []
    mref, mpair = [], []
    P = jget(os.path.join(HERE, "masco_pair_h2.json"))
    for tag in MASCO:
        fit = f"masco_refine_{tag}_fit.json"
        done = os.path.exists(os.path.join(HERE, fit))
        mref.append(("ref", "masco", tag, done))
        if done:
            mpair.append(("pair", "masco", fit, "tt_1e-06_h2" in P.get(fit, {})))
    for m in TREES:
        if not os.path.exists(os.path.join(HERE, f"{m}_seeds_done")):
            ev.append(("wait", m, None, False))
            continue
        E = jget(os.path.join(HERE, f"{m}_eval.json"))
        P = jget(os.path.join(HERE, f"{m}_pair_h2.json"))
        cand = []
        for model in ["asym", "sym"]:
            o = optima(m, model)
            if not o:
                continue
            best = o[0][0]
            for ll, f in o:
                done = "tt_1e-06" in E.get(f, {})
                if ll >= best - (REF_ASYM if model == "asym" else REF_SYM):
                    ev.append(("ev", m, f, done))
                    cand.append((E.get(f, {}).get("tt_1e-06", -1e9), model, f))
                elif ll >= best - EV_WINDOW:
                    ev2.append(("ev", m, f, done))
        # refine the exact-best asym and sym first, then the rest by exact logL
        if any(c[0] == -1e9 for c in cand):
            continue  # refine choice needs all exact values
        cand.sort(reverse=True)
        top = [c for i, c in enumerate(cand) if c[1] not in [d[1] for d in cand[:i]]]  # exact-best per model
        mbest = max((c for c in cand if c[1] == "asym"), key=lambda c: json.load(open(os.path.join(HERE, c[2])))["loglik"])
        if mbest not in top:
            top.append(mbest)
        for _, model, f in top:
            tag = f.replace(f"{m}_mascot_", "").replace(".json", "")
            fit = f"{m}_refine_{tag}_fit.json"
            done = os.path.exists(os.path.join(HERE, fit))
            ref.append(("ref", m, tag, done))
            if done:
                pair.append(("pair", m, fit, "tt_1e-06_h2" in P.get(fit, {})))
    return mref + mpair + ev + ref + pair + ev2


if __name__ == "__main__":
    failed = set()
    for fn in ["failed.txt", "claimed.txt"]:  # claimed.txt: refines pushed to Kaggle (kaggle/realdata2sl)
        if os.path.exists(os.path.join(HERE, fn)):
            failed |= set(open(os.path.join(HERE, fn)).read().split("\n"))
    T = tasks()
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        for t in T:
            print(*t)
        sys.exit()
    for kind, m, arg, done in T:
        if kind == "wait":
            print("wait", m)
            break
        line = f"{kind} {m} {arg}"
        if not done and line not in failed:
            print(line)
            break
