"""Convention check for the MTT runs: recompute MTT's complete-data structured-coalescent density from the logged
typed trees (.trees, flattened with single-child type-change nodes) and the logged parameters (.log), with OUR
convention (M[a][b] = backward rate a->b of one lineage, pair coalescence rate 1/Ne[a]); compare with the logged
treePrior column. Also checks that the coalescence times equal those of the input tree JSON.

usage: python typed_density.py RUN.log RUN.trees TREE.json   -> prints max |ours - MTT| and max coal-time error
"""
import json
import math
import re
import sys

import numpy as np


def parse_newick(s):
    """Return list of nodes: dict(parent, children, label, type, bl)."""
    nodes, stack, i = [], [], 0
    cur = None

    def new(parent):
        nodes.append({"parent": parent, "children": [], "label": None, "type": None, "bl": 0.0})
        if parent is not None:
            nodes[parent]["children"].append(len(nodes) - 1)
        return len(nodes) - 1
    cur = new(None)
    while i < len(s):
        c = s[i]
        if c == "(":
            stack.append(cur)
            cur = new(cur)
            i += 1
        elif c == ",":
            cur = new(stack[-1])
            i += 1
        elif c == ")":
            cur = stack.pop()
            i += 1
        elif c == "[":
            j = s.index("]", i)
            m = re.search(r'(?:type|state)="?([^",\]]+)"?', s[i:j + 1])
            if m:
                nodes[cur]["type"] = m.group(1)
            i = j + 1
        elif c == ":":
            m = re.match(r":([-+0-9.eE]+)", s[i:])
            nodes[cur]["bl"] = float(m.group(1))
            i += len(m.group(0))
        elif c == ";":
            break
        else:
            m = re.match(r"[^,():\[\];]+", s[i:])
            nodes[cur]["label"] = m.group(0)
            i += len(m.group(0))
    return nodes


def heights(nodes):
    depth = [0.0] * len(nodes)
    for v in range(len(nodes)):  # parents precede children in creation order
        p = nodes[v]["parent"]
        if p is not None:
            depth[v] = depth[p] + nodes[v]["bl"]
    leaf_depths = [depth[v] for v in range(len(nodes)) if not nodes[v]["children"]]
    H = max(leaf_depths)
    return [H - x for x in depth]


def density(nodes, Ne, M, tidx):
    h = heights(nodes)
    d = len(Ne)
    ev = []
    for v, n in enumerate(nodes):
        ty = tidx(n["type"])
        nc = len(n["children"])
        if nc == 0:
            ev.append((h[v], 0, ty, None))
        elif nc == 2:
            ev.append((h[v], 1, ty, None))
        else:
            below = tidx(nodes[n["children"][0]]["type"])
            ev.append((h[v], 2, ty, below))
    ev.sort(key=lambda e: e[0])
    k = np.zeros(d, int)
    out = M.sum(axis=1)
    lp, t = 0.0, None
    for (ht, kind, ty, below) in ev:
        if t is not None and ht > t:
            lam = float(np.sum(k * (k - 1) / 2.0 / Ne) + np.sum(k * out))
            lp -= (ht - t) * lam
        t = ht
        if kind == 0:
            k[ty] += 1
        elif kind == 1:
            k[ty] -= 1
            lp += math.log(1.0 / Ne[ty])
        else:
            k[below] -= 1
            k[ty] += 1
            lp += math.log(M[below, ty])
    coal_h = sorted(e[0] for e in ev if e[1] == 1)
    return lp, coal_h


def read_trees(path):
    trees, trans = {}, {}
    intrans = False
    for line in open(path):
        s = line.strip()
        if s.lower() == "translate":
            intrans = True
            continue
        if intrans:
            if s.startswith(";"):
                intrans = False
                continue
            a, b = s.rstrip(",;").split()
            trans[a] = b
            if s.endswith(";"):
                intrans = False
            continue
        m = re.match(r"tree STATE_(\d+)\s*=\s*(.*)", s)
        if m:
            trees[int(m.group(1))] = m.group(2)
    return trees, trans


def read_log(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path) if not l.startswith("#")]
    head = rows[0]
    data = {}
    for r in rows[1:]:
        if len(r) >= len(head) - 1 and r[0].strip():
            data[int(r[0])] = dict(zip(head, r))
    return head, data


def check(logf, treesf, jsonf):
    tree = json.load(open(jsonf))
    d = len(tree["Ne"])
    od = [(a, b) for a in range(d) for b in range(d) if a != b]
    coal_true = sorted(e["t"] for e in tree["events"] if e["type"] == "coal")
    head, rows = read_log(logf)
    trees, _ = read_trees(treesf)
    names = [f"d{a}" for a in range(d)]
    worst, worst_t, n = 0.0, 0.0, 0
    for st, nw in trees.items():
        if st not in rows:
            continue
        r = rows[st]
        Ne = np.array([float(r[f"popSizes.{a+1}"]) for a in range(d)])
        M = np.zeros((d, d))
        for j, (a, b) in enumerate(od):
            M[a, b] = float(r[f"rateMatrix.{j+1}"])
        lp, ch = density(parse_newick(nw), Ne, M, lambda s: names.index(s) if s in names else int(s))
        worst = max(worst, abs(lp - float(r["treePrior"])))
        worst_t = max(worst_t, max(abs(x - y) for x, y in zip(ch, coal_true)))
        n += 1
    return n, worst, worst_t


if __name__ == "__main__":
    n, w, wt = check(*sys.argv[1:4])
    print(f"checked {n} states: max |ours - MTT treePrior| = {w:.3e}; max coal-time error = {wt:.3e}")
