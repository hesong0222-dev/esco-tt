"""Continuous migration-scale measures of a tree + parameters (review/AUDIT_DESIGN.md M1).

  nm      = mean_a Ne[a] * m_out[a],  m_out[a] = sum_b M[a][b]       (scaled per-lineage outflow, "Nm")
  ratio   = mean_a m_out[a] / mean_a (1/Ne[a])                        (outflow / pairwise coalescence rate)
  emig    = total branch length * mean_a m_out[a] / (k-1)             (expected migrations per coalescence)
  fitch   = Fitch minimum number of deme changes from tip demes / (k-1) (parameter-free, also on real trees)
"""
import numpy as np


def metrics(tree):
    M = np.array(tree["M"], float)
    np.fill_diagonal(M, 0.0)
    Ne = np.array(tree["Ne"], float)
    mout = M.sum(1)
    ev = tree["events"]
    birth, children, deme = {}, {}, {}
    k, L = 0, 0.0
    for e in ev:
        if e["type"] == "sample":
            birth[e["id"]] = e["t"]
            deme[e["id"]] = e["deme"]
            k += 1
        else:
            for c in (e["a"], e["b"]):
                L += e["t"] - birth[c]
            birth[e["new"]] = e["t"]
            children[e["new"]] = (e["a"], e["b"])
    # Fitch, bottom-up in event order (children always precede parents)
    sets, changes = {}, 0
    for e in ev:
        if e["type"] == "sample":
            sets[e["id"]] = {e["deme"]}
        else:
            a, b = sets[e["a"]], sets[e["b"]]
            if a & b:
                sets[e["new"]] = a & b
            else:
                sets[e["new"]] = a | b
                changes += 1
    return dict(nm=float(np.mean(Ne * mout)), ratio=float(mout.mean() / np.mean(1 / Ne)),
                emig=float(L * mout.mean() / (k - 1)), fitch=changes / (k - 1))
