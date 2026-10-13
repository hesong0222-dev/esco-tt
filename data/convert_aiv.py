"""AIV H7 HA (133 taxa): MCC trees from nicfel/The-Structured-Coalescent (location runs) relabelled with the
host-order demes of the BASTA example XML (AIV_HA_BASTA_host_asymmetric.xml, typeTraitSet).

Usage: convert_aiv.py <mcc.trees> <basta_host.xml> <out.json>
Demes: 0 ans (Anseriformes), 1 cha (Charadriiformes), 2 gal (Galliformes), 3 pas (Passeriformes),
4 MexicoGal (Mexican 2012 poultry outbreak) -- the BASTA labels, unchanged.
"""
import json
import re
import sys

import convert

DEMES = ["ans", "cha", "gal", "pas", "MexicoGal"]


def host_map(xml):
    val = re.search(r'traitname="type" value="([^"]*)"', open(xml).read()).group(1)
    return {k.split("|")[0]: v for k, v in (x.rsplit("=", 1) for x in val.split(","))}


def main(tree, xml, out):
    hosts = host_map(xml)
    trans, trees = convert.read_nexus(tree)
    trans = {k: v.split("|")[0] + "|" + v.split("|")[1] + "|" + v.split("|")[2] + "|" + hosts[v.split("|")[0]]
             for k, v in trans.items()}
    assert len(hosts) == len(trans) == 133
    name, nwk = trees[0]
    ev, err, labels = convert.to_events(convert.parse_newick(nwk), trans, DEMES)
    d = len(DEMES)
    res = {"demes": d, "deme_names": DEMES, "source_tree": tree.split("/")[-1] + ":" + name, "tip_date_err": err,
           "M": [[0 if a == b else 1 for b in range(d)] for a in range(d)], "Ne": [1] * d, "events": ev,
           "tip_labels": labels}
    json.dump(res, open(out, "w"), indent=0)
    n = pk = 0
    for e in ev:
        n += 1 if e["type"] == "sample" else -1
        pk = max(pk, n)
    cnt = [sum(1 for e in ev if e["type"] == "sample" and e["deme"] == k) for k in range(d)]
    print(out, "tip date err", err, "tips per deme", dict(zip(DEMES, cnt)), "peak lineages", pk,
          "root height", ev[-1]["t"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
