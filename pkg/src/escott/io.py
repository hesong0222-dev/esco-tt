"""Read time trees (NEXUS/Newick), tip locations, BEAST 2 MASCOT logs and parameters into the escott tree dict.

Tree dict (same as the research code's JSON):
  {"demes": d, "deme_names": [...], "M": d x d (M[a][b] = backward rate a -> b), "Ne": [...],
   "events": [{"t", "type": "sample", "id", "deme"} | {"t", "type": "coal", "a", "b", "new"}]}
Times are heights above the most recent tip (backwards in time), in the tree's branch-length units.
"""
import csv
import json
import re

ANNOTATION_KEYS = ("type", "location")  # tip annotations read automatically when every tip has one
MAX_AUTO_DEMES = 10                      # more distinct tip locations than this from a label field is almost surely wrong

# ----------------------------------------------------------------------------- tree parsing


def _read_nexus(txt):
    trans = {}
    m = re.search(r"Translate(.*?);", txt, re.S | re.I)
    if m:
        for line in m.group(1).strip().splitlines():
            line = line.strip().rstrip(",")
            if line:
                k, v = line.split(None, 1)
                trans[k] = v.strip().strip("'\"")
    trees = re.findall(r"^\s*tree\s+(\S+)\s*=\s*(?:\[&[RU]\]\s*)?(.*?);\s*$", txt, re.M | re.I)
    return trans, trees


def _parse_annot(s):
    """'[&a=1,b={x,y},max=NY]' -> {'a': '1', 'b': '{x,y}', 'max': 'NY'}"""
    out, depth, cur = {}, 0, ""
    for ch in s.strip("[]&") + ",":
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            if "=" in cur:
                k, v = cur.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
            cur = ""
        else:
            cur += ch
    return out


def _parse_newick(s):
    pos = 0

    def annot():
        nonlocal pos
        a = {}
        while pos < len(s) and s[pos] == "[":
            end = s.index("]", pos)
            a.update(_parse_annot(s[pos:end + 1]))
            pos = end + 1
        return a

    def node():
        nonlocal pos
        ch = []
        if s[pos] == "(":
            pos += 1
            ch.append(node())
            while s[pos] == ",":
                pos += 1
                ch.append(node())
            if s[pos] != ")":
                raise ValueError(f"bad Newick near position {pos}: {s[max(0, pos - 20):pos + 20]!r}")
            pos += 1
        m = re.match(r"('[^']*'|[^:,()\[;]*)", s[pos:])
        name = m.group(1).strip().strip("'")
        pos += m.end()
        a = annot()
        length = 0.0
        if pos < len(s) and s[pos] == ":":
            pos += 1
            a.update(annot())
            m = re.match(r"[0-9eE.+-]+", s[pos:])
            length = float(m.group(0))
            pos += m.end()
            a.update(annot())
        return {"name": name, "len": length, "ch": ch, "annot": a}

    return node()


def read_trees(path):
    """All trees of a NEXUS (or one-tree Newick) file: (translate table, [(name, newick string), ...])."""
    with open(path) as f:
        txt = f.read()
    if txt.lstrip().upper().startswith("#NEXUS"):
        trans, trees = _read_nexus(txt)
        if not trees:
            raise ValueError(f"no 'tree NAME = ...;' line found in the NEXUS file {path}")
        return trans, trees
    nwk = txt.strip().rstrip(";")
    if not nwk.startswith("("):
        raise ValueError(f"{path} is neither a NEXUS file (#NEXUS header) nor a Newick tree")
    return {}, [("0", nwk)]


def read_tree(path, tree=0):
    """Return (root node, translate table). `tree`: index or name of the tree in a NEXUS file."""
    trans, trees = read_trees(path)
    if isinstance(tree, int) or str(tree).lstrip("-").isdigit():
        i = int(tree)
        if not -len(trees) <= i < len(trees):
            raise ValueError(f"{path} has {len(trees)} tree(s); index {i} is out of range")
        name, nwk = trees[i]
    else:
        hit = [t for t in trees if t[0] == tree]
        if not hit:
            raise ValueError(f"no tree named {tree!r} in {path} (first names: {[t[0] for t in trees[:3]]})")
        name, nwk = hit[0]
    return _parse_newick(nwk.strip()), trans


# ----------------------------------------------------------------------------- tips -> demes


def read_locations_csv(path):
    """CSV/TSV with two columns: tip label, deme name. A header row is allowed."""
    with open(path) as f:
        txt = f.read()
    lines = txt.splitlines()
    if not lines:
        raise ValueError(f"locations file {path} is empty")
    dialect = "excel-tab" if "\t" in lines[0] else "excel"
    rows = [r for r in csv.reader(lines, dialect) if r and r[0].strip()]
    bad = [r for r in rows if len(r) < 2]
    if bad:
        raise ValueError(f"locations file {path}: every row needs two columns (tip label, deme); got {bad[0]}")
    return {r[0].strip(): r[1].strip() for r in rows}


def _location_source(leaves, labels, locations, location_key, label_field, sep):
    """Decide where tip locations come from; returns (list of locations, human-readable source)."""
    if locations is not None:
        missing = [lab for lab in labels if lab not in locations]
        if missing:
            raise ValueError(f"{len(missing)} tip(s) not found in the locations file, e.g. {missing[0]!r} "
                             f"(the first column must hold the tip labels exactly as in the tree)")
        return [locations[lab] for lab in labels], "locations file"
    if location_key is not None:
        miss = [lab for lab, v in zip(labels, leaves) if location_key not in v["annot"]]
        if miss:
            raise ValueError(f"tip {miss[0]!r} has no annotation '{location_key}' "
                             f"(its annotations: {sorted(leaves[labels.index(miss[0])]['annot'])})")
        return [v["annot"][location_key] for v in leaves], f"tip annotation '{location_key}'"
    for key in ANNOTATION_KEYS:
        if all(key in v["annot"] for v in leaves):
            return [v["annot"][key] for v in leaves], f"tip annotation '{key}'"
    locs = []
    for lab in labels:
        parts = lab.split(sep)
        try:
            locs.append(parts[label_field])
        except IndexError:
            raise ValueError(f"tip label {lab!r} has no field {label_field} when split at {sep!r}; "
                             f"give the tip locations with --locations tips.csv") from None
    return locs, f"field {label_field} of the tip label split at '{sep}'"


def events_from_root(root, trans, locations=None, location_key=None, label_field=-1, sep="|", deme_names=None):
    """Parsed tree -> (events, deme_names, tip_labels, location source). See tree_events."""
    nodes = []
    stack = [(root, -root["len"])]
    while stack:                                  # iterative DFS (pre-order, children left to right)
        v, dep = stack.pop()
        v["depth"] = dep + v["len"]
        nodes.append(v)
        for c in reversed(v["ch"]):
            stack.append((c, v["depth"]))
    maxd = max(v["depth"] for v in nodes)
    for v in nodes:
        v["t"] = max(maxd - v["depth"], 0.0)
    leaves = [v for v in nodes if not v["ch"]]
    labels = [trans.get(v["name"], v["name"]) for v in leaves]
    locs, source = _location_source(leaves, labels, locations, location_key, label_field, sep)
    if deme_names is None:
        deme_names = sorted(set(locs))
        if len(deme_names) > MAX_AUTO_DEMES:
            raise ValueError(f"{len(deme_names)} distinct tip locations were read from the {source} "
                             f"(e.g. {deme_names[:3]}); this does not look like a deme assignment. "
                             f"Give the tip locations with --locations tips.csv or --location-key KEY")
    missing = sorted(set(locs) - set(deme_names))
    if missing:
        raise ValueError(f"tip locations {missing} (from the {source}) are not among the demes {list(deme_names)}")
    ev = []
    for i, (v, loc) in enumerate(zip(leaves, locs)):
        v["id"] = i
        ev.append({"t": v["t"], "type": "sample", "id": i, "deme": list(deme_names).index(loc)})
    internal = sorted([v for v in nodes if v["ch"]], key=lambda v: v["t"])
    nid = len(leaves)
    for v in internal:
        if len(v["ch"]) != 2:
            raise ValueError(f"the tree must be strictly bifurcating (found a node with {len(v['ch'])} children)")
        v["id"] = nid
        nid += 1
    for v in internal:
        ev.append({"t": v["t"], "type": "coal", "a": v["ch"][0]["id"], "b": v["ch"][1]["id"], "new": v["id"]})
    ev.sort(key=lambda e: (e["t"], 0 if e["type"] == "sample" else 1))
    return ev, list(deme_names), labels, source


def tree_events(path, tree=0, locations=None, location_key=None, label_field=-1, sep="|", deme_names=None):
    """Read a time tree and tip locations; return (events, deme_names, tip_labels).

    Tip location, in order of precedence: `locations` dict (tip label -> deme, e.g. from a CSV), tip annotation
    `location_key` (e.g. 'max' in a MASCOT/TreeAnnotator tree), a 'type' or 'location' annotation present on every
    tip, or field `label_field` of the tip label split at `sep` (default: last '|' field, the BEAST tutorial
    convention)."""
    root, trans = read_tree(path, tree)
    ev, names, labels, _ = events_from_root(root, trans, locations, location_key, label_field, sep, deme_names)
    return ev, names, labels


# ----------------------------------------------------------------------------- parameters


def params_from_json(path_or_dict, deme_names=None):
    """{"demes": [names], "Ne": [..] or {name: ..}, "M": [[..]] or {"A_to_B": rate}} -> (names, M, Ne).
    M[a][b] is the backward-in-time migration rate of one lineage from deme a to deme b (BEAST/MASCOT
    'b_migration' convention); Ne[a] is the effective population size (pairwise coalescence rate 1/Ne[a])."""
    if isinstance(path_or_dict, dict):
        p = path_or_dict
    else:
        with open(path_or_dict) as f:
            p = json.load(f)
    names = p.get("demes", deme_names)
    if isinstance(names, int) or names is None:
        names = deme_names
    if names is None:
        raise ValueError("parameter file needs 'demes': [names]")
    d = len(names)
    Ne = p["Ne"]
    Ne = [float(Ne[n]) for n in names] if isinstance(Ne, dict) else [float(x) for x in Ne]
    M = p["M"]
    if isinstance(M, dict):
        Mm = [[0.0] * d for _ in range(d)]
        for k, v in M.items():
            a, b = k.split("_to_")
            Mm[names.index(a)][names.index(b)] = float(v)
        M = Mm
    else:
        M = [[float(x) for x in row] for row in M]
    for a in range(d):
        M[a][a] = 0.0
    if len(Ne) != d or len(M) != d:
        raise ValueError("Ne / M sizes do not match the number of demes")
    return list(names), M, Ne


def read_beast_log(path):
    """BEAST 2 trace log -> (header, rows as dicts of strings). Lines starting with '#' are comments."""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise ValueError(f"{path} has no data (only comments or empty)")
    hdr = lines[0].split("\t")
    rows = []
    for i, ln in enumerate(lines[1:]):
        vals = ln.split("\t")
        if len(vals) != len(hdr):
            if i == len(lines) - 2:          # a run still writing its last line
                break
            raise ValueError(f"{path}: line {i + 2} has {len(vals)} columns, the header has {len(hdr)}")
        rows.append(dict(zip(hdr, vals)))
    return hdr, rows


def mascot_columns(header, path="the log"):
    """Recognise the constant-rate MASCOT columns NeConstant.<deme> and b_migrationConstant.<a>_to_<b>.
    Returns (deme names in log order, {deme: Ne column}, {(a, b): migration column})."""
    ne = {c.split(".", 1)[1]: c for c in header if c.startswith("NeConstant.")}
    if not ne:
        near = [c for c in header if "ne" in c.lower() or "migration" in c.lower()][:6]
        raise ValueError(f"no NeConstant.<deme> columns in {path}: escott reads constant-Ne, constant-migration "
                         f"MASCOT logs (columns NeConstant.<deme>, b_migrationConstant.<a>_to_<b>)"
                         + (f"; similar columns found: {near}" if near else ""))
    names = list(ne)
    mig, missing = {}, []
    for a in names:
        for b in names:
            if a != b:
                c = f"b_migrationConstant.{a}_to_{b}"
                if c in header:
                    mig[(a, b)] = c
                else:
                    missing.append(c)
    if missing:
        raise ValueError(f"{path} lacks migration columns {missing[:3]}{' ...' if len(missing) > 3 else ''} "
                         f"(expected b_migrationConstant.<a>_to_<b> for all pairs of the demes {names})")
    return names, ne, mig


def params_from_row(row, names, ne_cols, mig_cols):
    d = len(names)
    Ne = [float(row[ne_cols[n]]) for n in names]
    M = [[0.0] * d for _ in range(d)]
    for (a, b), c in mig_cols.items():
        M[names.index(a)][names.index(b)] = float(row[c])
    return M, Ne


def sample_column(header, path="the log"):
    for c in ("Sample", "state", "Sample#"):
        if c in header:
            return c
    raise ValueError(f"{path} has no 'Sample' (state number) column; header starts {header[:4]}")


def params_from_beast_log(path, row=-1, deme_names=None):
    """Parameters from one row of a BEAST 2 MASCOT trace log (columns NeConstant.<deme> and
    b_migrationConstant.<a>_to_<b>). `row`: index into the data rows (default: last)."""
    hdr, data = read_beast_log(path)
    if isinstance(row, str) and not row.lstrip("-").isdigit():
        raise ValueError("row must be an integer")
    names, ne, mig = mascot_columns(hdr, path)
    if deme_names:
        names = list(deme_names)
    r = data[int(row)]
    M, Ne = params_from_row(r, names, ne, mig)
    return names, M, Ne, r.get("Sample")


def load_tree(path, params=None, **kw):
    """Convenience: build the full tree dict from a tree file (NEXUS/Newick) or an escott JSON tree.
    `params`: (names, M, Ne) tuple, a params dict/JSON path, or None (deme names then come from the tips).
    Keyword arguments as in tree_events (tree, locations, location_key, label_field, sep)."""
    if str(path).endswith(".json"):
        with open(path) as f:
            t = json.load(f)
        if params is not None:
            names, M, Ne = params if isinstance(params, tuple) else params_from_json(params, t.get("deme_names"))
            t = dict(t, M=M, Ne=Ne, demes=len(M), deme_names=names)
        return t
    names = None
    if params is not None and not isinstance(params, tuple):
        params = params_from_json(params)
    if params is not None:
        names = params[0]
    which = kw.pop("tree", 0)
    root, trans = read_tree(path, which)
    ev, names, labels, source = events_from_root(root, trans, deme_names=names, **kw)
    t = {"demes": len(names), "deme_names": names, "events": ev, "tip_labels": labels, "location_source": source}
    if params is not None:
        t.update(M=params[1], Ne=params[2])
    return t
