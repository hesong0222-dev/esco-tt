"""Read the Taming-the-BEAST H3N2 MASCOT run: trace log rows and the per-state parameters.

load_log(path) -> list of dicts (one per logged state, header from the first non-comment line)
params(row)    -> (M, Ne): M[a][b] = b_migrationConstant.<a>_to_<b> (backward a -> b), Ne[a] = NeConstant.<a>
DEMES          -> deme order used everywhere (0 = Hong_Kong, 1 = New_York, 2 = New_Zealand)
"""

DEMES = ["Hong_Kong", "New_York", "New_Zealand"]


def load_log(path):
    rows = [l for l in open(path) if not l.startswith("#")]
    hdr = rows[0].rstrip("\n").split("\t")
    return [dict(zip(hdr, r.rstrip("\n").split("\t"))) for r in rows[1:]]


def params(row):
    Ne = [float(row[f"NeConstant.{a}"]) for a in DEMES]
    M = [[0.0] * 3 for _ in range(3)]
    for a in range(3):
        for b in range(3):
            if a != b:
                M[a][b] = float(row[f"b_migrationConstant.{DEMES[a]}_to_{DEMES[b]}"])
    return M, Ne
