"""Tables for bench/robustness.md from bench/robustness/*.csv.
Dense references: bench/tt_vs_dense.csv (ll_dense) and kaggle/dense_big.csv.
Writes bench/robustness/summary_*.csv and prints markdown tables."""
import glob, os
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "bench", "robustness")


def dense_refs():
    a = pd.read_csv(os.path.join(ROOT, "bench", "tt_vs_dense.csv"))[["file", "ll_dense"]]
    b = pd.read_csv(os.path.join(ROOT, "kaggle", "dense_big.csv"), header=None, names=["file", "alive", "ll_dense", "sec"])
    return dict(zip(pd.concat([a.file, b.file]), pd.concat([a.ll_dense, b.ll_dense])))


def load(patterns):
    fs = sorted(set(f for p in patterns for f in glob.glob(os.path.join(R, p))))
    if not fs:
        return pd.DataFrame()
    fs = [f for f in fs if os.path.getsize(f) > 0]
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for c, v in (("eps_total", np.nan), ("s_max", 0.25), ("order", 4)):
        if c not in df:
            df[c] = v
        df[c] = df[c].fillna(v) if c != "eps_total" else df[c]
    df = df[pd.to_numeric(df.loglik, errors="coerce").notna()].copy()
    df["loglik"] = df.loglik.astype(float)
    df["regime"] = df.tree.str.extract(r"_(slow|medium|fast)_")[0]
    df["d"] = df.tree.str.extract(r"^d(\d)")[0].astype(int)
    df["et"] = df.eps_total.notna() & (df.eps_total.astype(str) != "")
    df = df[(df.s_max == 0.25) & (df.order == 4)]
    return df.drop_duplicates(["tree", "eps", "eps_total", "steps_mult", "mass_fix"], keep="last")


def md(df, floatfmt="{:.2e}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def truncation_split(df, ref):
    """Per tree: dense ref, step error (eps 1e-10 or none), truncation part at each eps (sm1, sm2, mass 0)."""
    rows = []
    for tree, g in df[~df.et].groupby("tree"):
        dense = ref.get(tree, np.nan)
        def ll(eps, sm=1, mf=0):
            x = g[(g.eps == eps) & (g.steps_mult == sm) & (g.mass_fix == mf)]
            return (x.loglik.iloc[0], int(x.n_round.iloc[0]), int(x.max_rank.iloc[0]), float(x.runtime.iloc[0])) if len(x) else (np.nan, 0, 0, np.nan)
        base1, base2 = ll(1e-10, 1)[0], ll(1e-10, 2)[0]
        for sm in (1, 2):
            base = base1 if sm == 1 else base2
            for eps in sorted(g.eps.unique()):
                if eps <= 1e-10:
                    continue
                v, n, rk, rt = ll(eps, sm)
                if np.isnan(v):
                    continue
                v1 = ll(eps, sm, 1)[0]
                rows.append(dict(tree=tree, d=g.d.iloc[0], regime=g.regime.iloc[0], eps=eps, sm=sm, n_round=n, rank=rk,
                                 sec=rt, err=v - dense, err_mass=v1 - dense,
                                 step_err=base - dense, trunc=v - base if not np.isnan(base) else np.nan))
    t = pd.DataFrame(rows)
    t["c"] = t.trunc / (t.eps * t.n_round)
    t["c_tot"] = t.err / (t.eps * t.n_round)
    return t


def rank_table():
    df = load(["rank_v2_*.csv", "kaggle_R*.csv"])
    if df.empty:
        return df
    meta = {}
    for f in glob.glob(os.path.join(R, "rank_trees", "*.json")):
        import json
        m = json.load(open(f))["meta"]
        meta[os.path.basename(f)] = m["max_alive"]
    df["k"] = df.tree.str.extract(r"_k(\d+)_")[0].astype(int)
    df["alive"] = df.tree.map(meta)
    # error of eps=1e-6 vs eps=1e-7 on the same tree (k <= 20)
    e7 = df[df.eps == 1e-7].set_index("tree").loglik
    df["d_e7"] = df.tree.map(e7)
    df["err_vs_e7"] = np.where(df.eps == 1e-6, df.loglik - df.d_e7, np.nan)
    g = df.groupby(["d", "k", "regime", "eps"])
    out = g.agg(n=("max_rank", "size"), alive=("alive", "mean"), rank_mean=("max_rank", "mean"),
                rank_sd=("max_rank", "std"), rank_min=("max_rank", "min"), rank_max=("max_rank", "max"),
                sec_med=("runtime", "median"), abs_err_vs_e7_max=("err_vs_e7", lambda x: np.nanmax(np.abs(x)) if x.notna().any() else np.nan)).reset_index()
    return out


def mass_fix_table(t):
    x = t[(t.sm == 1) & (t.eps >= 1e-7) & t.err_mass.notna()].copy()
    x["set"] = np.where(x.tree.isin(SET_B), "B (d4, 13-14 alive)", "A (small)")
    x["ratio"] = x.err_mass.abs() / x.err.abs()
    return x.groupby(["set", "regime"]).agg(n=("ratio", "size"), median_ratio=("ratio", "median"),
                                             frac_improved=("ratio", lambda r: float(np.mean(r < 1))),
                                             mean_err_plain=("err", "mean"), mean_err_massfix=("err_mass", "mean")).reset_index()


SET_B = ["d4_k16_slow_r1.json", "d4_k18_slow_r1.json", "d4_k20_slow_r1.json", "d4_k16_fast_r1.json", "d4_k18_fast_r1.json",
         "d4_k14_medium_r1.json", "d4_k18_medium_r1.json", "d4_k20_medium_r2.json"]


def large_ladder():
    df = load(["setC_*.csv", "kaggle_KC*.csv"])
    if df.empty:
        return df
    rows = []
    for tree, g in df[~df.et].groupby("tree"):
        g1 = g[(g.steps_mult == 1) & (g.mass_fix == 0)].sort_values("eps")
        if g1.empty:
            continue
        refrow = g1.iloc[0]
        for _, r in g[~g.et].iterrows():
            rows.append(dict(tree=tree, eps=r.eps, sm=r.steps_mult, mass_fix=r.mass_fix, loglik=r.loglik,
                             diff_vs_tightest=r.loglik - refrow.loglik, ref_eps=refrow.eps, rank=r.max_rank,
                             n_round=r.n_round, epsN=r.eps * r.n_round, sec=r.runtime))
    return pd.DataFrame(rows).sort_values(["tree", "sm", "mass_fix", "eps"])


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 1000)
    ref = dense_refs()
    df = load(["setA_*.csv", "setB_*.csv", "kaggle_KB*.csv"])
    t = truncation_split(df, ref)
    t["set"] = np.where(t.tree.isin(SET_B), "B", "A")
    t.to_csv(os.path.join(R, "summary_truncation_split.csv"), index=False)
    s1 = t[t.sm == 1]
    print("## step error at default h (eps 1e-10 run - dense), set A")
    print(s1.groupby(["regime"]).apply(lambda g: g.groupby("tree").step_err.first().abs().agg(["median", "max"])).to_string())
    print("## c = (logL_TT - dense)/(eps*N_round), plain h, mass_fix off")
    s1 = s1.assign(ac=s1.c_tot.abs())
    print(s1.groupby(["set", "regime"]).agg(n=("ac", "size"), med_abs_c=("ac", "median"), max_abs_c=("ac", "max"),
                                            frac_neg=("err", lambda e: float(np.mean(e < 0)))).to_string())
    print(s1.groupby(["eps"]).agg(n=("ac", "size"), med_abs_c=("ac", "median"), max_abs_c=("ac", "max"),
                                 max_abs_err=("err", lambda e: e.abs().max())).to_string())
    p = t[t.eps == 1e-6].pivot_table(index="tree", columns="sm", values="err").dropna()
    r = (p[2] / p[1])
    print("## step ladder eps 1e-6: err(h/2)/err(h)  median %.2f IQR %.2f-%.2f n %d" % (r.median(), r.quantile(.25), r.quantile(.75), len(r)))
    mf = mass_fix_table(t)
    mf.to_csv(os.path.join(R, "summary_mass_fix.csv"), index=False)
    print("## mass fix"); print(mf.to_string())
    et = df[df.et].copy()
    et["err"] = et.loglik - et.tree.map(ref)
    et["ratio"] = et.err.abs() / et.eps_total.astype(float)
    et.to_csv(os.path.join(R, "summary_eps_total.csv"), index=False)
    print("## eps_total calibration"); print(et[["tree", "eps_total", "mass_fix", "eps", "err", "ratio", "max_rank", "runtime"]].to_string())
    lg = large_ladder()
    if len(lg):
        lg.to_csv(os.path.join(R, "summary_large_ladder.csv"), index=False)
        print("## large trees"); print(lg.to_string())
    rt = rank_table()
    rt.to_csv(os.path.join(R, "summary_rank_v2.csv"), index=False)
    print("## rank v2"); print(rt.to_string())
