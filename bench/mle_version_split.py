"""Median paired difference Delta (log m_hat MASCOT - log m_hat exact) by code version of the exact fit (2026-10-13).
Registered code (v1) vs later versions (v2/v3), per setting; from bench/mle_study.csv col tt_src (kernel name: r2-* = v2,
r3-* and later rounds = v3, others = v1). Descriptive only; used for the 'effect' column of the deviation table."""
import os
import re
import pandas as pd

D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mle_study.csv"))
D["ver"] = D.tt_src.fillna("").map(lambda s: "v3" if re.search(r"-r[3-9]-", s) else "v2" if "-r2-" in s else "v1")
D["later"] = D.ver != "v1"
print(D.groupby(["d", "k", "regime", "later"]).Delta_m.agg(["count", "median"]).round(3))
