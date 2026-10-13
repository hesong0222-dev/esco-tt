"""H3N2: shifts from the sampled posterior (B, BEAST 2 MASCOT run as logged) to the exact posterior (E), in percent and
in posterior SDs of E, plus Kish ESS (2026-10-13). Weights w ∝ exp(ℓ_exact − ℓ_BEAST_logged) = escott reweight default
(denominator = the chain's target density). Same states and burn-in (10%) as bench/h3n2_posterior_reweight.py."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3n2_posterior_reweight as R

E = R.load(0.1)
lw = E["exact"].values - E["beast_logged"].values
w = np.exp(lw - lw.max()); w /= w.sum()
print("n", len(E), "Kish ESS %.1f" % R.kish(w), "log-weight median %.3f SD %.3f" % (np.median(lw - np.median(lw)) + np.median(lw), lw.std()))
cols = [c for c in E.columns if c.startswith(("Ne_", "m_"))] + ["Tree.height", "clockRate"]
for c in cols:
    x = E[c].values.astype(float)
    mb, me = x.mean(), np.sum(w * x)
    sd = np.sqrt(np.sum(w * (x - me) ** 2))
    print(f"{c:28s} B {mb:.4g}  E {me:.4g}  shift {100*(me/mb-1):+.1f}%  shift/SD_E {(me-mb)/sd:+.3f}")
hk_b = E["root_mascot_HK"].mean(); hk_e = np.sum(w * E["root_exact_HK"].values)
print("root HK: B (MASCOT recursion) %.3f -> E (exact) %.3f" % (hk_b, hk_e))
