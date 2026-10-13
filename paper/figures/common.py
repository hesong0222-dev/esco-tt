import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "..", "bench")
KAGGLE = os.path.join(HERE, "..", "..", "kaggle")
REG_COL = {"slow": "#1b6ca8", "medium": "#e08214", "fast": "#5aae61"}
D_COL = {2: "#7570b3", 3: "#d95f02", 4: "#1b9e77", 6: "#666666"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight")
    print("wrote", name)


def regime_of(s):
    for r in ("slow", "medium", "fast"):
        if f"_{r}_" in s:
            return r
