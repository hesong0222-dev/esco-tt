"""Figure 1: method schematic (planar leaf order, one TT core per lineage, local event operators)."""
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch

from common import plt, save

DEME = ["#7570b3", "#d95f02", "#1b9e77"]  # deme 1, 2, 3
INK = "#222222"
GREY = "#8c8c8c"
CORE_FC = "#eef3f8"


# ---------------------------------------------------------------- drawing helpers
def core(ax, x, y, label, w=0.9, h=0.62, fc=CORE_FC, ec=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.0, ls=ls, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=4)


def bond(ax, x0, x1, y, label=None, ls="-"):
    ax.plot([x0, x1], [y, y], color=INK, lw=1.0, ls=ls, zorder=2)
    if label:
        ax.text((x0 + x1) / 2, y + 0.1, label, ha="center", va="bottom", fontsize=6.5, color=GREY)


def leg(ax, x, y0, y1, label=None, color=INK):
    ax.plot([x, x], [y0, y1], color=color, lw=1.0, zorder=2)
    if label:
        ax.text(x + 0.08, y1 + 0.02, label, ha="left", va="bottom", fontsize=6.5, color=GREY)


def opbox(ax, x, y, label, w=0.62, h=0.38, fc="#fff4e0", ec="#b26b00"):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=0.9, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=6.5, zorder=4)


def diamond(ax, x, y, s=0.22, fc="#fde0dd", ec="#c51b8a"):
    ax.add_patch(Polygon([(x, y + s), (x + s, y), (x, y - s), (x - s, y)], closed=True, fc=fc, ec=ec, lw=0.9,
                         zorder=3))


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=INK,
                                 zorder=2))


def delta_bars(ax, x, y, deme, w=0.13, hmax=0.3):
    for a in range(3):
        hh = hmax if a == deme else 0.03
        ax.add_patch(Rectangle((x + (a - 1) * (w + 0.03) - w / 2, y), w, hh, fc=DEME[a] if a == deme else "#cccccc",
                               ec="none", zorder=4))


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.2, 3.9))
axT = fig.add_axes([0.0, 0.04, 0.27, 0.90])
axB = fig.add_axes([0.29, 0.0, 0.71, 1.0])

# (a) time tree; time runs backward = upward.  Planar (depth-first) leaf order A, B, C, D.
x = {"A": 0, "B": 1, "C": 2, "D": 3}
tip_t = {"A": 0.0, "B": 0.0, "C": 2.2, "D": 0.0}
tip_deme = {"A": 0, "B": 1, "C": 2, "D": 0}
t_ab, t_abc, t_root = 1.3, 3.1, 4.2
x_ab = (x["A"] + x["B"]) / 2
x_abc = (x_ab + x["C"]) / 2
x_root = (x_abc + x["D"]) / 2
lw = 1.6
seg = [  # (x, t0, t1) vertical branches
    (x["A"], 0, t_ab), (x["B"], 0, t_ab), (x_ab, t_ab, t_abc), (x["C"], tip_t["C"], t_abc),
    (x_abc, t_abc, t_root), (x["D"], 0, t_root), (x_root, t_root, t_root + 0.45)]
for xx, a, b in seg:
    axT.plot([xx, xx], [a, b], color=INK, lw=lw, solid_capstyle="butt")
for xa, xb, t in [(x["A"], x["B"], t_ab), (x_ab, x["C"], t_abc), (x_abc, x["D"], t_root)]:
    axT.plot([xa, xb], [t, t], color=INK, lw=lw, solid_capstyle="projecting")
for nm in "ABCD":
    axT.plot(x[nm], tip_t[nm], "o", ms=8, mfc=DEME[tip_deme[nm]], mec=INK, mew=0.6, zorder=5)
    axT.text(x[nm], tip_t[nm] - 0.3, nm, ha="center", va="top", fontsize=8)
# planar positions
for nm, i in zip("ABCD", range(1, 5)):
    axT.text(x[nm], -0.78, f"{i}", ha="center", va="top", fontsize=6.5, color=GREY)
axT.text(1.5, -1.07, "planar (depth-first) leaf order", ha="center", va="top", fontsize=6.5, color=GREY)
# event markers (numbers match panel b)
# event 1: a time slice through the first interval
axT.plot([-0.45, 3.45], [0.65, 0.65], color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)


def badge(ax, x0, y0, s):
    ax.text(x0, y0, s, ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.18", fc=INK, ec="none"))


badge(axT, 3.62, 0.65, "1")
badge(axT, x["B"] + 0.33, t_ab, "2")
badge(axT, x["C"] + 0.36, tip_t["C"], "3")
for t in (t_ab, tip_t["C"]):
    axT.plot([-0.45, 3.45], [t, t], color=GREY, lw=0.5, ls=":", zorder=0)
# time axis
axT.annotate("", xy=(-0.75, t_root + 0.4), xytext=(-0.75, -0.05),
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=9))
axT.text(-0.92, (t_root + 0.4) / 2, "time before present", rotation=90, ha="center", va="center", fontsize=7)
# deme legend
for a in range(3):
    axT.plot(-0.3 + a * 1.4, t_root + 0.95, "o", ms=6, mfc=DEME[a], mec=INK, mew=0.5)
    axT.text(-0.12 + a * 1.4, t_root + 0.95, f"deme {a + 1}", va="center", fontsize=6.5)
axT.set_xlim(-1.1, 3.9)
axT.set_ylim(-1.25, t_root + 1.2)
axT.axis("off")
axT.text(-1.1, t_root + 1.2, "a", fontsize=11, fontweight="bold", va="top")

# (b) TT operations
ax = axB
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 10.0)
ax.axis("off")
ax.text(0.0, 10.0, "b", fontsize=11, fontweight="bold", va="top")
ax.text(0.55, 9.93, r"$P(l_1,\dots,l_k)=G_1[l_1]\,G_2[l_2]\cdots G_k[l_k]$: one core per alive lineage, in planar order",
        fontsize=7.5, va="top")

dx = 1.55  # core spacing

# --- row 1: one time step in the interval (A, B, D alive)
y1 = 7.85
badge(ax, 0.3, y1, "1")
xs = [1.3 + i * dx for i in range(3)]
names = [r"$G_A$", r"$G_B$", r"$G_D$"]
for i, (xx, nm) in enumerate(zip(xs, names)):
    core(ax, xx, y1, nm)
    leg(ax, xx, y1 - 0.31, y1 - 1.75)
    opbox(ax, xx, y1 - 0.72, r"$e^{hQ}$")
for i in range(2):
    bond(ax, xs[i] + 0.45, xs[i + 1] - 0.45, y1, r"$r$")
# diagonal count operator (MPO) acting on all legs
ym = y1 - 1.35
for xx in xs:
    ax.add_patch(plt.Circle((xx, ym), 0.15, fc="#e5f5e0", ec="#31a354", lw=0.9, zorder=3))
for i in range(2):
    ax.plot([xs[i] + 0.15, xs[i + 1] - 0.15], [ym, ym], color="#31a354", lw=1.0, zorder=2)
ax.text(xs[-1] + 0.45, y1 - 0.72, "migration: exact, one core at a time,\nranks unchanged",
        fontsize=6.8, va="center")
ax.text(xs[-1] + 0.45, ym, r"no-coalescence factor $e^{-hD}$: diagonal; bond carries the" "\n"
        r"per-deme count $n_a$ (then TT rounding)", fontsize=6.8, va="center")
for xx, l in zip(xs, ["$l_A$", "$l_B$", "$l_D$"]):
    ax.text(xx + 0.08, y1 - 1.78, l, fontsize=6.5, color=GREY, va="top")

# --- row 2: coalescence of A and B (adjacent cores)
y2 = 4.75
badge(ax, 0.3, y2, "2")
xa, xb, xd = 1.3, 1.3 + dx, 1.3 + 2 * dx
core(ax, xa, y2, r"$G_A$")
core(ax, xb, y2, r"$G_B$")
core(ax, xd, y2, r"$G_D$")
bond(ax, xa + 0.45, xb - 0.45, y2, r"$r$")
bond(ax, xb + 0.45, xd - 0.45, y2, r"$r$")
xm, ymd = (xa + xb) / 2, y2 - 1.05
ax.plot([xa, xa, xm - 0.2], [y2 - 0.31, ymd, ymd], color=INK, lw=1.0, zorder=2)
ax.plot([xb, xb, xm + 0.2], [y2 - 0.31, ymd, ymd], color=INK, lw=1.0, zorder=2)
diamond(ax, xm, ymd)
leg(ax, xm, ymd - 0.22, ymd - 0.62)
leg(ax, xd, y2 - 0.31, ymd - 0.62)
ax.text(xm - 0.05, ymd - 0.72, r"$\mathrm{diag}(1/N_{e,a})$", fontsize=6.8, color="#c51b8a", ha="center",
        va="top")
arrow(ax, xd + 0.65, y2 - 0.35, xd + 1.35, y2 - 0.35)
xh, xd2 = xd + 2.1, xd + 2.1 + dx
core(ax, xh, y2 - 0.35, r"$H_{AB}$", w=0.95)
core(ax, xd2, y2 - 0.35, r"$G_D$")
bond(ax, xh + 0.48, xd2 - 0.45, y2 - 0.35, r"$r'$")
leg(ax, xh, y2 - 0.66, y2 - 1.2)
leg(ax, xd2, y2 - 0.66, y2 - 1.2)
ax.text(xh - 0.5, y2 + 0.45, r"$H_{AB}[a]=G_A[a]\,G_B[a]\,/\,N_{e,a}$", fontsize=7, va="bottom")
ax.text(xh - 0.5, y2 - 1.45, r"divide by $\Sigma P$; $\log L \leftarrow \log L + \log \Sigma P$; round",
        fontsize=6.8, va="top")

# --- row 3: sampling of C (insert a delta core at its planar position)
y3 = 1.55
badge(ax, 0.3, y3, "3")
xh3, xc3, xd3 = 1.3, 1.3 + dx, 1.3 + 2 * dx
core(ax, xh3, y3, r"$H_{AB}$", w=0.95)
core(ax, xc3, y3, "", fc="#e5f5e0", ec=DEME[2])
delta_bars(ax, xc3, y3 - 0.17, 2)
ax.text(xc3, y3 + 0.48, r"$I_{r'}\otimes e_{3}$", fontsize=6.8, ha="center", va="bottom", color=DEME[2])
core(ax, xd3, y3, r"$G_D$")
bond(ax, xh3 + 0.48, xc3 - 0.45, y3, r"$r'$")
bond(ax, xc3 + 0.45, xd3 - 0.45, y3, r"$r'$")
for xx in (xh3, xc3, xd3):
    leg(ax, xx, y3 - 0.31, y3 - 0.85)
ax.text(xd3 + 0.75, y3 + 0.1,
        "new lineage C (sampled in deme 3): a point-mass core\n"
        "inserted at C's planar position, between AB and D\n"
        "(appended at the end when the new leaf is rightmost);\nranks unchanged",
        fontsize=6.8, va="center")

# separators
for yy in (6.05, 2.85):
    ax.plot([0.1, 13.1], [yy, yy], color="#dddddd", lw=0.6)

save(fig, "fig1_schematic")
