"""
08b - Research-methodology flowchart (black & white)
====================================================
Five-phase flowchart with decision points and feedback loops, drawn in
greyscale for print. Content mirrors the final pipeline reported in the paper.

Output: outputs/figures/FigM_methodology_flowchart.{pdf,png}
"""
from _common import C, get_logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

log = get_logger("08b_flowchart")

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

INK, GREY, LIGHT = "black", "#555555", "#efefef"
BW, BH = 2.30, 0.78            # box width / height
COLX = [1.35, 4.15, 6.95, 9.75, 12.55]   # column centres
GAP = 0.40                     # vertical gap between nodes

fig, ax = plt.subplots(figsize=(7.6, 7.6 * 8.95 / 13.9))
ax.set_xlim(0, 13.9)
ax.set_ylim(1.30, 10.25)
ax.axis("off")

nodes = {}


def box(key, col, y, title, sub, bold_border=False, fill="white"):
    x = COLX[col]
    p = FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0,rounding_size=0.12",
                       fc=fill, ec=INK, lw=1.5 if bold_border else 0.8)
    ax.add_patch(p)
    ax.text(x, y + 0.14, title, ha="center", va="center", fontsize=7.3, fontweight="bold", color=INK)
    ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=6.0, color=GREY, linespacing=1.05)
    nodes[key] = dict(x=x, y=y, w=BW, h=BH)


def diamond(key, col, y, text, w=1.55, h=0.66):
    x = COLX[col]
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, fc=LIGHT, ec=INK, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.5, color=INK, linespacing=1.0)
    nodes[key] = dict(x=x, y=y, w=w, h=h)


def terminal(key, col, y, text):
    x = COLX[col]
    ax.add_patch(FancyBboxPatch((x - 0.6, y - 0.22), 1.2, 0.44, boxstyle="round,pad=0,rounding_size=0.22",
                                fc=LIGHT, ec=INK, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.3, fontweight="bold")
    nodes[key] = dict(x=x, y=y, w=1.2, h=0.44)


def top(k): n = nodes[k]; return (n["x"], n["y"] + n["h"] / 2)
def bot(k): n = nodes[k]; return (n["x"], n["y"] - n["h"] / 2)
def left(k): n = nodes[k]; return (n["x"] - n["w"] / 2, n["y"])
def right(k): n = nodes[k]; return (n["x"] + n["w"] / 2, n["y"])


def arrow(pts, dashed=False, color=INK):
    """Poly-line arrow through pts; arrow head on the last segment."""
    ls = (0, (3, 2)) if dashed else "-"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.8, ls=ls, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7,
                                 color=color, lw=0.8, ls=ls, shrinkA=0, shrinkB=0))


def label(x, y, t, ha="left"):
    ax.text(x, y, t, fontsize=6.2, color=GREY, ha=ha, va="center", style="italic")


# ---------------------------------------------------------------- headers ---
heads = ["1. Research design", "2. Data preparation", "3. Model development",
         "4. XAI & PREX-Edu", "5. Reporting"]
for x, h in zip(COLX, heads):
    ax.add_patch(FancyBboxPatch((x - BW / 2, 9.72), BW, 0.48, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="black", ec="black"))
    ax.text(x, 9.96, h, ha="center", va="center", fontsize=7.4, fontweight="bold", color="white")
ax.plot([0.1, 13.8], [9.55, 9.55], color=GREY, lw=0.5)

Y = [9.0 - i * (BH + GAP) for i in range(8)]   # row centres

# ------------------------------------------------------- 1 research design --
box("p1", 0, Y[0], "Define problem", "early, leakage-free\nidentification of At-Risk")
box("p2", 0, Y[1], "Literature review", "early warning, XAI,\nfairness, imbalance")
diamond("p3", 0, Y[2], "Gap\nfound?")
box("p4", 0, Y[3], "Research questions", "RQ1 prediction · RQ2 XAI\nRQ3 PREX-Edu · RQ4 fairness")

# -------------------------------------------------------- 2 data preparation
box("d1", 1, Y[0], "Data collection", "primary survey, N = 1,208\nGrades 8–12, 15 variables")
box("d2", 1, Y[1], "Audit & cleaning", "types, ranges, duplicates;\n124 missing → Unknown")
box("d3", 1, Y[2], "Target & leakage", "At-Risk: last-term % ≤ 50\ndrop last-term %, grade")
diamond("d4", 1, Y[3], "Quality\nOK?")
box("d5", 1, Y[4], "Encoding (in-fold)", "one-hot · ordinal · min–max;\n13 predictors")

# ------------------------------------------------------- 3 model development
box("m1", 2, Y[0], "Model selection", "LR · SVM · DT\nRF · GB · XGBoost")
box("m2", 2, Y[1], "Nested 10×10 CV", "inner grid search (PR-AUC)\n+ F1-tuned threshold")
box("m3", 2, Y[2], "Evaluate performance", "PR/ROC-AUC, calibration;\ncorrected t, Friedman, DeLong")
diamond("m4", 2, Y[3], "Perf.\nOK?")
box("m5", 2, Y[4], "Robustness checks", "imbalance, synthetic data,\nτ sensitivity, decision curves")
box("m6", 2, Y[5], "Best model selected", "logistic regression\n(no resampling)", bold_border=True)

# ----------------------------------------------------- 4 XAI / PREX-Edu ----
box("x1", 3, Y[0], "Global XAI methods", "SHAP, permutation\nimportance, coefficients")
box("x2", 3, Y[1], "Local XAI methods", "SHAP, LIME, PDP/ICE;\nstability & agreement")
box("x3", 3, Y[2], "PREX-Edu layer", "4 pedagogical dimensions\n$R_k = \\Sigma\\,|\\phi_j|$ per dimension")
diamond("x4", 3, Y[3], "Faithful?")
box("x5", 3, Y[4], "Actionability checks", "interventions, counter-\nfactuals, cross-model")
box("x6", 3, Y[5], "Fairness audit", "gender, income, parent\neducation (bootstrap CIs)")
box("x7", 3, Y[6], "Explanation outputs", "dimension profile,\neducator-facing summary", bold_border=True)

# ------------------------------------------------------------- 5 reporting --
box("r1", 4, Y[0], "Summarise results", "performance, calibration,\nPREX-Edu validation")
box("r2", 4, Y[1], "Discuss", "answers to RQ1–RQ4;\nprior work, implications")
box("r3", 4, Y[2], "Conclude", "limitations;\nplanned teacher study")
terminal("r4", 4, Y[3] + 0.15, "End")

# --------------------------------------------------- within-column arrows --
for seq in [["p1", "p2", "p3", "p4"], ["d1", "d2", "d3", "d4", "d5"],
            ["m1", "m2", "m3", "m4", "m5", "m6"], ["x1", "x2", "x3", "x4", "x5", "x6", "x7"],
            ["r1", "r2", "r3", "r4"]]:
    for a, b in zip(seq, seq[1:]):
        arrow([bot(a), top(b)])

# Yes labels below diamonds
for k in ["p3", "d4", "m4", "x4"]:
    label(nodes[k]["x"] + 0.07, nodes[k]["y"] - nodes[k]["h"] / 2 - 0.12, "Yes")

# ------------------------------------------------- feedback (No) loops -----
def no_loop(dec, target, side_x):
    a = left(dec)
    t = left(target)
    arrow([a, (side_x, a[1]), (side_x, t[1]), t], dashed=True, color=GREY)
    label(a[0] - 0.05, a[1] + 0.11, "No", ha="right")

no_loop("p3", "p2", COLX[0] - BW / 2 - 0.13)
no_loop("d4", "d2", COLX[1] - BW / 2 - 0.13)
no_loop("m4", "m2", COLX[2] - BW / 2 - 0.13)
no_loop("x4", "x3", COLX[3] - BW / 2 - 0.13)

# ------------------------------------------------- between-column arrows ---
def cross(src, dst):
    """From the right side of src, down/up through the gutter, into dst's left side."""
    s, d = right(src), left(dst)
    gx = (s[0] + d[0]) / 2 + 0.02
    arrow([s, (gx, s[1]), (gx, d[1]), d])

cross("p4", "d1")
cross("d5", "m1")
cross("m6", "x1")
cross("x7", "r1")

fig.tight_layout(pad=0.1)
for ext in ["pdf", "png"]:
    fig.savefig(C.FIG_DIR / f"FigM_methodology_flowchart.{ext}", dpi=400 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
log.info("Methodology flowchart written")
