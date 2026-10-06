"""
One shared, journal-ready figure style.

* Serif type (Liberation Serif ≈ Times New Roman metrics) at 9-10 pt.
* Okabe-Ito colour-blind-safe categorical palette, assigned to models in a
  FIXED order (colour follows the model, never its rank).
* Every figure is saved as 600-dpi PNG + vector PDF (+ SVG) so the journal can
  take whichever it prefers.
* Recessive grid, no top/right spines, legends always present for ≥2 series,
  line-style as secondary encoding for colour-vision deficiency / greyscale.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#e5e7eb"

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MODEL_COLORS = dict(zip(C.MODEL_ORDER, OKABE_ITO))
MODEL_LS = dict(zip(C.MODEL_ORDER, ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]))
MODEL_MARK = dict(zip(C.MODEL_ORDER, ["o", "s", "^", "D", "v", "P"]))

DIM_COLORS = {"Engagement": "#0072B2", "Lifestyle": "#009E73", "Cognitive Load": "#D55E00",
              "Participation": "#CC79A7", "Context (non-actionable)": "#9ca3af"}
RISK = "#B2182B"        # pushes toward At-Risk
PROTECT = "#2166AC"     # pushes toward Not At-Risk
DIVERGING = LinearSegmentedColormap.from_list("prex_div", [PROTECT, "#f7f7f7", RISK])
SEQ = LinearSegmentedColormap.from_list("prex_seq", ["#f7fbff", "#6baed6", "#08306b"])


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "figure.dpi": 110,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,          # editable text in Illustrator / Word
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(fig, name: str, formats=("png", "pdf")):
    for ext in formats:
        fig.savefig(C.FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    return C.FIG_DIR / f"{name}.png"


def panel_label(ax, s: str, x: float = -0.12, y: float = 1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom",
            ha="left", color=INK)


set_style()
