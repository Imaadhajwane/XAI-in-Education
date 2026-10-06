"""
07 - Conceptual comparison of XAI techniques (Tables 2-3, Figure 2)
===================================================================
The 0-10 scores are an expert/literature-based rubric (they are NOT computed
from the data). They are stored in one editable table below so the rubric is
transparent and reproducible; the manuscript should state how they were
derived (e.g. two independent raters + consensus) and cite sources per cell.
"""
from _common import C, get_logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prexedu import plotting as P
from prexedu.tables import save_table

log = get_logger("07_xai_methods")

DIMS = ["Global\ninterpretability", "Local\ninterpretability", "Non-linear\nhandling", "Explanation\ndepth",
        "Teacher-\nfriendliness", "Computational\nefficiency"]
SCORES = pd.DataFrame({
    "SHAP": [9, 6, 9, 9, 5, 3],
    "LIME": [3, 9, 6, 7, 7, 7],
    "PDP": [6, 3, 6, 5, 8, 8],
    "Permutation Importance": [6, 2, 6, 4, 6, 7],
    "Chi-Square": [3, 1, 1, 3, 6, 9],
    "XGBoost Gain": [6, 1, 9, 4, 3, 8],
    "Mutual Information": [3, 1, 3, 3, 3, 7],
    "ANOVA F-test": [3, 1, 1, 3, 5, 9],
}, index=[d.replace("-\n", "-").replace("\n", " ") for d in DIMS]).T
FAMILY = {"SHAP": "XAI", "LIME": "XAI", "PDP": "XAI"}
SCORES.to_csv(C.TAB_DIR / "_xai_rubric_scores.csv")

# ---------------- Tables 2 & 3 (conceptual) -------------------------------- #
save_table(pd.DataFrame({
    "Criterion": ["Explanation level", "Domain awareness", "Actionability", "Educator interpretability", "Ethical transparency"],
    "SHAP / LIME": ["Feature-based", "No", "Limited", "Moderate", "Implicit"],
    "PREX-Edu": ["Pedagogical-dimension-based", "Yes", "Designed for high actionability (computationally supported; educator validation pending)",
                 "Expected strong (pending powered study)", "Explicit (dimension definitions + coverage reported)"]}),
    "T02_shap_lime_vs_prex", "Criterion-based comparison of SHAP/LIME and PREX-Edu.")
save_table(pd.DataFrame({
    "Method": ["SHAP", "LIME", "PDP", "PREX-Edu (proposed)"],
    "Strengths": ["Axiomatic (Shapley) attribution; consistent global/local view",
                  "Simple, instance-level, model-agnostic", "Visual population-level trends",
                  "Pedagogically grounded; inherits SHAP local accuracy at dimension level"],
    "Weaknesses": ["Computational cost; feature-level output is technical",
                   "Unstable local surrogate; no global consistency", "Ignores interactions; extrapolates under correlation",
                   "Depends on the mapping and on the underlying attribution method"],
    "Best use case": ["Auditing model behaviour", "Case-level teacher-facing explanation",
                      "Policy-level trend analysis", "Translating risk predictions into intervention areas"]}),
    "T03_xai_techniques", "Comparative analysis of XAI techniques.")
save_table(SCORES.assign(**{"Composite (mean)": SCORES.mean(axis=1).round(2)}).reset_index()
           .rename(columns={"index": "Method"}).sort_values("Composite (mean)", ascending=False),
           "S17_xai_rubric", "Literature-based rubric scores (0–10) underlying Figure 2.",
           note="Composite = arithmetic mean of the six dimensions (Eq. 4).")

# ---------------- Figure 2 ------------------------------------------------- #
comp = SCORES.mean(axis=1)
fig = plt.figure(figsize=(7.6, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.95)
ax = fig.add_subplot(gs[0])
im = ax.imshow(SCORES.values, cmap=P.DIVERGING, vmin=0, vmax=10, aspect="auto")
for i in range(SCORES.shape[0]):
    for j in range(SCORES.shape[1]):
        v = SCORES.values[i, j]
        ax.text(j, i, v, ha="center", va="center", fontsize=8, color="white" if v >= 8.5 or v <= 1.5 else P.INK,
                fontweight="bold" if i < 3 else "normal")
ax.set_xticks(range(6), [d.replace("-\n", "-").replace("\n", " ") for d in DIMS], fontsize=7, rotation=35, ha="right")
ax.set_yticks(range(8), SCORES.index, fontsize=8)
ax.axhline(2.5, color=P.INK, lw=1.2, ls="--")
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.ax.set_title("Score", fontsize=7.5, loc="left")
ax.set_title("(a) Per-dimension interpretability scores", loc="left", fontsize=9.5)
ax2 = fig.add_subplot(gs[1])
order = comp.sort_values().index
cols = [P.OKABE_ITO[0] if FAMILY.get(m) == "XAI" else "#9ca3af" for m in order]
ax2.barh(range(8), comp[order], color=cols, height=0.62)
for k, v in enumerate(comp[order]):
    ax2.text(v + 0.1, k, f"{v:.1f}", va="center", fontsize=7.5)
ax2.axvline(comp.mean(), color=P.RISK, ls=":", lw=0.9)
ax2.text(comp.mean() + 0.1, -0.75, f"mean {comp.mean():.1f}", fontsize=7, color=P.RISK, ha="left")
ax2.set_yticks(range(8), order, fontsize=8)
ax2.set_xlim(0, 8.2)
ax2.set_ylim(-1.0, 7.6)
ax2.set_xlabel("Composite score (mean of 6)")
ax2.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color=P.OKABE_ITO[0], label="XAI methods"), Patch(color="#9ca3af", label="Classical")],
           fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
ax2.set_title("(b) Composite ranking", loc="left", fontsize=9.5)
P.save(fig, "Fig02_xai_method_comparison")
log.info("done")
