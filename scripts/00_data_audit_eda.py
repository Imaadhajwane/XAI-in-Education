"""
00 - Data audit & exploratory analysis
=====================================
Outputs
  Table 4   class distribution                           -> T04_class_distribution
  Table S1  variable audit (types, missing, ranges)      -> S01_data_audit
  Table S2  descriptive statistics by class + tests      -> S02_descriptives_by_class
  Table S3  multicollinearity (VIF)                      -> S03_vif
  Fig S1    target construction (LastTermPercentage)     -> FigS1_target_construction
  Fig S2    correlation matrix (Spearman, incl. target)  -> FigS2_correlation_matrix
  Fig S3    behavioural predictors by class              -> FigS3_predictors_by_class
  Fig S4    At-Risk rate by categorical predictor        -> FigS4_risk_rate_by_category
"""
from _common import C, get_logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from prexedu import data as D
from prexedu import plotting as P
from prexedu.tables import save_table

log = get_logger("00_eda")

raw = D.load_raw()
df = D.clean(raw)
X = D.encode(df)
y = df[C.TARGET].values
df.to_csv(C.DATA_PROCESSED / "clean_dataset.csv", index=False)
X.assign(AtRisk=y).to_csv(C.DATA_PROCESSED / "encoded_dataset.csv", index=False)
log.info(f"N = {len(df)}, At-Risk = {y.sum()} ({100 * y.mean():.2f}%)")

# ---------------- Table 4 -------------------------------------------------- #
t4 = pd.DataFrame({
    "Class": ["At-Risk", "Not At-Risk", "Total"],
    "Label": [1, 0, ""],
    "Count": [int(y.sum()), int((1 - y).sum()), len(y)],
    "Percentage": [f"{100 * y.mean():.2f}%", f"{100 * (1 - y.mean()):.2f}%", "100.00%"],
})
save_table(t4, "T04_class_distribution",
           f"Class distribution of the binary target (At-Risk = LastTermPercentage ≤ {C.RISK_THRESHOLD:.0f}%).")

# ---------------- Table S1 ------------------------------------------------- #
save_table(D.data_audit(raw), "S01_data_audit",
           "Variable-level audit of the primary dataset (before cleaning).",
           note="DifficultSubject has missing responses; these are coded as an explicit 'Unknown' level.")

# ---------------- Table S2 ------------------------------------------------- #
rows = []
g0, g1 = df[df.AtRisk == 0], df[df.AtRisk == 1]
for c in C.NUMERIC:
    u, p = stats.mannwhitneyu(g1[c], g0[c], alternative="two-sided")
    rbc = 1 - 2 * u / (len(g1) * len(g0))           # rank-biserial correlation
    rows.append({"Variable": C.LABELS.get(c, c), "Level": "mean ± SD",
                 "Not At-Risk (n=%d)" % len(g0): f"{g0[c].mean():.2f} ± {g0[c].std():.2f}",
                 "At-Risk (n=%d)" % len(g1): f"{g1[c].mean():.2f} ± {g1[c].std():.2f}",
                 "Test": "Mann–Whitney U", "Statistic": round(u, 1), "p": p,
                 "Effect size": f"r_rb = {-rbc:.2f}"})
for c in list(C.ORDINAL) + list(C.NOMINAL) + C.BINARY:
    ct = pd.crosstab(df[c], df.AtRisk)
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    test = "χ²"
    if ct.shape == (2, 2) and (exp < 5).any():
        _, p = stats.fisher_exact(ct.values)
        test = "Fisher exact"
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    for i, lvl in enumerate(ct.index):
        rows.append({"Variable": C.LABELS.get(c, c) if i == 0 else "", "Level": lvl,
                     "Not At-Risk (n=%d)" % len(g0): f"{ct.loc[lvl, 0]} ({100 * ct.loc[lvl, 0] / len(g0):.1f}%)",
                     "At-Risk (n=%d)" % len(g1): f"{ct.loc[lvl, 1]} ({100 * ct.loc[lvl, 1] / len(g1):.1f}%)",
                     "Test": test if i == 0 else "", "Statistic": round(chi2, 2) if i == 0 else "",
                     "p": p if i == 0 else np.nan, "Effect size": f"Cramér's V = {v:.2f}" if i == 0 else ""})
s2 = pd.DataFrame(rows)
s2["p"] = s2["p"].apply(lambda v: "" if pd.isna(v) else ("< .001" if v < 0.001 else f"{v:.3f}"))
save_table(s2, "S02_descriptives_by_class",
           "Descriptive statistics of the 13 predictors by outcome class.",
           note="Continuous / Likert: Mann–Whitney U with rank-biserial r. Categorical: χ² (Fisher exact for sparse 2×2) with Cramér's V.")

# ---------------- Table S3 VIF -------------------------------------------- #
Xv = (X - X.mean()) / X.std()
Xv = Xv.assign(const=1.0)
vif = pd.DataFrame({"Feature": [C.LABELS.get(c, c) for c in X.columns],
                    "VIF": [variance_inflation_factor(Xv.values, i) for i in range(X.shape[1])]})
vif["VIF"] = vif["VIF"].round(3)
save_table(vif, "S03_vif", "Variance inflation factors of the 13 predictors.",
           note="All VIF < 5 indicates no problematic multicollinearity.")

# ---------------- Fig S1 target ------------------------------------------- #
fig, ax = plt.subplots(figsize=(6.2, 2.8))
bins = np.arange(30, 101, 2)
ax.hist(df.loc[df.AtRisk == 0, C.TARGET_SOURCE], bins=bins, color="#9ca3af", label="Not At-Risk",
        edgecolor="white", linewidth=0.6)
ax.hist(df.loc[df.AtRisk == 1, C.TARGET_SOURCE], bins=bins, color=P.RISK, label="At-Risk",
        edgecolor="white", linewidth=0.6)
ax.axvline(C.RISK_THRESHOLD, color=P.INK, ls="--", lw=1)
ax.text(C.RISK_THRESHOLD + 0.8, ax.get_ylim()[1] * 0.92, f"τ = {C.RISK_THRESHOLD:.0f}%", fontsize=8.5)
ax.set_xlabel("Last-term percentage (%)")
ax.set_ylabel("Students")
ax.legend(loc="upper right")
ax.set_title("Construction of the binary At-Risk label", loc="left")
P.save(fig, "FigS1_target_construction")

# ---------------- Fig S2 correlation -------------------------------------- #
corr = X.assign(**{"At-Risk": y, "LastTerm %": df[C.TARGET_SOURCE]}).corr(method="spearman")
corr.index = corr.columns = [C.LABELS.get(c, c) for c in corr.columns]
fig, ax = plt.subplots(figsize=(7.2, 6.2))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
im = ax.imshow(np.ma.masked_array(corr.values, mask), cmap=P.DIVERGING, vmin=-1, vmax=1)
for i in range(len(corr)):
    for j in range(i + 1):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(v) > 0.55 else P.INK)
ax.set_xticks(range(len(corr)), corr.columns, rotation=55, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Spearman ρ")
ax.set_title("Spearman correlation matrix (predictors, target and its source)", loc="left")
P.save(fig, "FigS2_correlation_matrix")

# ---------------- Fig S3 predictors by class ------------------------------- #
feats = ["StudyHoursPerWeek", "Motivation(1-5)", "Stress(1-5)", "SleepHoursPerNight",
         "ScreenTimeDaily", "Age"]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
for ax, f in zip(axes.ravel(), feats):
    data = [df.loc[df.AtRisk == 0, f], df.loc[df.AtRisk == 1, f]]
    vp = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, col in zip(vp["bodies"], ["#9ca3af", P.RISK]):
        body.set_facecolor(col)
        body.set_alpha(0.45)
        body.set_edgecolor("none")
    ax.boxplot(data, widths=0.18, showfliers=False, medianprops=dict(color=P.INK, lw=1.4),
               boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    ax.set_xticks([1, 2], ["Not At-Risk", "At-Risk"])
    ax.set_title(C.LABELS[f], loc="left", fontsize=9.5)
    p = stats.mannwhitneyu(*data).pvalue
    ax.set_title(C.LABELS[f] + ("   (p < .001)" if p < 0.001 else f"   (p = {p:.3f})"),
                 loc="left", fontsize=9.5)
fig.suptitle("Distribution of behavioural predictors by outcome class", x=0.01, ha="left",
             fontweight="bold", fontsize=10.5)
fig.tight_layout()
P.save(fig, "FigS3_predictors_by_class")

# ---------------- Fig S4 categorical risk rates (forest layout) ----------- #
cats = ["Gender", "IncomeRange", "ParentEducation", "StudySpace", "Extracurricular",
        "HighAttendance", "DifficultSubject"]
base = y.mean()
labels, rates, los, his, ypos, heads = [], [], [], [], [], []
pos = 0
for c in cats:
    heads.append((pos, C.LABELS.get(c, c)))
    pos += 1
    for lvl in (C.ORDINAL.get(c) or C.NOMINAL.get(c) or ["No", "Yes"]):
        sub = df.loc[df[c] == lvl, C.TARGET]
        n, k = len(sub), int(sub.sum())
        lo, hi = stats.binomtest(k, n).proportion_ci(method="wilson")
        labels.append(f"{lvl}  (n = {n})"); rates.append(k / n); los.append(lo); his.append(hi)
        ypos.append(pos); pos += 1
    pos += 0.4
fig, ax = plt.subplots(figsize=(5.6, 6.4))
ax.axvline(base, color=P.RISK, ls="--", lw=0.9, label=f"Overall prevalence ({base:.1%})")
ax.errorbar(rates, ypos, xerr=[np.array(rates) - los, np.array(his) - rates], fmt="o",
            color=P.OKABE_ITO[0], ecolor=P.OKABE_ITO[0], ms=5, lw=1.2, capsize=0,
            label="At-Risk rate (Wilson 95% CI)")
ax.set_yticks(ypos, labels, fontsize=8)
for yp, h in heads:
    ax.text(-0.005, yp, h, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontweight="bold", fontsize=8.5)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Proportion At-Risk")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=7.5)
ax.set_title("At-Risk prevalence by categorical predictor", loc="left")
P.save(fig, "FigS4_risk_rate_by_category")
log.info("done")
