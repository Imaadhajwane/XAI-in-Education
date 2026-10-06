"""
04 - Fairness & subgroup analysis (first-draft analyses, rebuilt)
================================================================
Outputs
  Table 13  demographic parity by gender / income (hold-out)    T13_demographic_parity
  Fig 12    mean |SHAP| of top features by subgroup             Fig12_subgroup_shap
  Table A7a mean |SHAP| by gender                               A07a_shap_by_gender
  Table A7b mean |SHAP| by income                               A07b_shap_by_income
  Table S10 PREX-Edu dominant dimension by subgroup + χ² test   S10_prex_by_subgroup
The extended, properly-powered fairness audit on all 1,208 students is in
script 14.
"""
from _common import C, get_logger

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from prexedu import data as D
from prexedu import plotting as P
from prexedu import prex
from prexedu.tables import save_table

log = get_logger("04_fairness")
FM = C.FINAL_MODEL
df, X, y = D.load_xy()
split = joblib.load(C.MODEL_DIR / "holdout_split.joblib")
Xte = X.loc[split["test_idx"]]
dte = df.loc[split["test_idx"]]
yte = y[split["test_idx"]]
pipe = joblib.load(C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[FM]}.joblib")
prob = pipe.predict_proba(Xte)[:, 1]
from prexedu import xai
THR = xai.decision_threshold(FM)
pred = (prob >= THR).astype(int)
S = pd.read_csv(C.MODEL_DIR / f"shap_test_{C.MODEL_SHORT[FM]}.csv", index_col=0)
R = prex.prex_scores(S)

GROUPS = {"Gender": ["Female", "Male", "Prefer not to say"], "IncomeRange": ["High", "Medium", "Low"]}

# ---------------- Table 13 ------------------------------------------------- #
rows = []
for g, levels in GROUPS.items():
    ref_rate = pred[dte[g].values == levels[0]].mean()
    for lvl in levels:
        m = dte[g].values == lvl
        n, npos = int(m.sum()), int(yte[m].sum())
        rate = pred[m].mean() if n else np.nan
        tpr = pred[m & (yte == 1)].mean() if npos else np.nan
        rows.append({"Attribute": C.LABELS.get(g, g), "Subgroup": lvl, "N (test)": n, "N At-Risk (true)": npos,
                     "Base rate": round(npos / n, 4) if n else np.nan, "P(ŷ=1 | group)": round(rate, 4),
                     "Parity diff. vs reference": "— (reference)" if lvl == levels[0] else round(rate - ref_rate, 4),
                     "Disparate-impact ratio": "—" if lvl == levels[0] else (round(rate / ref_rate, 3) if ref_rate > 0 else "n/a"),
                     "TPR (recall) in group": round(tpr, 3) if npos else "n/a"})
save_table(pd.DataFrame(rows), "T13_demographic_parity",
           "Demographic parity of At-Risk predictions by gender and family income (hold-out test set).",
           note=f"P(ŷ=1 | group) is the share of the subgroup flagged At-Risk (threshold {THR:.3f}, F1-optimal on training CV). Disparate-impact ratio "
                "< 0.8 or > 1.25 would breach the four-fifths rule. With ≤ 20 true At-Risk students in the test set, "
                "subgroup TPRs are highly uncertain; see script 14 for the full-sample audit with CIs.")

# ---------------- Fig 12 / Table A7 ---------------------------------------- #
top = S.abs().mean().sort_values(ascending=False).index[:6].tolist()
for g, name in [("Gender", "A07a_shap_by_gender"), ("IncomeRange", "A07b_shap_by_income")]:
    t = S[top].abs().groupby(dte[g].values).mean().reindex(GROUPS[g]).round(3)
    t.columns = [C.LABELS[c] for c in t.columns]
    kw = {c: stats.kruskal(*[S.loc[dte[g].values == l, c].abs() for l in GROUPS[g] if (dte[g].values == l).sum() > 1]).pvalue
          for c in top}
    t.loc["Kruskal–Wallis p"] = [round(kw[c], 3) for c in top]
    save_table(t.rename_axis(C.LABELS.get(g, g)).reset_index(), name,
               f"Mean absolute SHAP value of the six most important features by {C.LABELS.get(g, g).lower()}.",
               note="Last row: Kruskal–Wallis test of equal |SHAP| distributions across subgroups (no multiplicity correction).",
               )

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
pal = ["#0072B2", "#E69F00", "#009E73"]
for ax, (g, levels) in zip(axes, GROUPS.items()):
    w = 0.8 / len(levels)
    for j, lvl in enumerate(levels):
        m = dte[g].values == lvl
        mu = S.loc[m, top].abs().mean()
        se = S.loc[m, top].abs().std(ddof=1) / np.sqrt(m.sum())
        ax.bar(np.arange(len(top)) + (j - (len(levels) - 1) / 2) * w, mu, w * 0.92, yerr=1.96 * se,
               color=pal[j], label=f"{lvl} (n={m.sum()})", error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(range(len(top)), [C.LABELS[c] for c in top], fontsize=7.5, rotation=28, ha="right")
    ax.set_title(f"({'a' if g == 'Gender' else 'b'}) by {C.LABELS.get(g, g).lower()}", loc="left")
    ax.legend(fontsize=7, loc="upper right")
axes[0].set_ylabel("Mean |SHAP| (± 95% CI)")
fig.tight_layout()
P.save(fig, "Fig12_subgroup_shap")

# ---------------- PREX dominant dimension by subgroup ---------------------- #
dom = prex.dominant(R)["Dominant"]
rows = []
for g in ["Gender", "IncomeRange", "ParentEducation"]:
    ct = pd.crosstab(dte[g].values, dom.values)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    for lvl in ct.index:
        r_ = {"Attribute": C.LABELS.get(g, g), "Subgroup": lvl, "n": int(ct.loc[lvl].sum())}
        for k in C.PREX_DIMENSIONS:
            r_[f"{k} (%)"] = round(100 * ct.loc[lvl].get(k, 0) / ct.loc[lvl].sum(), 1)
        r_["χ² p (attribute)"] = round(p, 3)
        rows.append(r_)
save_table(pd.DataFrame(rows), "S10_prex_by_subgroup",
           "Distribution of the dominant PREX-Edu dimension by demographic subgroup (test set).",
           note="χ² test of independence between subgroup and dominant dimension. Non-significant p indicates that the "
                "pedagogical explanation a teacher receives does not systematically depend on the student's demographics.")
log.info("done")
