"""
03 - PREX-Edu: pedagogical aggregation, faithfulness, normalisation
===================================================================
Outputs
  Table 1   theoretical grounding of the dimensions           T01_prex_dimensions
  Fig 9     PREX-Edu profile of a representative student      Fig09_prex_profile
  Table 12  global faithfulness ablation (PREX vs random)     T12_faithfulness_ablation
  Fig 10    accuracy / ROC-AUC degradation curves             Fig10_faithfulness_curves
  Table S7  instance-level deletion (AOPC): PREX vs SHAP vs random   S07_instance_aopc
  Fig S10   instance-level deletion curves                    FigS10_instance_deletion
  Table S8  dimension-level ablation vs mean R_k rank         S08_dimension_ablation
  Table A6a/b + Fig 11  normalised vs naive aggregation       A06a_/A06b_ / Fig11_normalized_vs_naive
  Table A8  per-student R_k, dominant dimension, margin       A08_per_student_prex
  Table S9  dominant-dimension distribution + coverage       S09_dominant_distribution
  Text      example natural-language explanations            outputs/tables/prex_explanations_examples.md
"""
from _common import C, Timer, get_logger

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score

from prexedu import data as D
from prexedu import plotting as P
from prexedu import prex
from prexedu import xai
from prexedu.tables import save_table

log = get_logger("03_prexedu")
FM = C.FINAL_MODEL
df, X, y = D.load_xy()
split = joblib.load(C.MODEL_DIR / "holdout_split.joblib")
Xtr, Xte = X.loc[split["train_idx"]], X.loc[split["test_idx"]]
ytr, yte = y[split["train_idx"]], y[split["test_idx"]]
pipe = joblib.load(C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[FM]}.joblib")
S = pd.read_csv(C.MODEL_DIR / f"shap_test_{C.MODEL_SHORT[FM]}.csv", index_col=0)
prob = pipe.predict_proba(Xte)[:, 1]
rep = pd.read_csv(C.MODEL_DIR / "representative_students.csv", index_col=0).iloc[:, 0]
ref = xai.reference_values(Xtr)
THR = xai.decision_threshold(FM)
log.info(f"decision threshold = {THR:.3f}")

R = prex.prex_scores(S)                      # operator chosen in config (C.PREX_AGGREGATION)
AGG_LABEL = "sum of |SHAP|" if C.PREX_AGGREGATION == "sum" else "mean |SHAP|"
Req3 = prex.prex_scores(S, normalize=True)  # Eq. 3 explicitly (for Fig 11 / Table A6)
Rn = prex.prex_scores(S, normalize=False)    # naive
Rs = prex.prex_scores(S, signed=True)        # signed (direction)
R.to_csv(C.MODEL_DIR / "prex_scores_test.csv")

# ---------------- Table 1 -------------------------------------------------- #
t1 = pd.DataFrame({
    "PREX-Edu dimension": list(C.PREX_DIMENSIONS),
    "Engagement–disaffection construct": ["Behavioural engagement (effort, persistence)",
                                          "Contextual antecedent (self-regulation of routines)",
                                          "Emotional engagement / disaffection (Pekrun, 2006)",
                                          "Behavioural engagement (presence)"],
    "Source features": [", ".join(C.LABELS[f] for f in v) for v in C.PREX_DIMENSIONS.values()],
    "|p_k|": [len(v) for v in C.PREX_DIMENSIONS.values()],
    "Typical intervention": [prex.INTERVENTIONS[k] for k in C.PREX_DIMENSIONS],
})
save_table(t1, "T01_prex_dimensions", "Theoretical grounding of the PREX-Edu pedagogical dimensions.",
           note="Socio-demographic features (age, gender, parent education, income, study space, extracurricular, "
                "difficult subject) are retained in the model but form a separate, non-actionable context block.")

# ---------------- Fig 9 PREX profile --------------------------------------- #
i = int(rep["high_risk_test_pos"])
fig, (ax, axt) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
r_row, s_row = R.iloc[i].sort_values(), Rs.iloc[i]
cols = [P.DIM_COLORS[k] for k in r_row.index]
ax.barh(range(4), r_row.values, color=cols, height=0.6)
for k, (d, v) in enumerate(r_row.items()):
    arrow = "▲ risk" if s_row[d] > 0 else "▼ protective"
    ax.text(v + 0.02 * r_row.max(), k, f"{v:.2f}  {arrow}", va="center", fontsize=7.8)
ax.set_yticks(range(4), r_row.index)
ax.set_xlim(0, r_row.max() * 1.45)
ax.set_xlabel(r"$R_k(x)$ = " + AGG_LABEL + " of dimension (log-odds)")
ax.grid(axis="y", visible=False)
ax.set_title(f"(a) PREX-Edu risk profile (p = {prob[i]:.2f})", loc="left")
axt.axis("off")
axt.set_title("(b) Educator-facing explanation", loc="left")
txt = prex.explain_text(R.iloc[i], Rs.iloc[i], prob[i])
import textwrap
axt.text(0, 0.95, "\n".join(textwrap.wrap(txt, 52)), va="top", fontsize=7.8, family="serif",
         bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec=P.GRID))
x_row = Xte.iloc[i]
facts = ", ".join(f"{C.LABELS[f]}: {D.decode_value(f, x_row[f])}" for f in
                  ["StudyHoursPerWeek", "Motivation(1-5)", "Stress(1-5)", "SleepHoursPerNight",
                   "ScreenTimeDaily", "HighAttendance"])
axt.text(0, 0.18, "\n".join(textwrap.wrap("Profile — " + facts, 60)), va="top", fontsize=6.8, color=P.MUTED)
fig.tight_layout()
P.save(fig, "Fig09_prex_profile")

# ---------------- Global faithfulness ablation (Table 12 / Fig 10) --------- #
gimp = S.abs().mean()
order_prex = prex.feature_order_from_prex(R.mean(), gimp)
order_shap = gimp.sort_values(ascending=False).index.tolist()
ks = list(range(0, len(X.columns) + 1))


def scores_after_removal(order, k):
    p = xai.masked_predict(pipe, Xte, order[:k], ref)
    return accuracy_score(yte, p >= .5), roc_auc_score(yte, p) if k < len(order) else 0.5


base_acc, base_auc = scores_after_removal(order_prex, 0)
prex_curve = np.array([scores_after_removal(order_prex, k) for k in ks])
shap_curve = np.array([scores_after_removal(order_shap, k) for k in ks])
rng = np.random.default_rng(C.SEED)
rand_curves = []
with Timer(log, f"{C.N_RANDOM_ABLATION} random removal orders"):
    for _ in range(C.N_RANDOM_ABLATION):
        o = list(rng.permutation(X.columns))
        rand_curves.append([scores_after_removal(o, k) for k in ks])
rand_curves = np.array(rand_curves)       # (trials, k, 2)

rows = []
for k in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
    for m, mname in [(1, "ROC-AUC"), (0, "Accuracy")]:
        base_v = base_auc if m == 1 else base_acc
        pd_ = base_v - prex_curve[k, m]
        rd = base_v - rand_curves[:, k, m]
        t, p_t = stats.ttest_1samp(rd, pd_)
        p_perm = (np.sum(rd >= pd_) + 1) / (len(rd) + 1)
        d = (pd_ - rd.mean()) / rd.std(ddof=1) if rd.std() > 0 else np.nan
        rows.append({"Metric": mname, "k removed": k, "% removed": round(100 * k / 13),
                     "Removed (PREX order)": ", ".join(C.LABELS[f] for f in order_prex[:k]) if k <= 3 else f"top {k}",
                     "PREX score": round(prex_curve[k, m], 4), "PREX drop": round(pd_, 4),
                     "Random drop mean [95% CI]": f"{rd.mean():.4f} [{np.percentile(rd, 2.5):.4f}, {np.percentile(rd, 97.5):.4f}]",
                     "Δ drop (PREX − random)": round(pd_ - rd.mean(), 4), "t": round(t, 2),
                     "p (t-test)": f"{p_t:.2e}", "p (permutation)": round(p_perm, 4), "Cohen's d": round(d, 2)})
t12 = pd.DataFrame(rows).sort_values(["Metric", "k removed"], ascending=[False, True])
_trap = getattr(np, "trapezoid", None) or np.trapz
audc = lambda c: _trap(c, dx=1) / (len(c) - 1)
save_table(t12, "T12_faithfulness_ablation",
           "Faithfulness ablation: performance drop when features are removed in PREX-Edu order vs random order.",
           note=f"Removal = replacement by the training mean (numeric) or mode (categorical). Baseline on the test set "
                f"(n = {len(yte)}): ROC-AUC = {base_auc:.4f}, accuracy = {base_acc:.4f}. Random: "
                f"{C.N_RANDOM_ABLATION} random orders. Area under the degradation curve (ROC-AUC): PREX = "
                f"{audc(prex_curve[:, 1]):.4f}, SHAP order = {audc(shap_curve[:, 1]):.4f}, random = "
                f"{audc(rand_curves[:, :, 1].mean(0)):.4f} (lower = more faithful). Accuracy is reported for "
                f"continuity but is uninformative at 5.5% prevalence; ROC-AUC is the primary faithfulness metric.")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, m, lab in [(axes[0], 1, "ROC-AUC"), (axes[1], 0, "Accuracy")]:
    lo, hi = np.percentile(rand_curves[:, :, m], [2.5, 97.5], axis=0)
    ax.fill_between(ks, lo, hi, color=P.MUTED, alpha=0.2, lw=0, label="Random order (95% band)")
    ax.plot(ks, rand_curves[:, :, m].mean(0), color=P.MUTED, ls="--", lw=1.2, label="Random order (mean)")
    ax.plot(ks, shap_curve[:, m], color=P.OKABE_ITO[2], ls="-.", marker="s", ms=3, lw=1.3, label="Global SHAP order")
    ax.plot(ks, prex_curve[:, m], color=P.OKABE_ITO[0], marker="o", ms=3.5, lw=1.8, label="PREX-Edu order")
    ax.set_xlabel("Number of features removed")
    ax.set_ylabel(f"Test {lab}")
    ax.set_title(f"({'a' if m == 1 else 'b'}) {lab} degradation", loc="left")
axes[1].axhline(1 - yte.mean(), color=P.RISK, ls=":", lw=0.9)
axes[1].text(12.8, 1 - yte.mean() + 0.002, "majority-class rate", ha="right", va="bottom", fontsize=7, color=P.RISK)
axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
P.save(fig, "Fig10_faithfulness_curves")

# ---------------- Instance-level deletion (AOPC) ---------------------------- #
pos = np.where(prob >= THR)[0]
if len(pos) < 10:
    pos = np.argsort(-prob)[:20]
L = len(X.columns)
curves = {"PREX-Edu": [], "SHAP": [], "Random": []}
for i in pos:
    x = Xte.iloc[[i]]
    orders = {"PREX-Edu": prex.instance_feature_order(S.iloc[i], R.iloc[i]),
              "SHAP": S.iloc[i].abs().sort_values(ascending=False).index.tolist()}
    for name, o in orders.items():
        curves[name].append([xai.masked_predict(pipe, x, o[:k], ref)[0] for k in range(L + 1)])
    rc = []
    for _ in range(20):
        o = list(rng.permutation(X.columns))
        rc.append([xai.masked_predict(pipe, x, o[:k], ref)[0] for k in range(L + 1)])
    curves["Random"].append(np.mean(rc, axis=0))
curves = {k: np.array(v) for k, v in curves.items()}
aopc = {k: (v[:, [0]] - v[:, 1:7]).mean(axis=1) for k, v in curves.items()}   # first 6 removals (≈ the 4 dims)
rows = []
for k in curves:
    rows.append({"Removal order": k, "Students": len(pos), "AOPC@6 mean": round(aopc[k].mean(), 4),
                 "AOPC@6 SD": round(aopc[k].std(ddof=1), 4),
                 "Mean prob. after 1 removal": round(curves[k][:, 1].mean(), 4),
                 "Mean prob. after 3 removals": round(curves[k][:, 3].mean(), 4)})
w1 = stats.wilcoxon(aopc["PREX-Edu"], aopc["Random"])
w2 = stats.wilcoxon(aopc["PREX-Edu"], aopc["SHAP"]) if not np.allclose(aopc["PREX-Edu"], aopc["SHAP"]) else None
save_table(pd.DataFrame(rows), "S07_instance_aopc",
           "Instance-level faithfulness: area over the perturbation curve (AOPC) for students flagged At-Risk.",
           note=f"Flagged = P(At-Risk) ≥ {THR:.3f} (F1-optimal threshold from training-set CV). AOPC@6 = mean drop in P(At-Risk) over the first six removals; higher = removed features mattered more. "
                f"Wilcoxon PREX vs random: W = {w1.statistic:.1f}, p = {w1.pvalue:.2e}"
                + (f"; PREX vs SHAP: W = {w2.statistic:.1f}, p = {w2.pvalue:.3f}" if w2 else "; PREX = SHAP order for all students")
                + ". SHAP order is the per-instance upper bound; PREX-Edu constrains removal to whole dimensions.")
fig, ax = plt.subplots(figsize=(4.8, 3.0))
for k, col, ls in [("PREX-Edu", P.OKABE_ITO[0], "-"), ("SHAP", P.OKABE_ITO[2], "-."), ("Random", P.MUTED, "--")]:
    m = curves[k].mean(0)
    se = curves[k].std(0, ddof=1) / np.sqrt(len(pos))
    ax.plot(range(L + 1), m, color=col, ls=ls, label=k)
    ax.fill_between(range(L + 1), m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.15, lw=0)
ax.set(xlabel="Features removed (instance-specific order)", ylabel="Mean P(At-Risk)")
ax.legend(fontsize=7.5)
ax.set_title(f"Instance-level deletion curves (n = {len(pos)} flagged At-Risk)", loc="left")
P.save(fig, "FigS10_instance_deletion")

# ---------------- Dimension-level ablation --------------------------------- #
rows = []
for k, feats in prex.dimension_map(include_context=True).items():
    p = xai.masked_predict(pipe, Xte, feats, ref)
    rows.append({"Dimension": k, "Features": len(feats), "Mean R_k":
                 round(prex.prex_scores(S, include_context=True)[k].mean(), 4),
                 "Mean |SHAP| per feature (Eq. 3 variant)": round(prex.prex_scores(S, normalize=True, include_context=True)[k].mean(), 4),
                 "ROC-AUC after removal": round(roc_auc_score(yte, p), 4),
                 "ΔROC-AUC": round(base_auc - roc_auc_score(yte, p), 4)})
s8 = pd.DataFrame(rows)
s8["Rank by R_k"] = s8["Mean R_k"].rank(ascending=False).astype(int)
s8["Rank by ΔAUC"] = s8["ΔROC-AUC"].rank(ascending=False).astype(int)
rho_dim = stats.spearmanr(s8["Mean R_k"], s8["ΔROC-AUC"])[0]
save_table(s8, "S08_dimension_ablation", "Dimension-level ablation: does the R_k ranking predict the performance cost of removing a dimension?",
           note=f"R_k = {AGG_LABEL} (operator set in config). Spearman ρ(mean R_k, ΔROC-AUC) = {rho_dim:.3f}. "
                "Context block shown for completeness.")

# ---------------- Normalised vs naive (Fig 11 / Table A6) ------------------ #
taus = []
for i in range(len(Req3)):
    t, _ = stats.kendalltau(Req3.iloc[i].values, Rn.iloc[i].values)
    taus.append(t)
taus = np.array(taus)
mismatch = (Req3.idxmax(axis=1) != Rn.idxmax(axis=1)).mean()
a6a = pd.DataFrame({"Metric": ["Mean Kendall's τ (per instance)", "SD Kendall's τ", "Median Kendall's τ",
                               "Top-1 dimension mismatch (%)", "N instances"],
                    "Value": [round(np.nanmean(taus), 4), round(np.nanstd(taus, ddof=1), 4), round(np.nanmedian(taus), 4),
                              f"{100 * mismatch:.2f}%", len(Req3)]})
save_table(a6a, "A06a_aggregation_summary", "Sensitivity of dimension rankings to the aggregation operator: sum vs cardinality-normalised mean.")
nr = Rn.mean().rank(ascending=False).astype(int)
rr = Req3.mean().rank(ascending=False).astype(int)
a6b = pd.DataFrame({"Dimension": Req3.columns, "Sum mean": Rn.mean().round(4).values,
                    "|p_k|": [len(C.PREX_DIMENSIONS[k]) for k in Req3.columns], "Mean-normalised mean": Req3.mean().round(4).values,
                    "Rank shift": [f"{nr[k]} → {rr[k]}" for k in Req3.columns],
                    "Top-1 share sum (%)": [round(100 * (Rn.idxmax(axis=1) == k).mean(), 1) for k in Req3.columns],
                    "Top-1 share mean-normalised (%)": [round(100 * (Req3.idxmax(axis=1) == k).mean(), 1) for k in Req3.columns]})
save_table(a6b, "A06b_dimension_scores", "Mean dimension scores under sum and cardinality-normalised (mean) aggregation.")

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
dims = list(Req3.columns)
for ax, Mx, title in [(axes[0], Rn, "(a) Sum (PREX-Edu)" if C.PREX_AGGREGATION == "sum" else "(a) Naive sum"), (axes[1], Req3, "(b) Cardinality-normalised (mean)")]:
    v = Mx.mean()[dims]
    bars = ax.bar(range(4), v, color=[P.DIM_COLORS[d] for d in dims], width=0.62)
    for b_, vv in zip(bars, v):
        ax.text(b_.get_x() + b_.get_width() / 2, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(4), dims, fontsize=7.5, rotation=25, ha="right")
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_ylabel("Mean score")
ax = axes[2]
ax.hist(taus[~np.isnan(taus)], bins=np.linspace(-1, 1, 21), color=P.OKABE_ITO[0], edgecolor="white", lw=0.5)
ax.axvline(np.nanmean(taus), color=P.RISK, ls="--", lw=1)
ax.axvline(0, color=P.MUTED, ls=":", lw=0.8)
ax.text(np.nanmean(taus) - 0.04, ax.get_ylim()[1] * 0.92, f"mean τ = {np.nanmean(taus):.3f}", ha="right", fontsize=7.5)
ax.set(xlabel="Per-instance Kendall τ", ylabel="Students")
ax.set_title(f"(c) Rank agreement; top-1 mismatch {100 * mismatch:.1f}%", loc="left", fontsize=9.5)
fig.tight_layout()
P.save(fig, "Fig11_normalized_vs_naive")

# ---------------- Table A8 per-student ------------------------------------- #
pred_pos = np.where(prob >= THR)[0]
pick = pred_pos[np.argsort(-prob[pred_pos])][:10]
dom = prex.dominant(R)
a8 = pd.DataFrame({"Student ID": Xte.index[pick] + 1, "P(At-Risk)": prob[pick].round(3),
                   "True label": np.where(yte[pick] == 1, "At-Risk", "Not At-Risk")})
for k in dims:
    a8[k] = R.iloc[pick][k].values.round(3)
a8["Dominant"] = dom.iloc[pick]["Dominant"].values
a8["Margin"] = dom.iloc[pick]["Margin"].values.round(3)
save_table(a8, "A08_per_student_prex",
           "Per-student PREX-Edu risk magnitudes for the 10 flagged test students with the highest predicted risk.",
           note="Student ID = row number in the original dataset. Margin = dominant minus second-ranked R_k; "
                "a larger margin indicates a clearer single intervention target.")

# ---------------- Dominant-dimension distribution + coverage --------------- #
cov = prex.coverage(S)
rows = []
for grp, mask in [("All test students", np.ones(len(R), bool)), ("Flagged At-Risk (p ≥ threshold)", prob >= THR),
                  ("True At-Risk", yte == 1)]:
    d = dom[mask]["Dominant"].value_counts(normalize=True)
    row = {"Group": grp, "n": int(mask.sum())}
    for k in dims:
        row[f"{k} dominant (%)"] = round(100 * d.get(k, 0), 1)
    row["Median margin"] = round(dom[mask]["Margin"].median(), 3)
    row["4-dimension |SHAP| coverage (median %)"] = round(100 * cov[mask].median(), 1)
    rows.append(row)
save_table(pd.DataFrame(rows), "S09_dominant_distribution",
           "Distribution of the dominant PREX-Edu dimension and attribution coverage.",
           note="Coverage = share of a student's total |SHAP| captured by the four actionable dimensions; the remainder "
                "belongs to non-actionable context features.")

# ---------------- Example explanations ------------------------------------- #
lines = ["# Example PREX-Edu explanations (test students with highest predicted risk)\n"]
for i in pick[:5]:
    lines.append(f"**Student {Xte.index[i] + 1}** — " + prex.explain_text(R.iloc[i], Rs.iloc[i], prob[i]) + "\n")
(C.TAB_DIR / "prex_explanations_examples.md").write_text("\n".join(lines), encoding="utf-8")
log.info("done")
