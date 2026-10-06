"""
02 - Explainability of the final model: SHAP, LIME, PDP/ICE, PFI
================================================================
Explained model: tuned Logistic Regression from script 01 (C.FINAL_MODEL),
held-out test set (n = 363).

Outputs
  Fig 4     LIME - representative high-risk student          Fig04_lime_high_risk
  Fig 5     LIME - representative low-risk student           Fig05_lime_low_risk
  Fig 6     SHAP beeswarm (global)                           Fig06_shap_beeswarm
  Fig 7     SHAP waterfall (local, at-risk student)          Fig07_shap_waterfall
  Fig 8     PDP + ICE for StudyHoursPerWeek                  Fig08_pdp_studyhours
  Fig S7    PDP grid for the six actionable features         FigS7_pdp_grid
  Fig S8    SHAP rank-stability heat-map (10 seeds)          FigS8_shap_stability
  Fig S9    global importance: SHAP vs PFI vs |coef|         FigS9_importance_comparison
  Table 10  SHAP rank stability summary                      T10_shap_stability
  Table A4  10x10 Spearman matrix                            A04_shap_stability_matrix
  Table 11  SHAP-LIME agreement summary                      T11_shap_lime_agreement
  Table A5  per-instance SHAP-LIME agreement (50)            A05_shap_lime_instances
  Table S6  global importance (SHAP, PFI, odds ratios)       S06_global_importance
  Data      outputs/models/shap_test_LR.csv (used by 03/04/05)
"""
from _common import C, Timer, get_logger

import itertools

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import permutation_importance

from prexedu import data as D
from prexedu import models as M
from prexedu import plotting as P
from prexedu import xai
from prexedu.tables import save_table

log = get_logger("02_xai")
FM = C.FINAL_MODEL
df, X, y = D.load_xy()
split = joblib.load(C.MODEL_DIR / "holdout_split.joblib")
Xtr, Xte = X.loc[split["train_idx"]], X.loc[split["test_idx"]]
ytr, yte = y[split["train_idx"]], y[split["test_idx"]]
pipe = joblib.load(C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[FM]}.joblib")
prob = pipe.predict_proba(Xte)[:, 1]
LBL = lambda cols: [C.LABELS.get(c, c) for c in cols]

# ---------------- SHAP on the test set ------------------------------------- #
with Timer(log, "SHAP"):
    S, base, space = xai.compute_shap(pipe, Xtr, Xte, FM, n_bg=len(Xtr))
S.index = Xte.index
S.to_csv(C.MODEL_DIR / f"shap_test_{C.MODEL_SHORT[FM]}.csv")
pd.Series({"base_value": base, "space": space}).to_csv(C.MODEL_DIR / "shap_base.csv")
pd.DataFrame({"idx": Xte.index, "y": yte, "prob": prob}).to_csv(C.MODEL_DIR / "test_predictions_final.csv", index=False)
log.info(f"SHAP space={space}, base={base:.3f}")
gimp = xai.global_importance(S)

# representative students
tp = np.where((yte == 1) & (prob >= 0.5))[0]
hi_i = tp[np.argsort(prob[tp])[len(tp) // 2]] if len(tp) else int(np.argmax(prob))   # median TP
tn = np.where((yte == 0) & (prob < 0.5))[0]
lo_i = tn[np.argmin(prob[tn])]
pd.Series({"high_risk_test_pos": hi_i, "low_risk_test_pos": lo_i,
           "high_risk_idx": Xte.index[hi_i], "low_risk_idx": Xte.index[lo_i]}).to_csv(
    C.MODEL_DIR / "representative_students.csv")

# ---------------- Fig 6 beeswarm ------------------------------------------- #
order = gimp.index.tolist()
fig, ax = plt.subplots(figsize=(6.4, 4.6))
Z = (Xte - Xtr.mean()) / Xtr.std().replace(0, 1)
rng = np.random.default_rng(0)
for r, f in enumerate(order[::-1]):
    v = S[f].values
    # simple beeswarm: jitter proportional to local density
    hist, edges = np.histogram(v, bins=60)
    dens = hist[np.clip(np.digitize(v, edges[1:-1]), 0, len(hist) - 1)]
    jitter = rng.uniform(-1, 1, len(v)) * 0.38 * dens / dens.max()
    zc = np.clip(Z[f].values, -2, 2)
    sc = ax.scatter(v, r + jitter, c=zc, cmap=P.DIVERGING, vmin=-2, vmax=2, s=6, lw=0, alpha=0.9)
ax.axvline(0, color=P.MUTED, lw=0.8)
ax.set_yticks(range(len(order)), LBL(order[::-1]))
ax.grid(axis="y", visible=False)
ax.set_xlabel(f"SHAP value (impact on {space} of At-Risk)")
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, ticks=[-2, 0, 2])
cb.ax.set_yticklabels(["Low", "Mid", "High"])
cb.set_label("Feature value (standardised)")
ax.set_title(f"Global SHAP attributions — {FM} (test set, n = {len(Xte)})", loc="left")
P.save(fig, "Fig06_shap_beeswarm")

# ---------------- Fig 7 waterfall ------------------------------------------ #
def waterfall(ax, s_row, x_row, base_val, title, max_show=10):
    s_row = s_row.reindex(s_row.abs().sort_values(ascending=False).index)
    show = s_row.iloc[:max_show]
    rest = s_row.iloc[max_show:].sum()
    items = list(show.items()) + ([("Other features", rest)] if len(s_row) > max_show else [])
    items = items[::-1]                                   # smallest at the bottom
    start, ys, ends = base_val, [], [base_val]
    for k, (f, v) in enumerate(items):
        ax.barh(k, v, left=start, color=P.RISK if v > 0 else P.PROTECT, height=0.62)
        start += v
        ends.append(start)
        lab = C.LABELS.get(f, f)
        if f in x_row.index:
            lab = f"{lab} = {D.decode_value(f, x_row[f])}"
        ys.append(lab)
    span = max(ends) - min(ends)
    pad = 0.04 * span
    start = base_val
    for k, (f, v) in enumerate(items):
        end = start + v
        ax.text(max(start, end) + pad, k, f"{v:+.2f}", va="center", ha="left", fontsize=7.5, color=P.INK)
        start = end
    ax.set_xlim(min(ends) - 0.05 * span, max(ends) + 0.16 * span)
    ax.set_ylim(-1.3, len(items) - 0.4)
    ax.set_yticks(range(len(items)), ys, fontsize=8)
    ax.axvline(base_val, color=P.MUTED, ls=":", lw=0.9)
    ax.axvline(start, color=P.INK, ls="--", lw=0.9)
    ax.text(base_val, -1.0, f" E[f(x)] = {base_val:.2f}", fontsize=7.5, ha="left", color=P.MUTED, va="center")
    ax.text(start, -1.0, f"f(x) = {start:.2f} ", fontsize=7.5, ha="right", color=P.INK, va="center")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")


fig, ax = plt.subplots(figsize=(6.2, 4.2))
waterfall(ax, S.iloc[hi_i], Xte.iloc[hi_i], base,
          f"SHAP waterfall — representative At-Risk student (p = {prob[hi_i]:.2f})")
ax.set_xlabel(f"Model output ({space}); red = toward At-Risk, blue = toward Not At-Risk")
P.save(fig, "Fig07_shap_waterfall")

# ---------------- Fig 4 / 5 LIME ------------------------------------------- #
lime_ex = xai.make_lime_explainer(Xtr)


def lime_plot(i, title, name):
    w, exp = xai.lime_explain(lime_ex, pipe, Xte.values[i], X.columns)
    lst = exp.as_list(label=1)[:10]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = [l for l, _ in lst][::-1]
    vals = [v for _, v in lst][::-1]
    cols = [P.RISK if v > 0 else P.PROTECT for v in vals]
    ax.barh(range(len(vals)), vals, color=cols, height=0.62)
    for k, v in enumerate(vals):
        ax.text(v + (0.002 if v > 0 else -0.002), k, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=7.5)
    ax.set_yticks(range(len(vals)), labels, fontsize=7.8)
    ax.axvline(0, color=P.MUTED, lw=0.8)
    lim = max(abs(np.array(vals))) * 1.35
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("LIME weight (local linear surrogate, P(At-Risk))")
    ax.set_title(title, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=P.RISK, label="Increases At-Risk probability"),
                       Patch(color=P.PROTECT, label="Decreases At-Risk probability")],
              loc="lower right", fontsize=7)
    ax.text(0.0, -0.2, f"Local surrogate R² = {exp.score:.2f}; model p = {prob[i]:.2f}; "
            f"true label = {'At-Risk' if yte[i] else 'Not At-Risk'}", transform=ax.transAxes, fontsize=7,
            color=P.MUTED)
    P.save(fig, name)


lime_plot(hi_i, "LIME explanation — representative high-risk student", "Fig04_lime_high_risk")
lime_plot(lo_i, "LIME explanation — representative low-risk student", "Fig05_lime_low_risk")

# ---------------- Fig 8 PDP / ICE ------------------------------------------ #
f = "StudyHoursPerWeek"
grid, pdp, ice = xai.pdp_ice(pipe, Xte, f, grid=np.linspace(1, 24, 47))
fig, ax = plt.subplots(figsize=(5.4, 3.3))
sel = np.random.default_rng(1).choice(len(Xte), 80, replace=False)
for i in sel:
    ax.plot(grid, ice[i], color=P.OKABE_ITO[0], alpha=0.12, lw=0.6)
ax.plot(grid, pdp, color=P.RISK, lw=2.2, label="Partial dependence (mean)")
ax.plot([], [], color=P.OKABE_ITO[0], alpha=0.5, lw=0.8, label="ICE curves (80 students)")
ax.axhline(0.5, color=P.MUTED, ls=":", lw=0.8)
for hv, (tx, ty) in zip([5, 10, 15], [(6.5, 0.42), (11.5, 0.27), (16.5, 0.15)]):
    vv = np.interp(hv, grid, pdp)
    ax.plot(hv, vv, "o", color=P.INK, ms=4, zorder=5)
    ax.annotate(f"{vv:.1%} at {hv} h/week", (hv, vv), xytext=(tx, ty), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.7))
ax.plot(Xte[f], np.full(len(Xte), -0.03), "|", color=P.MUTED, ms=5, alpha=0.4)
ax.set(xlabel="Study hours per week", ylabel="Predicted P(At-Risk)", ylim=(-0.05, 1.02))
ax.legend(loc="upper right", fontsize=7.5)
ax.set_title(f"PDP and ICE for study hours — {FM} (n = {len(Xte)})", loc="left")
P.save(fig, "Fig08_pdp_studyhours")
pd.DataFrame({"StudyHoursPerWeek": grid, "PDP": pdp}).to_csv(C.TAB_DIR / "_pdp_studyhours.csv", index=False)

# PDP grid (supplementary)
feats = list(C.ACTIONABLE)
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey=True)
for ax, f in zip(axes.ravel(), feats):
    if f == "HighAttendance":
        g = np.array([0.0, 1.0])
    elif f in ("Motivation(1-5)", "Stress(1-5)"):
        g = np.arange(1, 6, dtype=float)
    else:
        g = None
    gg, pd_, ice_ = xai.pdp_ice(pipe, Xte, f, grid=g)
    lo_, hi_ = np.percentile(ice_, [10, 90], axis=0)
    ax.fill_between(gg, lo_, hi_, color=P.OKABE_ITO[0], alpha=0.15, lw=0, label="ICE 10–90th pct.")
    ax.plot(gg, pd_, color=P.RISK, lw=2, marker="o" if len(gg) <= 5 else None, ms=3.5, label="PDP")
    ax.set_title(C.LABELS[f], loc="left", fontsize=9)
    if f == "HighAttendance":
        ax.set_xticks([0, 1], ["No", "Yes"])
axes[0, 0].set_ylabel("P(At-Risk)")
axes[1, 0].set_ylabel("P(At-Risk)")
axes[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle("Partial dependence of predicted risk on the six actionable features", x=0.01, ha="left",
             fontweight="bold", fontsize=10)
fig.tight_layout()
P.save(fig, "FigS7_pdp_grid")

# ---------------- Global importance: SHAP vs PFI vs coefficients ----------- #
with Timer(log, "permutation importance"):
    pfi = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=50 if not C.FAST else 10,
                                 random_state=C.SEED, n_jobs=C.N_JOBS)
pfi_s = pd.Series(pfi.importances_mean, index=X.columns)
pfi_sd = pd.Series(pfi.importances_std, index=X.columns)
# standardised coefficients (one-hot features: max |coef| of their levels)
prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
groups = D.transformed_groups(prep)
Xt = prep.transform(Xtr)
coef = clf.coef_.ravel() * Xt.std(axis=0)
coef_s = pd.Series({c: np.abs(coef[groups[c]]).max() for c in X.columns})
sd_t = Xt.std(axis=0)
odds = pd.Series({c: (np.exp(clf.coef_.ravel()[groups[c]]).round(3).tolist()        # one-hot: per level
                      if len(groups[c]) > 1 else round(float(np.exp(clf.coef_.ravel()[groups[c][0]] * sd_t[groups[c][0]])), 3))
                  for c in X.columns})
imp = pd.DataFrame({"Feature": LBL(gimp.index), "Mean |SHAP|": gimp.values.round(4),
                    "SHAP rank": range(1, 14),
                    "PFI (ΔROC-AUC)": pfi_s[gimp.index].values.round(4),
                    "PFI SD": pfi_sd[gimp.index].values.round(4),
                    "PFI rank": pfi_s.rank(ascending=False)[gimp.index].astype(int).values,
                    "|Std. coef.|": coef_s[gimp.index].values.round(4),
                    "Coef rank": coef_s.rank(ascending=False)[gimp.index].astype(int).values,
                    "Odds ratio (+1 SD; per level for nominal)": odds[gimp.index].astype(str).values})
rho_pfi = stats.spearmanr(gimp.values, pfi_s[gimp.index].values)[0]
rho_coef = stats.spearmanr(gimp.values, coef_s[gimp.index].values)[0]
save_table(imp, "S06_global_importance",
           "Global feature importance of the final model: mean |SHAP|, permutation importance and standardised coefficients.",
           note=f"Spearman ρ(SHAP, PFI) = {rho_pfi:.3f}; ρ(SHAP, |std. coef|) = {rho_coef:.3f}. PFI: mean drop in "
                f"test ROC-AUC over permutations. Odds ratios: per +1 SD increase (binary/ordinal "
                f"features: per SD of their code); nominal features: one OR per one-hot level vs. the intercept.")
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6), sharey=True)
ordr = gimp.index[::-1]
for ax, (vals, err, lab) in zip(axes, [(gimp[ordr], None, "Mean |SHAP|"),
                                       (pfi_s[ordr], pfi_sd[ordr], "PFI (Δ ROC-AUC)"),
                                       (coef_s[ordr], None, "|Standardised coefficient|")]):
    ax.barh(range(13), vals, xerr=err, color=P.OKABE_ITO[0], height=0.62,
            error_kw=dict(lw=0.7, ecolor=P.INK))
    ax.set_title(lab, loc="left", fontsize=9.5)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(13), LBL(ordr))
fig.suptitle(f"Agreement of global importance measures (ρ SHAP–PFI = {rho_pfi:.2f}; ρ SHAP–coef = {rho_coef:.2f})",
             x=0.01, ha="left", fontweight="bold", fontsize=10)
fig.tight_layout()
P.save(fig, "FigS9_importance_comparison")

# ---------------- SHAP stability across 10 seeds --------------------------- #
params = {k: pipe.get_params()[k] for k in M.param_grids()[FM]}
imps = {}
with Timer(log, "SHAP stability (10 seeds)"):
    for sd in C.SHAP_SEEDS:
        a_tr, a_te, b_tr, _ = D.holdout_split(X, y, seed=sd)
        pp = M.make_pipeline(FM, X.columns, C.PRIMARY_STRATEGY, seed=sd, params=params).fit(a_tr, b_tr)
        ss, _, _ = xai.compute_shap(pp, a_tr, a_te, FM, n_bg=len(a_tr), seed=sd)
        imps[sd] = ss.abs().mean()
IM = pd.DataFrame(imps)
rho = IM.corr(method="spearman")
pairs = [rho.loc[a, b] for a, b in itertools.combinations(C.SHAP_SEEDS, 2)]
top5 = [set(IM[s].nlargest(5).index) for s in C.SHAP_SEEDS]
top5_j = np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(top5, 2)])
t10 = pd.DataFrame({"Statistic": ["Mean Spearman ρ", "SD", "Minimum ρ", "Maximum ρ", "Pairwise comparisons",
                                  "Mean top-5 Jaccard overlap", "Top-1 feature identical across seeds"],
                    "Value": [round(np.mean(pairs), 4), round(np.std(pairs, ddof=1), 4), round(min(pairs), 4),
                              round(max(pairs), 4), len(pairs), round(top5_j, 4),
                              f"{(IM.idxmax() == IM.idxmax().mode()[0]).sum()}/10 ({IM.idxmax().mode()[0]})"]})
save_table(t10, "T10_shap_stability",
           "Rank stability of global SHAP importance across 10 random seeds (re-split + re-fit).")
save_table(rho.round(3).rename_axis("Seed").reset_index(), "A04_shap_stability_matrix",
           "Pairwise Spearman correlations of global SHAP rankings across 10 seeds.")
fig, ax = plt.subplots(figsize=(4.6, 3.9))
im = ax.imshow(rho.values, cmap=P.SEQ, vmin=min(0.7, rho.values.min()), vmax=1)
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                color="white" if rho.values[i, j] > 0.93 else P.INK)
ax.set_xticks(range(10), C.SHAP_SEEDS, rotation=45, fontsize=7)
ax.set_yticks(range(10), C.SHAP_SEEDS, fontsize=7)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("Spearman ρ")
ax.set_title(f"SHAP rank stability (mean ρ = {np.mean(pairs):.3f})", loc="left")
P.save(fig, "FigS8_shap_stability")

# ---------------- SHAP vs LIME agreement ----------------------------------- #
rng = np.random.default_rng(C.SEED)
# stratified: every true At-Risk test student + random Not At-Risk students up to 50
pos_i = np.where(yte == 1)[0]
if len(pos_i) > C.N_LIME_INSTANCES // 2:          # larger datasets (e.g. synthetic replica)
    pos_i = np.sort(rng.choice(pos_i, C.N_LIME_INSTANCES // 2, replace=False))
neg_i = rng.choice(np.where(yte == 0)[0], C.N_LIME_INSTANCES - len(pos_i), replace=False)
idx50 = np.concatenate([pos_i, neg_i])
rows = []
with Timer(log, "LIME x 50"):
    for k, i in enumerate(idx50):
        w, _ = xai.lime_explain(lime_ex, pipe, Xte.values[i], X.columns, num_samples=3000 if C.FAST else 5000)
        a_, b_ = S.iloc[i].abs(), w.abs()
        tau, p = stats.kendalltau(a_.values, b_.values)
        t3 = len(set(a_.nlargest(3).index) & set(b_.nlargest(3).index)) / 3
        t5 = len(set(a_.nlargest(5).index) & set(b_.nlargest(5).index)) / 5
        rows.append({"Instance": k + 1, "Test row": int(Xte.index[i]),
                     "True class": "At-Risk" if yte[i] else "Not At-Risk",
                     "Top SHAP feature": C.LABELS[a_.idxmax()], "Top LIME feature": C.LABELS[b_.idxmax()],
                     "Kendall τ": round(tau, 4), "p": round(p, 4), "Top-1 agree": "✓" if a_.idxmax() == b_.idxmax() else "✗",
                     "Top-3 overlap": round(t3, 3), "Top-5 overlap": round(t5, 3),
                     "Sign agreement (top-5)": round(np.mean(np.sign(S.iloc[i][a_.nlargest(5).index]) ==
                                                             np.sign(w[a_.nlargest(5).index])), 3)})
a5 = pd.DataFrame(rows)
save_table(a5, "A05_shap_lime_instances", "Instance-level SHAP–LIME agreement (50 test students: all true At-Risk + random Not At-Risk).",
           note="τ computed over all 13 features on absolute attributions. Sign agreement: share of SHAP top-5 features "
                "where LIME weight has the same sign (LIME weights are on the probability scale, SHAP on log-odds).")
t11 = pd.DataFrame({"Agreement measure": [
    "Mean Kendall's τ (± SD)", "Median Kendall's τ", "Top-1 feature agreement", "Mean top-3 overlap",
    "Mean top-5 overlap", "Instances with significant τ (p < .05)", "Mean sign agreement (top-5)",
    "τ — At-Risk students", "τ — Not At-Risk students"],
    "Result": [f"{a5['Kendall τ'].mean():.4f} ± {a5['Kendall τ'].std():.4f}", f"{a5['Kendall τ'].median():.4f}",
               f"{(a5['Top-1 agree'] == '✓').mean():.1%}", f"{a5['Top-3 overlap'].mean():.1%}",
               f"{a5['Top-5 overlap'].mean():.1%}", f"{(a5.p < .05).mean():.1%}",
               f"{a5['Sign agreement (top-5)'].mean():.1%}",
               f"{a5.loc[a5['True class'] == 'At-Risk', 'Kendall τ'].mean():.4f} (n = {(a5['True class'] == 'At-Risk').sum()})",
               f"{a5.loc[a5['True class'] == 'Not At-Risk', 'Kendall τ'].mean():.4f} (n = {(a5['True class'] == 'Not At-Risk').sum()})"]})
save_table(t11, "T11_shap_lime_agreement", "SHAP–LIME rank agreement (50 test students, stratified).")
log.info("done")
