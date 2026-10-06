"""
17 - Consolidated, publication-grade figures for the manuscript
===============================================================
Rebuilds the main-text figures as dense multi-panel figures from cached
results (no model is re-tuned). Every number drawn here comes from the
pipeline outputs (data/processed, outputs/models, outputs/tables).

  PF02_data_overview          outcome construction + standardised class differences
  PF03_discrimination         OOF ROC / PR (iso-F1, operating points)
  PF04_calibration_utility    reliability, decision curves, teacher workload
  PF05_shap_global            beeswarm annotated with dimensions and mean |SHAP|
  PF06_explanation_compare    SHAP waterfall vs PREX-Edu profile for one student
  PF07_faithfulness           global ablation, instance deletion, dimension ablation
  PF08_operator_action        operator agreement + intervention effect by rank
  PF09_fairness_gaps          subgroup gap forest with CIs
  PF10_synthetic              fidelity, privacy, utility, augmentation
  PF11_threshold_sensitivity  discrimination, PR lift, dominant dimension vs tau
"""
from _common import C, get_logger

import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from prexedu import data as D
from prexedu import plotting as P
from prexedu import prex, xai
from prexedu import evaluation as E

log = get_logger("17_paper_figures")
T = C.TAB_DIR
LAB = C.LABELS
SHORT = C.MODEL_SHORT
DIMS = list(C.PREX_DIMENSIONS)
FEAT_DIM = {f: k for k, v in C.PREX_DIMENSIONS.items() for f in v}
DCOL = P.DIM_COLORS
CTX = "#9ca3af"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "legend.fontsize": 7.6})


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=9, fontweight="bold")


def fcol(f):
    return DCOL.get(FEAT_DIM.get(f), CTX)


# ------------------------------------------------------------------ data ---
df_raw, X, y = D.load_xy()
split = joblib.load(C.MODEL_DIR / "holdout_split.joblib")
Xtr, Xte = X.loc[split["train_idx"]], X.loc[split["test_idx"]]
ytr, yte = y[split["train_idx"]], y[split["test_idx"]]
pipe = joblib.load(C.MODEL_DIR / "holdout_LR.joblib")
S = pd.read_csv(C.MODEL_DIR / "shap_test_LR.csv", index_col=0)
R = pd.read_csv(C.MODEL_DIR / "prex_scores_test.csv", index_col=0)
pred = pd.read_csv(C.MODEL_DIR / "test_predictions_final.csv").set_index("idx")
base = float(pd.read_csv(C.MODEL_DIR / "shap_base.csv", index_col=0).iloc[0, 0])
THR = xai.decision_threshold()
oof = pd.read_csv(C.DATA_PROCESSED / "oof_predictions.csv", index_col=0)
yo = oof["y"].values

# ================================================================ PF02 =====
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
v = df_raw[C.TARGET_SOURCE]
bins = np.arange(30, 101, 2.5)
a1.hist(v[v > C.RISK_THRESHOLD], bins=bins, color="#c9ccd1", ec="white", lw=0.4, label=f"Not At-Risk (n = {(v > 50).sum():,})")
a1.hist(v[v <= C.RISK_THRESHOLD], bins=bins, color=P.RISK, ec="white", lw=0.4, label=f"At-Risk (n = {(v <= 50).sum()}; 5.5%)")
a1.axvline(50, color="black", ls="--", lw=0.9)
a1.text(49.2, a1.get_ylim()[1] * 0.55, r"$\tau = 50$", fontsize=8, ha="right")
a1.text(v.median() + 1, a1.get_ylim()[1] * 0.75, f"median = {v.median():.1f}", fontsize=7.4, color=P.MUTED)
a1.set(xlabel="Last-term percentage", ylabel="Students")
a1.legend(loc="upper left", handlelength=1.2)
title(a1, "(a) Outcome construction")

s2 = pd.read_csv(T / "S02_descriptives_by_class.csv")
s2 = s2[s2["Level"] == "mean ± SD"].copy()
s2["r"] = s2["Effect size"].str.extract(r"=\s*(-?[\d.]+)").astype(float)
fmap = {"Study hours / week": "StudyHoursPerWeek", "Motivation (1–5)": "Motivation(1-5)", "Stress (1–5)": "Stress(1-5)",
        "Sleep hours / night": "SleepHoursPerNight", "Screen time / day": "ScreenTimeDaily", "Age": "Age"}
s2["feat"] = s2["Variable"].map(fmap)
s2 = s2.reindex(s2["r"].abs().sort_values().index)
yy = np.arange(len(s2))
a2.barh(yy, s2["r"], color=[fcol(f) for f in s2["feat"]], height=0.62)
for i, (r, p) in enumerate(zip(s2["r"], s2["p"])):
    stars = "***" if "<" in str(p) else ("**" if float(p) < .01 else "*")
    if r > 0:
        a2.text(r + 0.02, i, f"{r:+.2f}{stars}", va="center", ha="left", fontsize=7.4)
    else:
        a2.text(r + 0.02, i, f"{r:+.2f}{stars}", va="center", ha="left", fontsize=7.4, color="white", fontweight="bold")
a2.set_yticks(yy, [LAB.get(f, f) for f in s2["feat"]])
a2.axvline(0, color="black", lw=0.7)
a2.set_xlim(-0.85, 0.65)
a2.set_xlabel("Rank-biserial $r$ (At-Risk vs Not At-Risk)")
a2.grid(axis="y", visible=False)
a2.legend(handles=[Patch(color=DCOL[k], label=k) for k in ["Engagement", "Lifestyle", "Cognitive Load"]] +
          [Patch(color=CTX, label="Context")], loc="lower left", fontsize=7, handlelength=1)
title(a2, "(b) Class differences")
fig.tight_layout(w_pad=1.5)
P.save(fig, "PF02_data_overview")
log.info("PF02 done")

# ================================================================ PF03 =====
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.35))
p01 = pd.read_csv(T / "P01_nested_cv_performance.csv").set_index("Model")
thr_med = p01["Tuned thr. (median)"].astype(float)
for m in C.MODEL_ORDER[::-1]:
    p = oof[m].values
    lw = 2.2 if m == C.FINAL_MODEL else 1.1
    fpr, tpr, _ = roc_curve(yo, p)
    a1.plot(fpr, tpr, color=P.MODEL_COLORS[m], ls=P.MODEL_LS[m], lw=lw, zorder=3 if m == C.FINAL_MODEL else 2)
    pr, rc, _ = precision_recall_curve(yo, p)
    a2.plot(rc, pr, color=P.MODEL_COLORS[m], ls=P.MODEL_LS[m], lw=lw, zorder=3 if m == C.FINAL_MODEL else 2)
# operating point of LR at its tuned threshold
pl = oof[C.FINAL_MODEL].values
t = thr_med[C.FINAL_MODEL]
tp = ((pl >= t) & (yo == 1)).sum(); fp = ((pl >= t) & (yo == 0)).sum(); fn = ((pl < t) & (yo == 1)).sum()
rec, prec, fpr0 = tp / (tp + fn), tp / (tp + fp), fp / (yo == 0).sum()
a1.plot(fpr0, rec, "o", ms=6, mfc="white", mec="black", zorder=5)
a1.annotate(f"LR at t = {t:.2f}\nrecall {rec:.2f}\nFPR {fpr0:.3f}", (fpr0, rec), xytext=(0.12, 0.22),
            fontsize=7.2, arrowprops=dict(arrowstyle="-", lw=0.6))
a2.plot(rec, prec, "o", ms=6, mfc="white", mec="black", zorder=5)
a2.annotate(f"LR at t = {t:.2f}\nprecision {prec:.2f}\nrecall {rec:.2f}", (rec, prec), xytext=(0.72, 0.84),
            fontsize=7.2, arrowprops=dict(arrowstyle="-", lw=0.6))
a1.plot([0, 1], [0, 1], color=P.MUTED, lw=0.7, ls=":")
a1.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", xlim=(-0.01, 1), ylim=(0, 1.01))
title(a1, "(a) ROC, out-of-fold")
# inset zoom on the low-FPR region
ax_in = a1.inset_axes([0.47, 0.08, 0.5, 0.42])
for m in C.MODEL_ORDER[::-1]:
    fpr, tpr, _ = roc_curve(yo, oof[m].values)
    ax_in.plot(fpr, tpr, color=P.MODEL_COLORS[m], ls=P.MODEL_LS[m], lw=1.6 if m == C.FINAL_MODEL else 0.9)
ax_in.set(xlim=(0, 0.2), ylim=(0.4, 1.0))
ax_in.tick_params(labelsize=6.2)
ax_in.set_title("FPR ≤ 0.20", fontsize=6.8, pad=2)
# iso-F1 curves
for f1 in [0.2, 0.4, 0.6, 0.8]:
    r_ = np.linspace(f1 / (2 - f1) + 1e-3, 1, 200)
    p_ = f1 * r_ / (2 * r_ - f1)
    a2.plot(r_, p_, color="#d1d5db", lw=0.6, zorder=1)
    a2.text(f1 / (2 - f1) + 0.01, 1.005, f"F1={f1}", fontsize=6.2, color=P.MUTED, ha="left", va="bottom")
prev = yo.mean()
a2.axhline(prev, color=P.MUTED, lw=0.8, ls=":")
a2.text(0.02, prev + 0.015, f"prevalence {prev:.3f}", fontsize=6.8, color=P.MUTED)
a2.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.0), ylim=(0, 1.02))
title(a2, "(b) Precision–recall, out-of-fold")
handles = []
for m in C.MODEL_ORDER:
    p = oof[m].values
    handles.append(Line2D([], [], color=P.MODEL_COLORS[m], ls=P.MODEL_LS[m], lw=2.2 if m == C.FINAL_MODEL else 1.2,
                          label=f"{SHORT[m]}  ROC {roc_auc_score(yo, p):.3f} · AP {average_precision_score(yo, p):.3f}"))
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), columnspacing=1.4)
fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2)
P.save(fig, "PF03_discrimination")
log.info("PF03 done")

# ================================================================ PF04 =====
MODELS3 = ["Logistic Regression", "SVM", "XGBoost"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75))
a = axs[0]
for m in MODELS3:
    p = oof[m].values
    q = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": yo, "q": q}).groupby("q", observed=True)
    mp, ob, n = g.p.mean().values, g.y.mean().values, g.size().values
    se = np.sqrt(np.clip(ob * (1 - ob), 1e-9, None) / n)
    a.errorbar(mp, ob, yerr=1.96 * se, fmt=P.MODEL_MARK[m] + "-", color=P.MODEL_COLORS[m], ls=P.MODEL_LS[m],
               ms=3.5, lw=1.2 if m != C.FINAL_MODEL else 1.8, capsize=0, elinewidth=0.6, label=SHORT[m])
a.plot([0, 1], [0, 1], color=P.MUTED, lw=0.7, ls=":")
a.set(xlim=(0, 0.75), ylim=(0, 0.75), xlabel="Mean predicted risk (decile)", ylabel="Observed At-Risk rate")
s20 = pd.read_csv(T / "S20_calibration.csv").set_index("Model")
lr = s20.loc[C.FINAL_MODEL]
a.text(0.03, 0.70, f"LR: slope {lr['Calibration slope (ideal 1)']:.2f}, intercept {lr['Calibration intercept (ideal 0)']:+.2f}\n"
       f"ECE {lr['ECE (10 quantile bins)']:.3f}, Brier {lr['Brier']:.3f}", fontsize=6.6, va="top")
a.legend(loc="lower right")
title(a, "(a) Calibration")

a = axs[1]
ths = np.linspace(0.01, 0.5, 99)
nb_all = prev - (1 - prev) * ths / (1 - ths)
a.plot(ths, nb_all, color="black", lw=0.9, ls="--", label="Flag all")
a.axhline(0, color="black", lw=0.9, label="Flag none")
for m in MODELS3:
    nb = E.net_benefit(yo, oof[m].values, ths)
    a.plot(ths, nb, color=P.MODEL_COLORS[m], ls=P.MODEL_LS[m], lw=1.8 if m == C.FINAL_MODEL else 1.1, label=SHORT[m])
a.set(ylim=(-0.01, 0.058), xlim=(0, 0.5), xlabel="Threshold probability $p_t$", ylabel="Net benefit")
a.legend(loc="upper right", ncol=1)
title(a, "(b) Decision curves")

a = axs[2]
N = len(yo)
for m in MODELS3:
    p = oof[m].values
    order = np.argsort(-p)
    cum_tp = np.cumsum(yo[order])
    recall = cum_tp / yo.sum()
    flagged = np.arange(1, N + 1)
    a.plot(recall, flagged / N * 100, color=P.MODEL_COLORS[m], ls=P.MODEL_LS[m], lw=1.8 if m == C.FINAL_MODEL else 1.1,
           label=SHORT[m])
s21 = pd.read_csv(T / "S21_operating_points.csv")
s21 = s21[s21["Model"] == C.FINAL_MODEL]
offs = {"≥ 95%": (-82, 10), "≥ 90%": (-86, -4), "≥ 80%": (6, -14)}
for _, r in s21.iterrows():
    tr = r["Target recall"].strip()
    if tr in offs:
        a.plot(r["Recall"], r["Students flagged"] / N * 100, "o", mfc="white", mec="black", ms=4.5, zorder=5)
        a.annotate(f"{tr[2:]} recall: {int(r['Students flagged'])} flagged", (r["Recall"], r["Students flagged"] / N * 100),
                   xytext=offs[tr], textcoords="offset points", fontsize=6.4,
                   arrowprops=dict(arrowstyle="-", lw=0.5, color=P.MUTED))
a.axhline(prev * 100, color=P.MUTED, lw=0.7, ls=":")
a.text(0.02, prev * 100 + 0.8, "ideal (5.5%)", fontsize=6.4, color=P.MUTED)
a.set(xlim=(0, 1.0), ylim=(0, 40), xlabel="Recall (share of At-Risk found)", ylabel="Students flagged (%)")
a.legend(loc="upper left")
title(a, "(c) Teacher workload")
fig.tight_layout(w_pad=1.2)
P.save(fig, "PF04_calibration_utility")
log.info("PF04 done")

# ================================================================ PF05 =====
gimp = S.abs().mean().sort_values()
fig, ax = plt.subplots(figsize=(6.6, 3.7))
rng = np.random.default_rng(0)
for i, f in enumerate(gimp.index):
    sv = S[f].values
    xv = Xte[f].astype(float).values
    rk = stats.rankdata(xv) / len(xv)
    # density-based jitter
    hist, edges = np.histogram(sv, bins=40)
    idx = np.clip(np.digitize(sv, edges) - 1, 0, len(hist) - 1)
    dens = hist[idx] / hist.max()
    jit = rng.uniform(-1, 1, len(sv)) * 0.36 * np.sqrt(dens)
    sc = ax.scatter(sv, i + jit, c=rk, cmap=P.DIVERGING, vmin=0, vmax=1, s=5, lw=0, alpha=0.85, rasterized=True)
    ax.text(ax.get_xlim()[1] if False else 6.7, i, f"{gimp[f]:.2f}", va="center", fontsize=7.2, color=P.INK)
ax.set_yticks(range(len(gimp)), [LAB[f] for f in gimp.index])
for tl, f in zip(ax.get_yticklabels(), gimp.index):
    tl.set_color(P.INK)
    tl.set_bbox(dict(fc=fcol(f), ec="none", alpha=0.22, boxstyle="round,pad=0.15"))
ax.axvline(0, color=P.MUTED, lw=0.7)
ax.set_xlim(-4.6, 6.4)
ax.text(6.7, len(gimp) - 0.35, "mean |φ|", fontsize=7.2, fontweight="bold", ha="left")
ax.set_xlabel("SHAP value (log-odds); right = towards At-Risk")
ax.grid(axis="y", visible=False)
cb = fig.colorbar(sc, ax=ax, pad=0.13, fraction=0.03, aspect=30)
cb.set_ticks([0.02, 0.98], labels=["low", "high"])
cb.set_label("Feature value (rank)", fontsize=7.5)
cb.outline.set_visible(False)
ax.legend(handles=[Patch(color=DCOL[k], alpha=0.35, label=k) for k in DIMS] + [Patch(color=CTX, alpha=0.35, label="Context")],
          loc="lower right", fontsize=7, title="PREX-Edu dimension", title_fontsize=7.2)
fig.tight_layout()
P.save(fig, "PF05_shap_global")
log.info("PF05 done")

# ================================================================ PF06 =====
sid = 97   # Student ID 98 in Table A08 (1-based)
row, xr = S.loc[sid], Xte.loc[sid]
prob_i = float(pred.loc[sid, "prob"])
Rr = R.loc[sid]
Ssign = pd.Series({k: row[v].sum() for k, v in C.PREX_DIMENSIONS.items()})
fig = plt.figure(figsize=(7.2, 3.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 0.42], hspace=0.55, wspace=0.55)
a1 = fig.add_subplot(gs[:, 0])
order = row.abs().sort_values(ascending=False).index
top = list(order[:9])
rest = row[order[9:]].sum()
vals = [row[f] for f in top] + [rest]
labs = [f"{LAB[f]} = {D.decode_value(f, xr[f])}" for f in top] + [f"{len(order) - 9} other features"]
cols = [fcol(f) for f in top] + [CTX]
cum = base
for i, (vv, lb, cc) in enumerate(zip(vals, labs, cols)):
    yy_ = len(vals) - 1 - i
    a1.barh(yy_, vv, left=cum, color=P.RISK if vv > 0 else P.PROTECT, height=0.62)
    a1.text(cum + vv + (0.08 if vv > 0 else -0.08), yy_, f"{vv:+.2f}", va="center", ha="left" if vv > 0 else "right", fontsize=7)
    cum += vv
a1.set_yticks(range(len(vals)), labs[::-1], fontsize=7.4)
for tl, cc in zip(a1.get_yticklabels(), cols[::-1]):
    tl.set_bbox(dict(fc=cc, ec="none", alpha=0.22, boxstyle="round,pad=0.12"))
a1.axvline(base, color=P.MUTED, ls=":", lw=0.8)
a1.axvline(cum, color="black", ls="--", lw=0.8)
a1.text(base, len(vals) - 0.35, f" E[f(x)] = {base:.2f}", fontsize=6.8, color=P.MUTED)
a1.text(cum, -0.85, f"f(x) = {cum:.2f}  (p = {prob_i:.2f}) ", fontsize=6.8, ha="right")
a1.set_xlabel("Log-odds of At-Risk")
a1.set_ylim(-1.1, len(vals) - 0.1)
a1.grid(axis="y", visible=False)
title(a1, "(a) Feature-level SHAP (13 attributions)")

a2 = fig.add_subplot(gs[0, 1])
od = Rr.sort_values().index
a2.barh(range(4), Rr[od], color=[DCOL[k] for k in od], height=0.6)
for i, k in enumerate(od):
    arrow = "▲ risk" if Ssign[k] > 0 else "▼ protective"
    a2.text(Rr[k] + 0.1, i, f"{Rr[k]:.2f}  {arrow}", va="center", fontsize=7.2)
a2.set_yticks(range(4), list(od))
a2.set_xlim(0, Rr.max() * 1.75)
a2.set_xlabel(r"$R_k$ = sum of $|\phi_j|$ in dimension (log-odds)")
a2.grid(axis="y", visible=False)
cov = float(prex.coverage(S.loc[[sid]]).iloc[0])
a2.text(0.98, 0.04, f"coverage {cov:.0%}", transform=a2.transAxes, ha="right", fontsize=7, color=P.MUTED)
title(a2, "(b) PREX-Edu profile (4 dimensions)")

a3 = fig.add_subplot(gs[1, 1]); a3.axis("off")
txt = (f"Risk {prob_i:.0%}. Primary concern: Engagement (motivation 1/5,\n"
       f"{xr['StudyHoursPerWeek']:.1f} study h/week) → study plan and mentoring.\n"
       "Secondary: Cognitive Load (stress 5/5) → counsellor referral.")
a3.text(0, 1.0, "(c) Educator-facing summary", fontsize=9, fontweight="bold", va="top")
a3.text(0, 0.62, txt, fontsize=7.3, va="top", linespacing=1.35,
        bbox=dict(fc="#f5f5f5", ec="#bdbdbd", boxstyle="round,pad=0.4"))
P.save(fig, "PF06_explanation_compare")
log.info("PF06 done")

# ================================================================ PF07 =====
ref = xai.reference_values(Xtr)
order_prex = prex.feature_order_from_prex(R.mean(), S.abs().mean())
order_shap = S.abs().mean().sort_values(ascending=False).index.tolist()
ks = list(range(0, 14))


def auc_after(order, k):
    if k >= len(order):
        return 0.5
    return roc_auc_score(yte, xai.masked_predict(pipe, Xte, order[:k], ref))


cur_p = np.array([auc_after(order_prex, k) for k in ks])
cur_s = np.array([auc_after(order_shap, k) for k in ks])
rng = np.random.default_rng(C.SEED)
rand = np.array([[auc_after(list(rng.permutation(X.columns)), k) for k in ks] for _ in range(C.N_RANDOM_ABLATION)])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw=dict(width_ratios=[1.15, 1, 0.95]))
a = axs[0]
lo, hi = np.percentile(rand, [2.5, 97.5], axis=0)
a.fill_between(ks, lo, hi, color="#d1d5db", alpha=0.7, lw=0, label="Random (95% range)")
a.plot(ks, rand.mean(0), color=P.MUTED, ls="--", lw=1.1, label="Random (mean)")
a.plot(ks, cur_s, color="#000000", ls=":", lw=1.3, label="SHAP order")
a.plot(ks, cur_p, color=DCOL["Engagement"], lw=2, marker="o", ms=3, label="PREX-Edu order")
a.axhline(0.5, color=P.MUTED, lw=0.6)
for k, lab_, off in [(1, "− motivation", (5, 2)), (2, "− study hours", (-6, -11)), (5, "− stress", (5, 3))]:
    a.annotate(lab_, (k, cur_p[k]), xytext=off, textcoords="offset points", fontsize=6.4, color=DCOL["Engagement"],
               ha="right" if off[0] < 0 else "left")
a.set(xlabel="Features removed", ylabel="Test ROC-AUC", ylim=(0.2, 1.02), xticks=range(0, 14, 2))
title(a, "(a) Global ablation")

a = axs[1]
flag = pred.index[pred["prob"] >= THR]
curves = {"PREX-Edu": [], "SHAP": [], "Random": []}
for i in flag:
    xi = Xte.loc[[i]]
    o_p = prex.instance_feature_order(S.loc[i], R.loc[i])
    o_s = S.loc[i].abs().sort_values(ascending=False).index.tolist()
    curves["PREX-Edu"].append([xai.masked_predict(pipe, xi, o_p[:k], ref)[0] for k in ks])
    curves["SHAP"].append([xai.masked_predict(pipe, xi, o_s[:k], ref)[0] for k in ks])
    rr = []
    for _ in range(30):
        o_r = list(rng.permutation(X.columns))
        rr.append([xai.masked_predict(pipe, xi, o_r[:k], ref)[0] for k in ks])
    curves["Random"].append(np.mean(rr, 0))
sty = {"PREX-Edu": (DCOL["Engagement"], "-", 2), "SHAP": ("black", ":", 1.3), "Random": (P.MUTED, "--", 1.1)}
for k_, cv in curves.items():
    cv = np.array(cv); m_ = cv.mean(0); se = cv.std(0, ddof=1) / np.sqrt(len(cv))
    c_, ls_, lw_ = sty[k_]
    a.fill_between(ks, m_ - 1.96 * se, m_ + 1.96 * se, color=c_, alpha=0.15, lw=0)
    a.plot(ks, m_, color=c_, ls=ls_, lw=lw_, label=k_)
aopc = pd.read_csv(T / "S07_instance_aopc.csv").set_index("Removal order")["AOPC@6 mean"]
a.text(0.97, 0.95, f"AOPC@6\nPREX-Edu {aopc['PREX-Edu']:.3f}\nSHAP {aopc['SHAP']:.3f}\nRandom {aopc['Random']:.3f}",
       transform=a.transAxes, ha="right", va="top", fontsize=6.6)
a.axhline(THR, color=P.RISK, lw=0.6, ls=":")
a.text(13, THR - 0.012, "flag threshold", fontsize=6.2, color=P.RISK, ha="right", va="top")
a.set(xlabel="Features removed (per student)", ylabel="Mean P(At-Risk)", xticks=range(0, 14, 2), ylim=(0, 0.75))
title(a, f"(b) Instance deletion (n = {len(flag)})")

a = axs[2]
s8 = pd.read_csv(T / "S08_dimension_ablation.csv")
for _, r in s8.iterrows():
    k = r["Dimension"]; c_ = DCOL.get(k, CTX)
    a.scatter(r["Mean R_k"], r["ΔROC-AUC"], s=40, color=c_, ec="black", lw=0.5, zorder=3)
    nm = "Context" if k.startswith("Context") else k
    off = {"Engagement": (-4, 6, "right"), "Cognitive Load": (5, 3, "left"), "Lifestyle": (6, 3, "left"),
           "Participation": (0, 9, "center"), "Context": (8, -3, "left")}[nm]
    a.annotate(nm, (r["Mean R_k"], r["ΔROC-AUC"]), xytext=off[:2], textcoords="offset points", fontsize=6.8, ha=off[2])
rho = stats.spearmanr(s8.loc[~s8.Dimension.str.startswith("Context"), "Mean R_k"], s8.loc[~s8.Dimension.str.startswith("Context"), "ΔROC-AUC"]).correlation
a.text(0.04, 0.95, f"Spearman ρ = {rho:.2f}\n(4 dimensions)", transform=a.transAxes, va="top", fontsize=6.8)
a.axhline(0, color=P.MUTED, lw=0.6)
a.set(xlabel="Mean $R_k$ (log-odds)", ylabel="ΔROC-AUC when removed", xlim=(0, 4.8), ylim=(-0.02, 0.23))
title(a, "(c) Dimension ablation")
fig.legend(handles=[Line2D([], [], color=DCOL["Engagement"], lw=2, marker="o", ms=3, label="PREX-Edu order"),
                    Line2D([], [], color="black", ls=":", lw=1.3, label="SHAP order"),
                    Line2D([], [], color=P.MUTED, ls="--", lw=1.1, label="Random order (mean)"),
                    Patch(color="#d1d5db", label="Random (95% range / CI)")],
           loc="lower center", ncol=4, bbox_to_anchor=(0.38, -0.01), fontsize=6.8)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
P.save(fig, "PF07_faithfulness")
log.info("PF07 done")

# ================================================================ PF08 =====
s24 = pd.read_csv(T / "S24_aggregation_operators.csv")
s12 = pd.read_csv(T / "S12_intervention_per_student.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
a = axs[0]
refcols = ["Top-1 = exact group-Shapley", "Top-1 = dimension ablation", "Top-1 = equal-budget intervention"]
ops = ["sum|φ| (PREX-Edu)", "|Σφ| (group SHAP)", "RMS(φ)", "mean|φ| (cardinality-normalised)"]
oplab = ["Σ|φ| (PREX-Edu)", "|Σφ|", "RMS(φ)", "mean|φ| (earlier)"]
M = np.array([[float(str(s24.set_index("Operator").loc[o, c]).rstrip("%")) for c in refcols] for o in ops])
im = a.imshow(M, cmap=P.SEQ, vmin=40, vmax=100, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        a.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center", fontsize=7.6, color="white" if M[i, j] > 80 else P.INK,
               fontweight="bold" if i == 0 else "normal")
a.set_xticks(range(3), ["Exact\ngroup-Shapley", "Dimension\nablation", "Equal-budget\nintervention"], fontsize=7)
a.set_yticks(range(4), oplab, fontsize=7.3)
a.grid(False)
a.tick_params(length=0)
for sp in a.spines.values(): sp.set_visible(False)
title(a, "(a) Top-1 agreement (n = 21)")


def by_rank(design):
    d = s12[s12["Design"] == design]
    rows = []
    for _, r in d.iterrows():
        for k in DIMS:
            rows.append({"rank": int(r[f"rank {k}"]), "dP": r[f"ΔP {k}"], "dim": k})
    return pd.DataFrame(rows)


s11 = pd.read_csv(T / "S11_intervention_simulation.csv")
for a, design, lbl, share in [(axs[1], "full", "(b) Full dose: rank 1 best 100%", "100%"), (axs[2], "equal", "(c) Equal budget: rank 1 best 81%", "81%")]:
    br = by_rank(design)
    data = [br.loc[br["rank"] == k, "dP"].values for k in range(1, 5)]
    bp = a.boxplot(data, positions=range(1, 5), widths=0.5, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", lw=1.1), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    for b_ in bp["boxes"]:
        b_.set(facecolor="#e5e7eb", edgecolor="black", lw=0.7)
    for k in range(1, 5):
        sub = br[br["rank"] == k]
        a.scatter(k + rng.uniform(-0.15, 0.15, len(sub)), sub["dP"], s=9, c=[DCOL[d_] for d_ in sub["dim"]], lw=0.3, ec="white", zorder=3)
    a.plot(range(1, 5), [np.mean(d_) for d_ in data], "D", color="black", ms=3.5, zorder=4)
    a.set(xlabel="PREX-Edu rank of targeted dimension", xticks=range(1, 5))
    title(a, lbl)
axs[1].set_ylabel("Reduction in P(At-Risk)")
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=DCOL[k], label=f"targeted: {k}", ms=4) for k in DIMS] +
           [Line2D([], [], marker="D", ls="", color="black", label="mean", ms=3.5)],
           loc="lower center", ncol=5, fontsize=6.8, bbox_to_anchor=(0.62, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
P.save(fig, "PF08_operator_action")
log.info("PF08 done")

# ================================================================ PF09 =====
s23 = pd.read_csv(T / "S23_fairness_gaps.csv")
s22 = pd.read_csv(T / "S22_fairness_subgroups.csv")
prevmap = {(r["Attribute"], r["Subgroup"]): (r["n"], r["At-Risk n"], r["Prevalence"]) for _, r in s22.iterrows()}


def parse(s):
    s = str(s); sig = "*" in s
    v_ = [float(x) for x in re.findall(r"[-+]?\d*\.\d+", s)]
    return v_[0], v_[1], v_[2], sig


short_attr = {"Gender": "", "Income range": " income", "Parent education": " parents"}
metrics = [("Δ Selection rate", "Δ Selection rate"), ("Δ TPR (equal opportunity)", "Δ True-positive rate"),
           ("Δ FPR", "Δ False-positive rate")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.4, 1]))
labels = []
for _, r in s23.iterrows():
    a_, b_ = [x.strip() for x in r["Comparison"].split("−")]
    na = prevmap.get((r["Attribute"], a_)); nb = prevmap.get((r["Attribute"], b_))
    a_lab = "Not stated" if a_.startswith("Prefer") else a_
    labels.append(f"{a_lab} vs {b_}{short_attr[r['Attribute']]}\n(prev. {na[2]*100:.1f}% vs {nb[2]*100:.1f}%; "
                  f"{int(na[1])} vs {int(nb[1])} cases)")
yy = np.arange(len(s23))[::-1]
for a, (col, lbl) in zip(axs, metrics):
    for yi, (_, r) in zip(yy, s23.iterrows()):
        v_, lo_, hi_, sig = parse(r[col])
        c_ = {"Gender": "#0072B2", "Income range": "#009E73", "Parent education": "#CC79A7"}[r["Attribute"]]
        a.plot([lo_, hi_], [yi, yi], color=c_, lw=1.2)
        a.plot(v_, yi, "o", ms=5, mfc=c_ if sig else "white", mec=c_, mew=1.2, zorder=3)
    a.axvline(0, color="black", lw=0.7)
    a.set_xlabel(lbl)
    a.grid(axis="y", visible=False)
for yb in [4.5, 1.5]:
    for a in axs:
        a.axhline(yb, color="#d1d5db", lw=0.8)
axs[0].set_yticks(yy, labels, fontsize=6.6)
di = s23["Disparate-impact ratio"].values
for yi, d_ in zip(yy, di):
    axs[2].text(1.02, yi, f"DI {d_:.2f}", transform=axs[2].get_yaxis_transform(), va="center", fontsize=6.8,
                color=P.RISK if (d_ < 0.8 or d_ > 1.25) else P.INK)
fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="black", mec="black", label="95% CI excludes 0"),
                    Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="CI includes 0")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.6, -0.03))
fig.tight_layout(rect=(0, 0.06, 0.95, 1), w_pad=0.8)
P.save(fig, "PF09_fairness_gaps")
log.info("PF09 done")

# ================================================================ PF10 =====
s18 = pd.read_csv(T / "S18_synthetic_fidelity_privacy.csv").set_index("Generator")
aug = pd.read_csv(T / "_augmentation_fold_metrics.csv")
gens = ["Gaussian Copula", "CTGAN", "TVAE"]
gcol = {"Gaussian Copula": "#0072B2", "CTGAN": "#D55E00", "TVAE": "#009E73"}
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9))
axs = axs.ravel()
a = axs[0]
for i, g in enumerate(gens):
    a.bar(i - 0.18, s18.loc[g, "SDMetrics overall"], 0.36, color=gcol[g])
    a.bar(i + 0.18, s18.loc[g, "Discriminator AUC (↓0.5)"], 0.36, color="white", ec=gcol[g], hatch="////", lw=0.8)
    a.text(i - 0.18, s18.loc[g, "SDMetrics overall"] + 0.02, f"{s18.loc[g, 'SDMetrics overall']:.2f}", ha="center", fontsize=6.3)
    a.text(i + 0.18, s18.loc[g, "Discriminator AUC (↓0.5)"] + 0.02, f"{s18.loc[g, 'Discriminator AUC (↓0.5)']:.2f}", ha="center", fontsize=6.3)
a.axhline(0.5, color="black", lw=0.6, ls=":")
a.set_xticks(range(3), ["Gaussian copula", "CTGAN", "TVAE"]); a.set_ylim(0, 1.25)
a.legend(handles=[Patch(color="#555", label="SDMetrics quality (higher = better)"),
                  Patch(fc="white", ec="#555", hatch="////", label="Discriminator AUC (lower = better)"),
                  Line2D([], [], color="black", ls=":", lw=0.8, label="AUC 0.5: indistinguishable from real")],
         fontsize=6.4, loc="upper left", ncol=1)
title(a, "(a) Fidelity")
a = axs[1]
vals = [s18.loc[g, "DCR < 5th pct of real hold-out (%)"] for g in gens]
a.bar(range(3), vals, color=[gcol[g] for g in gens], width=0.6)
for i, v_ in enumerate(vals):
    a.text(i, max(v_, 5) + 0.8, f"{v_:.1f}%", ha="center", fontsize=6.6)
a.axhline(5, color="black", lw=0.8, ls="--"); a.text(1.5, 10, "expected if private: 5%", fontsize=6.2, ha="center")
a.set_xticks(range(3), ["Gaussian copula", "CTGAN", "TVAE"]); a.set_ylabel("Synthetic records closer than\n5th pct. of real hold-out (%)"); a.set_ylim(0, 40)
title(a, "(b) Privacy")
a = axs[2]
vals = [s18.loc[g, "TSTR ROC-AUC (real test)"] for g in gens]
a.bar(range(3), vals, color=[gcol[g] for g in gens], width=0.6)
for i, v_ in enumerate(vals):
    a.text(i, v_ + 0.004, f"{v_:.3f}", ha="center", fontsize=6.6)
a.axhline(0.9669, color="black", lw=0.8, ls="--"); a.text(2.45, 0.972, "train on real: 0.967", fontsize=6.4, ha="right")
a.set_xticks(range(3), ["Gaussian copula", "CTGAN", "TVAE"]); a.set_ylabel("ROC-AUC on real test students"); a.set_ylim(0.78, 1.0)
title(a, "(c) Utility: train synthetic, test real")
a = axs[3]
cfg = [("smotenc", "SMOTE-NC → 20%"), ("copula-min", "Copula min. → 20%"), ("tvae-min", "TVAE min. → 20%"),
       ("copula-x2", "Copula ×2"), ("tvae-x2", "TVAE ×2"), ("copula-x5", "Copula ×5")]
for j, (m, off, mk) in enumerate([("Logistic Regression", 0.14, "o"), ("XGBoost", -0.14, "s")]):
    b = aug[(aug.model == m) & (aug.config == "baseline")].set_index("fold")["pr_auc"]
    for i, (c_, _) in enumerate(cfg):
        d_ = aug[(aug.model == m) & (aug.config == c_)].set_index("fold")["pr_auc"] - b
        mu = d_.mean(); ci = stats.t.ppf(0.975, len(d_) - 1) * d_.std(ddof=1) / np.sqrt(len(d_))
        yi = len(cfg) - 1 - i + off
        a.plot([mu - ci, mu + ci], [yi, yi], color=P.MODEL_COLORS[m], lw=1)
        a.plot(mu, yi, mk, color=P.MODEL_COLORS[m], ms=4, label=SHORT[m] if i == 0 else None)
a.axvline(0, color="black", lw=0.7)
a.set_yticks(range(len(cfg)), [c[1] for c in cfg][::-1], fontsize=6.6)
a.set_xlabel("ΔPR-AUC vs training on real data only (95% CI, 10 folds)")
a.grid(axis="y", visible=False)
a.legend(loc="lower left", fontsize=6.4)
title(a, "(d) Augmentation effect")
fig.tight_layout(w_pad=1.5, h_pad=1.5)
P.save(fig, "PF10_synthetic")
log.info("PF10 done")

# ================================================================ PF11 =====
s26 = pd.read_csv(T / "S26_threshold_sensitivity.csv")
tau = s26["τ (last-term % ≤)"].values


def ms(col):
    m_ = s26[col].str.extract(r"([\d.]+)\s*±\s*([\d.]+)").astype(float)
    return m_[0].values, m_[1].values


fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.95))
a = axs[0]
for m, col, off in [("Logistic Regression", "LR ROC-AUC", -0.4), ("XGBoost", "XGB ROC-AUC", 0.4)]:
    mu, sd = ms(col)
    a.errorbar(tau + off, mu, yerr=sd, color=P.MODEL_COLORS[m], marker=P.MODEL_MARK[m], ms=4, lw=1.4, ls=P.MODEL_LS[m],
               capsize=2, elinewidth=0.7, label=SHORT[m])
a.axvline(50, color=P.MUTED, lw=0.6, ls=":")
a.set_xticks(tau, [f"{t_}\nn={n_}" for t_, n_ in zip(tau, s26["At-Risk n"])])
a.set(xlabel=r"$\tau$ and number At-Risk", ylabel="ROC-AUC (mean ± SD)", ylim=(0.82, 1.04))
a.legend(loc="lower right")
title(a, "(a) Discrimination")
a = axs[1]
for m, col in [("Logistic Regression", "LR PR-AUC lift"), ("XGBoost", "XGB PR-AUC lift")]:
    a.plot(tau, s26[col], color=P.MODEL_COLORS[m], marker=P.MODEL_MARK[m], ms=4, ls=P.MODEL_LS[m], label=SHORT[m])
a.set_yscale("log"); a.set_yticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
a.axhline(1, color=P.MUTED, lw=0.6)
a.axvline(50, color=P.MUTED, lw=0.6, ls=":")
a.set_xticks(tau, [f"{t_}\n{pv}" for t_, pv in zip(tau, s26["Prevalence"])])
a.set(xlabel=r"$\tau$ and prevalence", ylabel="PR-AUC / prevalence (log)", ylim=(1, 130))
a.legend(loc="upper right")
title(a, "(b) Lift over chance")
a = axs[2]
shares = []
for s_ in s26["Dominant dimension (top-risk students)"]:
    d_ = dict((k, 0.0) for k in DIMS)
    for k, v_ in re.findall(r"([A-Za-z ]+?)\s+(\d+)%", s_):
        d_[k.strip()] = float(v_)
    rem = 100 - sum(d_.values())
    d_["Lifestyle"] += max(rem, 0)  # unreported remainder is < 2% and all Lifestyle at tau=60
    shares.append(d_)
sh = pd.DataFrame(shares, index=tau)[DIMS]
left = np.zeros(len(tau))
for k in DIMS:
    a.barh(range(len(tau)), sh[k], left=left, color=DCOL[k], height=0.62, label=k)
    for i, v_ in enumerate(sh[k]):
        if v_ >= 12:
            a.text(left[i] + v_ / 2, i, f"{v_:.0f}", ha="center", va="center", fontsize=6.5, color="white")
    left += sh[k].values
a.set_yticks(range(len(tau)), [f"τ = {t_}" for t_ in tau])
a.set_xlabel("Dominant dimension, top-risk students (%)")
a.set_xlim(0, 100); a.grid(axis="y", visible=False); a.invert_yaxis()
title(a, "(c) Explanation stability")
fig.legend(handles=[Patch(color=DCOL[k], label=k) for k in DIMS], loc="lower center", ncol=4, fontsize=6.8,
           bbox_to_anchor=(0.5, -0.01), handlelength=1)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
P.save(fig, "PF11_threshold_sensitivity")
log.info("PF11 done")
