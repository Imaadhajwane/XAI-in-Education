"""
01 - Six-classifier comparison (first-draft protocol, rebuilt)
=============================================================
Protocol (as described in the manuscript)
  * stratified 70:30 split (845 train / 363 test)
  * imbalance strategy C.PRIMARY_STRATEGY (default: none; see script 11) inside the pipeline
  * GridSearchCV, stratified 10-fold, on the training partition
  * 10-fold CV of the tuned pipelines on the training partition  -> mean ± SD
  * held-out test set (n = 363, 19 At-Risk) -> bootstrap CIs, κ, Brier, ROC,
    McNemar, DeLong

Outputs
  Table 5   class distribution before / after SMOTE-NC        T05_smote_distribution
  Table 6   model comparison (CV mean ± SD + test bootstrap CI) T06_model_comparison
  Table 7   class-wise precision / recall / F1                 T07_classwise
  Table 8   LR vs XGBoost: Welch t + corrected resampled t     T08_lr_vs_xgb
  Table 9   Cohen's κ and Brier score                          T09_kappa_brier
  Table A1  McNemar pairwise                                   A01_mcnemar
  Table A2  Friedman ranks (+ Nemenyi)                         A02_friedman
  Table A3  DeLong pairwise AUC                                A03_delong
  Table S4  selected hyper-parameters                          S04_hyperparameters
  Fig 3     ROC (and PR) curves on the hold-out set            Fig03_roc_pr_holdout
  Fig A1    bootstrap distributions                            FigA1_bootstrap_hist
  Fig A2    KDE of fold accuracies LR vs XGB                   FigA2_kde_lr_xgb
  Fig A3    fold-wise accuracy box-plot                        FigA3_fold_boxplot
  Fig S5    confusion matrices                                 FigS5_confusion_matrices
  Models    outputs/models/holdout_<model>.joblib  (+ split indices, predictions)
"""
from _common import C, Timer, get_logger

import itertools

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from prexedu import data as D
from prexedu import evaluation as E
from prexedu import models as M
from prexedu import plotting as P
from prexedu.cv import cv_predict
from prexedu.tables import mean_sd, save_table

log = get_logger("01_model_comparison")
STRATEGY = C.PRIMARY_STRATEGY

df, X, y = D.load_xy()
Xtr, Xte, ytr, yte, dtr, dte = D.holdout_split(X, y, df)
joblib.dump({"train_idx": Xtr.index.values, "test_idx": Xte.index.values},
            C.MODEL_DIR / "holdout_split.joblib")
log.info(f"train {len(ytr)} ({ytr.sum()} At-Risk) | test {len(yte)} ({yte.sum()} At-Risk)")

# ---------------- Table 5: SMOTE-NC on training partition ------------------ #
sm = M.make_sampler("smotenc", X.columns)
_, yres = sm.fit_resample(Xtr, ytr)
t5 = pd.DataFrame({
    "Class": ["At-Risk (1)", "Not At-Risk (0)", "Total"],
    "Pre-SMOTE count": [int(ytr.sum()), int((1 - ytr).sum()), len(ytr)],
    "Pre-SMOTE (%)": [round(100 * ytr.mean(), 1), round(100 * (1 - ytr.mean()), 1), 100.0],
    "Post-SMOTE count": [int(yres.sum()), int((1 - yres).sum()), len(yres)],
    "Post-SMOTE (%)": [round(100 * yres.mean(), 1), round(100 * (1 - yres.mean()), 1), 100.0],
})
save_table(t5, "T05_smote_distribution",
           "Class distribution of the training partition before and after SMOTE-NC.",
           note=f"Held-out test set (n = {len(yte)}, {yte.sum()} At-Risk) is never resampled. Inside cross-validation "
                f"SMOTE-NC is re-fitted on each training fold only. Primary analysis uses: "
                f"{M.STRATEGY_LABELS[C.PRIMARY_STRATEGY]} (justified in Table S5); SMOTE-NC is a sensitivity analysis.")

# ---------------- Tuning ---------------------------------------------------- #
grids = M.param_grids()
cv10 = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
fitted, hp_rows = {}, []
for name in C.MODEL_ORDER:
    with Timer(log, f"GridSearchCV {name}"):
        gs = GridSearchCV(M.make_pipeline(name, X.columns, STRATEGY), grids[name],
                          scoring=C.TUNING_SCORING, cv=cv10, n_jobs=C.N_JOBS)
        gs.fit(Xtr, ytr)
    fitted[name] = gs.best_estimator_
    joblib.dump(gs.best_estimator_, C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[name]}.joblib")
    hp_rows.append({"Model": name,
                    "Selected hyper-parameters": ", ".join(f"{k.replace('clf__', '')}={v}"
                                                           for k, v in gs.best_params_.items()),
                    f"Inner-CV {C.TUNING_SCORING}": round(gs.best_score_, 4)})
save_table(pd.DataFrame(hp_rows), "S04_hyperparameters",
           "Hyper-parameters selected by stratified 10-fold GridSearchCV on the training partition.",
           note=f"Optimisation criterion: {C.TUNING_SCORING} (PR-AUC), appropriate for a 5.5% minority class.")

# ---------------- 10-fold CV on training partition ------------------------- #
fold_res, test_prob, thr = {}, {}, {}
from prexedu.cv import _best_f1_threshold
for name in C.MODEL_ORDER:
    fold_res[name], oof_tr = cv_predict(fitted[name], Xtr, ytr)
    thr[name] = _best_f1_threshold(ytr, oof_tr)          # chosen on TRAINING out-of-fold predictions only
    test_prob[name] = fitted[name].predict_proba(Xte)[:, 1]
import json
(C.MODEL_DIR / "decision_thresholds.json").write_text(json.dumps(thr, indent=2))
log.info("F1-optimal thresholds (training OOF): " + ", ".join(f"{C.MODEL_SHORT[k]}={v:.3f}" for k, v in thr.items()))
pd.concat([d.assign(model=n) for n, d in fold_res.items()]).to_csv(
    C.TAB_DIR / "_cv10_train_fold_metrics.csv", index=False)
pd.DataFrame(test_prob).assign(y=yte, idx=Xte.index.values).to_csv(
    C.TAB_DIR / "_holdout_predictions.csv", index=False)

# ---------------- Table 6 -------------------------------------------------- #
rows, boot_samples = [], {}
for name in C.MODEL_ORDER:
    f = fold_res[name]
    ci, samp = E.bootstrap_metrics(yte, test_prob[name], n_boot=C.N_BOOT, seed=C.SEED)
    boot_samples[name] = samp
    rows.append({
        "Model": name,
        "Accuracy (CV)": mean_sd(f.accuracy, 4), "Precision (CV)": mean_sd(f.precision, 4),
        "Recall (CV)": mean_sd(f.recall, 4), "F1 (CV)": mean_sd(f.f1, 4),
        "ROC-AUC (CV)": mean_sd(f.roc_auc, 4), "PR-AUC (CV)": mean_sd(f.pr_auc, 4),
        "Accuracy test [95% CI]": E.fmt_ci(ci["accuracy"]), "Recall test [95% CI]": E.fmt_ci(ci["recall"]),
        "F1 test [95% CI]": E.fmt_ci(ci["f1"]), "ROC-AUC test [95% CI]": E.fmt_ci(ci["roc_auc"]),
        "PR-AUC test [95% CI]": E.fmt_ci(ci["pr_auc"]),
    })
t6 = pd.DataFrame(rows)
save_table(t6, "T06_model_comparison",
           "Model comparison: stratified 10-fold CV on the training partition (mean ± SD) and "
           "held-out test performance with stratified bootstrap 95% CIs.",
           note=f"CV columns: 10 folds of the training partition (n = {len(ytr)}). Test columns: "
                f"n = {len(yte)} ({yte.sum()} At-Risk), {C.N_BOOT} stratified bootstrap replicates, "
                "percentile intervals. Threshold = 0.5.")

# ---------------- Table 7 (class-wise) ------------------------------------- #
rows = []
for name in C.MODEL_ORDER:
    f = fold_res[name]
    for cls, sup in [("Not At-Risk", int((1 - yte).sum())), ("At-Risk", int(yte.sum()))]:
        rows.append({"Model": name if cls == "Not At-Risk" else "", "Class": cls,
                     "Precision (%)": mean_sd(f[f"{cls}|precision"], 2, pct=True),
                     "Recall (%)": mean_sd(f[f"{cls}|recall"], 2, pct=True),
                     "F1 (%)": mean_sd(f[f"{cls}|f1"], 2, pct=True),
                     "Test precision (%)": round(100 * E.classwise_metrics(yte, test_prob[name])[f"{cls}|precision"], 2),
                     "Test recall (%)": round(100 * E.classwise_metrics(yte, test_prob[name])[f"{cls}|recall"], 2),
                     "Test F1 (%)": round(100 * E.classwise_metrics(yte, test_prob[name])[f"{cls}|f1"], 2),
                     "Test support": sup})
save_table(pd.DataFrame(rows), "T07_classwise",
           "Class-wise precision, recall and F1 for both classes.",
           note="'(%)' columns: mean ± SD over 10 CV folds of the training partition. 'Test' columns: "
                "single evaluation on the held-out set; support = number of test students per class.")

# ---------------- Table 8 -------------------------------------------------- #
a, b = fold_res["Logistic Regression"], fold_res["XGBoost"]
n_tr = int(len(ytr) * (C.CV_FOLDS - 1) / C.CV_FOLDS)
n_te = len(ytr) - n_tr
rows = []
for metric in ["accuracy", "roc_auc", "pr_auc", "f1"]:
    t, dfw, p, d = E.welch_t(a[metric], b[metric])
    tc, dfc, pc = E.corrected_resampled_t(a[metric].values, b[metric].values, n_tr, n_te)
    rows.append({"Metric": E.METRIC_LABELS[metric],
                 "LR mean ± SD": mean_sd(a[metric], 4), "XGB mean ± SD": mean_sd(b[metric], 4),
                 "Welch t": round(t, 3), "Welch df": round(dfw, 2), "Welch p": round(p, 4),
                 "Corrected t (Nadeau–Bengio)": round(tc, 3), "Corrected p": round(pc, 4),
                 "Cohen's d": round(d, 3),
                 "Significant (corrected, α=.05)": "Yes" if pc < 0.05 else "No"})
save_table(pd.DataFrame(rows), "T08_lr_vs_xgb",
           "Logistic Regression vs XGBoost on 10-fold CV scores: Welch's t-test and the "
           "Nadeau–Bengio corrected resampled t-test.",
           note="CV folds share training data, so Welch's t-test (which assumes independence) is "
                "anti-conservative; the corrected resampled t-test should be used for inference.")

# ---------------- Table 9 -------------------------------------------------- #
def kappa_level(k):
    return ("Poor" if k < 0 else "Slight" if k <= .2 else "Fair" if k <= .4 else
            "Moderate" if k <= .6 else "Substantial" if k <= .8 else "Almost perfect")


rows = []
for name in C.MODEL_ORDER:
    m = E.all_metrics(yte, test_prob[name])
    rows.append({"Model": name, "Accuracy (CV mean)": round(fold_res[name].accuracy.mean(), 4),
                 "Accuracy (test)": round(m["accuracy"], 4), "Cohen's κ (test)": round(m["kappa"], 4),
                 "Agreement level": kappa_level(m["kappa"]), "MCC (test)": round(m["mcc"], 4),
                 "Brier score (test)": round(m["brier"], 4),
                 "Brier skill score": round(1 - m["brier"] / (yte.mean() * (1 - yte.mean())), 4)})
save_table(pd.DataFrame(rows), "T09_kappa_brier",
           "Reliability of the six classifiers: Cohen's κ, MCC and Brier score on the held-out test set.",
           note="Agreement levels follow Landis & Koch (1977). Brier skill score > 0 means better than "
                "always predicting the base rate.")

# ---------------- Table A1 McNemar ----------------------------------------- #
rows = []
for m1, m2 in itertools.combinations(C.MODEL_ORDER, 2):
    b_, c_, chi, p, kind = E.mcnemar(yte, test_prob[m1] >= .5, test_prob[m2] >= .5)
    rows.append({"Model A": m1, "Model B": m2, "b (A✓ B✗)": b_, "c (A✗ B✓)": c_,
                 "χ² (cc)": "" if np.isnan(chi) else round(chi, 3), "Test": kind, "p": p})
a1 = pd.DataFrame(rows)
a1["p (BH-FDR)"] = E.bh_fdr(a1.p)
a1["Sig. (FDR α=.05)"] = np.where(a1["p (BH-FDR)"] < .05, "✓", "✗")
a1[["p", "p (BH-FDR)"]] = a1[["p", "p (BH-FDR)"]].round(4)
save_table(a1, "A01_mcnemar", "Pairwise McNemar tests on hold-out hard predictions.",
           note="Exact binomial test when b + c < 25, otherwise continuity-corrected χ². "
                "p-values adjusted for 15 comparisons with Benjamini–Hochberg.")

# ---------------- Table A2 Friedman ---------------------------------------- #
acc = pd.DataFrame({n: fold_res[n].accuracy.values for n in C.MODEL_ORDER})
chi2, pf, ranks, cd, nem = E.friedman_nemenyi(acc)
a2 = pd.DataFrame({"Rank": range(1, len(ranks) + 1), "Model": ranks.index,
                   "Mean rank": ranks.values.round(2)})
a2["Friedman χ²"] = round(chi2, 3)
a2["df"] = len(ranks) - 1
a2["p"] = f"{pf:.2e}"
a2["Nemenyi CD (α=.05)"] = round(cd, 3)
save_table(a2, "A02_friedman", "Friedman test on fold-wise accuracy ranks (10 folds, six models).",
           note="Models whose mean ranks differ by more than the critical difference (CD) differ "
                "significantly (Nemenyi post-hoc).")
if nem is not None:
    save_table(nem.round(4).reset_index().rename(columns={"index": "Model"}), "A02b_nemenyi",
               "Nemenyi post-hoc p-values (fold-wise accuracy).")

# ---------------- Table A3 DeLong ------------------------------------------ #
rows = []
for m1, m2 in itertools.combinations(C.MODEL_ORDER, 2):
    a_, b_, z, p = E.delong_test(yte, test_prob[m1], test_prob[m2])
    rows.append({"Model A": m1, "Model B": m2, "AUC (A)": round(a_, 4), "AUC (B)": round(b_, 4),
                 "Z": round(z, 4), "p": p})
a3 = pd.DataFrame(rows)
a3["p (BH-FDR)"] = E.bh_fdr(a3.p)
a3["Sig. (FDR α=.05)"] = np.where(a3["p (BH-FDR)"] < .05, "✓", "✗")
a3[["p", "p (BH-FDR)"]] = a3[["p", "p (BH-FDR)"]].round(4)
save_table(a3, "A03_delong", "Pairwise DeLong tests of hold-out ROC-AUC.",
           note="Two-sided; Benjamini–Hochberg adjustment over 15 comparisons.")

# ---------------- Fig 3 ROC + PR ------------------------------------------- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
for name in C.MODEL_ORDER:
    fpr, tpr, _ = roc_curve(yte, test_prob[name])
    auc, lo, hi = E.delong_ci(yte, test_prob[name])
    ax1.plot(fpr, tpr, color=P.MODEL_COLORS[name], ls=P.MODEL_LS[name],
             label=f"{C.MODEL_SHORT[name]}  {auc:.3f} [{lo:.2f}–{hi:.2f}]")
    pr, rc, _ = precision_recall_curve(yte, test_prob[name])
    ap = E.all_metrics(yte, test_prob[name])["pr_auc"]
    ax2.step(rc, pr, where="post", color=P.MODEL_COLORS[name], ls=P.MODEL_LS[name],
             label=f"{C.MODEL_SHORT[name]}  {ap:.3f}")
ax1.plot([0, 1], [0, 1], color=P.MUTED, lw=0.8, ls=":")
ax1.set(xlabel="False-positive rate (1 − specificity)", ylabel="True-positive rate (sensitivity)",
        xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
ax1.set_title("(a) ROC curves", loc="left")
ax1.legend(title="Model   AUC [95% CI]", loc="lower right", fontsize=7, title_fontsize=7.5)
ax2.axhline(yte.mean(), color=P.MUTED, lw=0.8, ls=":")
ax2.text(0.98, yte.mean() + 0.02, f"prevalence = {yte.mean():.3f}", ha="right", fontsize=7, color=P.MUTED)
ax2.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", xlim=(-0.01, 1.01), ylim=(-0.01, 1.03))
ax2.set_title("(b) Precision–recall curves", loc="left")
ax2.legend(title="Model   AP", loc="lower left", bbox_to_anchor=(0.0, 0.09), fontsize=7, title_fontsize=7.5)
fig.tight_layout()
P.save(fig, "Fig03_roc_pr_holdout")

# ---------------- Fig A1 bootstrap histograms ------------------------------ #
fig, axes = plt.subplots(2, 6, figsize=(7.4, 3.0), sharey="row")
for j, name in enumerate(C.MODEL_ORDER):
    for i, metric in enumerate(["accuracy", "f1"]):
        ax = axes[i, j]
        s = boot_samples[name][metric]
        if np.std(s) == 0:
            ax.text(0.5, 0.5, f"constant\n= {s[0]:.3f}\n(no At-Risk\npredictions)" if metric == "f1" else f"constant\n= {s[0]:.3f}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=7, color=P.MUTED)
            ax.set_xticks([]); ax.set_yticks([]) if j else None
            if i == 0:
                ax.set_title(C.MODEL_SHORT[name], fontsize=9)
            continue
        ax.hist(s, bins=30, color=P.MODEL_COLORS[name], alpha=0.85, edgecolor="white", lw=0.3)
        lo, hi = np.percentile(s, [2.5, 97.5])
        for v in (lo, hi):
            ax.axvline(v, color=P.INK, ls="--", lw=0.7)
        ax.tick_params(labelsize=6.5)
        ax.grid(False)
        if i == 0:
            ax.set_title(C.MODEL_SHORT[name], fontsize=9)
        if j == 0:
            ax.set_ylabel("Accuracy" if i == 0 else "F1 (At-Risk)", fontsize=8.5)
fig.suptitle(f"Stratified bootstrap distributions on the hold-out set ({C.N_BOOT} replicates; dashed = 95% CI)",
             x=0.01, ha="left", fontsize=9.5, fontweight="bold")
fig.tight_layout()
P.save(fig, "FigA1_bootstrap_hist")

# ---------------- Fig A2 KDE LR vs XGB ------------------------------------- #
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for name in ["Logistic Regression", "XGBoost"]:
    v = fold_res[name].accuracy.values
    grid = np.linspace(v.min() - 0.03, v.max() + 0.03, 300)
    ax.fill_between(grid, gaussian_kde(v)(grid), color=P.MODEL_COLORS[name], alpha=0.25)
    ax.plot(grid, gaussian_kde(v)(grid), color=P.MODEL_COLORS[name], ls=P.MODEL_LS[name],
            label=f"{name} (mean {v.mean():.3f})")
    ax.plot(v, np.zeros_like(v) - 0.5, "|", color=P.MODEL_COLORS[name], ms=10)
r8 = pd.read_csv(C.TAB_DIR / "T08_lr_vs_xgb.csv").iloc[0]
ax.set_title("Fold-wise accuracy: LR vs XGBoost", loc="left")
welch_p, corr_p, cd_ = r8["Welch p"], r8["Corrected p"], r8["Cohen's d"]
ax.text(0.02, 0.95, f"Welch p = {welch_p:.4f}\nCorrected p = {corr_p:.4f}\nCohen's d = {cd_:.2f}",
        transform=ax.transAxes, va="top", fontsize=7.5)
ax.set(xlabel="10-fold CV accuracy", ylabel="Density")
ax.legend(loc="upper right", fontsize=7)
P.save(fig, "FigA2_kde_lr_xgb")

# ---------------- Fig A3 fold boxplot -------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, metric in zip(axes, ["accuracy", "pr_auc"]):
    data = [fold_res[n][metric].values for n in C.MODEL_ORDER]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=P.INK, lw=1.3))
    for patch, n in zip(bp["boxes"], C.MODEL_ORDER):
        patch.set_facecolor(P.MODEL_COLORS[n]); patch.set_alpha(0.35); patch.set_edgecolor(P.MODEL_COLORS[n])
    for i, (d_, n) in enumerate(zip(data, C.MODEL_ORDER)):
        jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(d_))
        ax.scatter(np.full(len(d_), i + 1) + jit, d_, s=9, color=P.MODEL_COLORS[n], zorder=3)
    ax.set_xticks(range(1, 7), [C.MODEL_SHORT[n] for n in C.MODEL_ORDER])
    ax.set_ylabel(E.METRIC_LABELS[metric])
axes[0].set_title("(a) Accuracy per fold", loc="left")
axes[1].set_title("(b) PR-AUC per fold", loc="left")
fig.tight_layout()
P.save(fig, "FigA3_fold_boxplot")

# ---------------- Fig S5 confusion matrices -------------------------------- #
fig, axes = plt.subplots(1, 6, figsize=(7.4, 1.9))
for ax, n in zip(axes, C.MODEL_ORDER):
    cm = confusion_matrix(yte, test_prob[n] >= .5)
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=P.SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            v = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n({v:.0%})", ha="center", va="center", fontsize=7,
                    color="white" if v > .6 else P.INK)
    ax.set_xticks([0, 1], ["Not", "At-Risk"], fontsize=7)
    ax.set_yticks([0, 1], ["Not", "At-Risk"] if n == C.MODEL_ORDER[0] else ["", ""], fontsize=7)
    ax.set_title(C.MODEL_SHORT[n], fontsize=9)
    ax.grid(False)
axes[0].set_ylabel("True", fontsize=8)
fig.supxlabel("Predicted", fontsize=8, y=-0.04)
fig.tight_layout()
P.save(fig, "FigS5_confusion_matrices")
log.info("done")
