"""
10 - PRIMARY EVALUATION: nested, repeated stratified cross-validation  (NEW)
===========================================================================
This is the analysis that answers the JLA rejection ("test set too small").

Design
  * Outer loop : R x 10 repeated stratified K-fold on ALL 1,208 students
                 (R = 10 -> 100 outer test folds). Every student - including
                 all 66 At-Risk students - is predicted out-of-sample R times.
  * Inner loop : 5-fold GridSearchCV (PR-AUC) inside every outer training fold
                 -> hyper-parameter selection never sees the outer test fold.
  * Resampling : PRIMARY_STRATEGY (default SMOTE-NC) fitted inside each
                 training fold only.
  * Threshold  : (i) 0.5 and (ii) the F1-optimal threshold chosen on INNER
                 out-of-fold predictions (no test-fold peeking).
  * Inference  : Nadeau–Bengio corrected repeated-CV t-test (pairwise, BH-FDR),
                 Friedman + Nemenyi with critical-difference diagram, DeLong
                 and McNemar on per-student averaged OOF predictions (n = 1,208,
                 66 positives instead of 19-20).

Outputs
  Table P1  primary model comparison                   P01_nested_cv_performance
  Table P2  class-wise performance                     P02_nested_cv_classwise
  Table P3  pairwise corrected t-tests (ROC/PR-AUC)    P03_corrected_ttests
  Table P4  Friedman / Nemenyi                         P04_friedman_nemenyi
  Table P5  DeLong on OOF (n = 1,208)                  P05_delong_oof
  Table P6  McNemar on OOF                             P06_mcnemar_oof
  Table P7  hyper-parameter selection frequency        P07_param_frequency
  Fig P1    ROC + PR curves (OOF, all students)        FigP1_roc_pr_oof
  Fig P2    critical-difference diagram                FigP2_cd_diagram
  Fig P3    fold distribution of PR-AUC / ROC-AUC      FigP3_fold_distributions
  Data      data/processed/oof_predictions.csv
"""
from _common import C, Timer, get_logger

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from prexedu import data as D
from prexedu import evaluation as E
from prexedu import models as M
from prexedu import plotting as P
from prexedu.cv import nested_repeated_cv
from prexedu.tables import mean_sd, save_table

log = get_logger("10_nested_cv")
df, X, y = D.load_xy()
N = len(y)
log.info(f"N={N}, positives={y.sum()}, repeats={C.CV_REPEATS}, strategy={C.PRIMARY_STRATEGY}")

make = lambda name: M.make_pipeline(name, X.columns, C.PRIMARY_STRATEGY)
cache_f, cache_o = C.TAB_DIR / "_nested_fold_metrics.csv", C.DATA_PROCESSED / "oof_predictions_long.csv"
if os.environ.get("PREX_REUSE_CV") == "1" and cache_f.exists() and cache_o.exists():
    # re-draw tables/figures from a previous run without re-fitting (set PREX_REUSE_CV=1)
    log.info("PREX_REUSE_CV=1 -> loading cached nested-CV results")
    folds, oof = pd.read_csv(cache_f), pd.read_csv(cache_o)
    folds["threshold"] = folds["threshold"].astype(str)
else:
    with Timer(log, "nested repeated CV (all models)"):
        folds, oof = nested_repeated_cv(C.MODEL_ORDER, make, M.param_grids(), X, y, verbose=0)
    folds.to_csv(cache_f, index=False)
    oof.to_csv(cache_o, index=False)

# per-student mean OOF probability across repeats (one score per student & model)
avg = oof.groupby(["model", "idx"]).agg(prob=("prob", "mean"), thr=("thr_f1", "median")).reset_index()
P_oof = avg.pivot(index="idx", columns="model", values="prob")[C.MODEL_ORDER].sort_index()
THR = avg.groupby("model").thr.median()
P_oof.assign(y=y).to_csv(C.DATA_PROCESSED / "oof_predictions.csv")

# ---------------- Table P1 ------------------------------------------------- #
rows = []
for name in C.MODEL_ORDER:
    f5 = folds[(folds.model == name) & (folds.threshold == "0.5")]
    ft = folds[(folds.model == name) & (folds.threshold == "tuned")]
    ci, _ = E.bootstrap_metrics(y, P_oof[name].values, n_boot=C.N_BOOT, seed=C.SEED)
    cit, _ = E.bootstrap_metrics(y, P_oof[name].values, n_boot=C.N_BOOT, seed=C.SEED, thr=THR[name],
                                 metrics=("recall", "precision", "f1", "mcc"))
    rows.append({
        "Model": name,
        "ROC-AUC": mean_sd(f5.roc_auc), "ROC-AUC pooled [95% CI]": E.fmt_ci(ci["roc_auc"]),
        "PR-AUC": mean_sd(f5.pr_auc), "PR-AUC pooled [95% CI]": E.fmt_ci(ci["pr_auc"]),
        "Accuracy": mean_sd(f5.accuracy), "Balanced acc.": mean_sd(f5.balanced_accuracy),
        "Recall@0.5": mean_sd(f5.recall), "Precision@0.5": mean_sd(f5.precision),
        "F1@0.5": mean_sd(f5.f1), "MCC@0.5": mean_sd(f5.mcc), "Brier": mean_sd(f5.brier, 4),
        "Tuned thr. (median)": round(THR[name], 3),
        "Recall@tuned": mean_sd(ft.recall), "F1@tuned": mean_sd(ft.f1),
        "F1@tuned pooled [95% CI]": E.fmt_ci(cit["f1"]),
    })
p1 = pd.DataFrame(rows)
save_table(p1, "P01_nested_cv_performance",
           f"Primary evaluation: nested {C.CV_REPEATS}×{C.CV_FOLDS}-fold repeated stratified cross-validation "
           f"on all {N} students ({y.sum()} At-Risk).",
           note=f"'mean ± SD' over {C.CV_REPEATS * C.CV_FOLDS} outer test folds. 'pooled [95% CI]': metric on the "
                f"per-student out-of-fold probability averaged over repeats (n = {N}), stratified bootstrap "
                f"({C.N_BOOT} replicates). Hyper-parameters tuned by inner {C.INNER_FOLDS}-fold GridSearchCV "
                f"(PR-AUC); tuned threshold maximises F1 on inner out-of-fold predictions. Imbalance handling: "
                f"{M.STRATEGY_LABELS[C.PRIMARY_STRATEGY]} inside training folds only.")

# ---------------- Table P2 class-wise -------------------------------------- #
rows = []
for name in C.MODEL_ORDER:
    for thr_name in ["0.5", "tuned"]:
        f = folds[(folds.model == name) & (folds.threshold == thr_name)]
        for cls in ["Not At-Risk", "At-Risk"]:
            rows.append({"Model": name, "Threshold": thr_name, "Class": cls,
                         "Precision (%)": mean_sd(f[f"{cls}|precision"], 2, True),
                         "Recall (%)": mean_sd(f[f"{cls}|recall"], 2, True),
                         "F1 (%)": mean_sd(f[f"{cls}|f1"], 2, True),
                         "Support (per repeat)": int((y == (cls == "At-Risk")).sum())})
save_table(pd.DataFrame(rows), "P02_nested_cv_classwise",
           "Class-wise performance under nested repeated cross-validation.",
           note="Mean ± SD over outer folds. Support = number of students of that class, each scored once per repeat.")

# ---------------- Table P3 corrected t-tests ------------------------------- #
n_tr = int(N * (C.CV_FOLDS - 1) / C.CV_FOLDS)
n_te = N - n_tr
f5 = folds[folds.threshold == "0.5"].sort_values(["rep", "fold"])
rows = []
for metric in ["roc_auc", "pr_auc", "mcc"]:
    for a, b in itertools.combinations(C.MODEL_ORDER, 2):
        sa = f5[f5.model == a][metric].values
        sb = f5[f5.model == b][metric].values
        t, dfree, p = E.corrected_resampled_t(sa, sb, n_tr, n_te)
        rows.append({"Metric": E.METRIC_LABELS[metric], "Model A": a, "Model B": b,
                     "Mean diff (A−B)": round(np.mean(sa - sb), 4), "t": round(t, 3), "df": dfree, "p": p})
p3 = pd.DataFrame(rows)
p3["p (BH-FDR)"] = p3.groupby("Metric").p.transform(E.bh_fdr)
p3["Sig."] = np.where(p3["p (BH-FDR)"] < .05, "✓", "✗")
p3[["p", "p (BH-FDR)"]] = p3[["p", "p (BH-FDR)"]].round(4)
save_table(p3, "P03_corrected_ttests",
           "Pairwise model comparison with the Nadeau–Bengio corrected repeated-CV t-test.",
           note="Corrects for the overlap of training sets across CV folds (variance inflated by 1/J + n_test/n_train). "
                "Benjamini–Hochberg adjustment within each metric (15 comparisons).")

# ---------------- Table P4 Friedman / Nemenyi + Fig P2 CD ------------------ #
cd_res = {}
rows = []
for metric in ["roc_auc", "pr_auc"]:
    tab = f5.pivot_table(index=["rep", "fold"], columns="model", values=metric)[C.MODEL_ORDER]
    chi2, pf, ranks, cd, nem = E.friedman_nemenyi(tab)
    cd_res[metric] = (ranks, cd, pf)
    for i, (m, r) in enumerate(ranks.items()):
        rows.append({"Metric": E.METRIC_LABELS[metric], "Rank": i + 1, "Model": m, "Mean rank": round(r, 3),
                     "Friedman χ²": round(chi2, 2) if i == 0 else "", "p": f"{pf:.2e}" if i == 0 else "",
                     "CD (α=.05)": round(cd, 3) if i == 0 else ""})
    if nem is not None:
        save_table(nem.round(4).reset_index().rename(columns={"index": "Model"}),
                   f"P04b_nemenyi_{metric}", f"Nemenyi post-hoc p-values ({E.METRIC_LABELS[metric]}).")
save_table(pd.DataFrame(rows), "P04_friedman_nemenyi",
           f"Friedman test with Nemenyi post-hoc over {len(tab)} outer folds.")


def cd_diagram(ax, ranks, cd, title):
    k = len(ranks)
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.8, lo, hi, color=P.INK, lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0.8, 0.84, color=P.INK, lw=1)
        ax.text(t, 0.87, str(t), ha="center", fontsize=8)
    ax.hlines(0.97, lo, lo + cd, color=P.RISK, lw=2)
    ax.text(lo + cd / 2, 0.99, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7.5, color=P.RISK)
    items = list(ranks.items())
    half = (k + 1) // 2
    for i, (m, r) in enumerate(items):
        left = i < half
        yy = 0.65 - 0.13 * (i if left else (k - 1 - i))
        xe = lo - 0.2 if left else hi + 0.2
        ax.plot([r, r, xe], [0.8, yy, yy], color=P.MODEL_COLORS[m], lw=1.2)
        ax.text(xe + (-0.05 if left else 0.05), yy, f"{C.MODEL_SHORT[m]} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=8)
    # cliques: groups not significantly different
    rv = ranks.values
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and rv[j + 1] - rv[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n_, (i, j) in enumerate(cliques):
        yy = 0.76 - 0.035 * n_
        ax.hlines(yy, rv[i] - 0.03, rv[j] + 0.03, color=P.INK, lw=2.4)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=20)


fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.4))
for ax, metric in zip(axes, ["roc_auc", "pr_auc"]):
    ranks, cd, pf = cd_res[metric]
    cd_diagram(ax, ranks, cd, f"{E.METRIC_LABELS[metric]}  (Friedman p = {pf:.1e}; lower rank = better)")
fig.tight_layout()
P.save(fig, "FigP2_cd_diagram")

# ---------------- Table P5 DeLong / P6 McNemar on OOF ---------------------- #
rows, rows_m = [], []
for a, b in itertools.combinations(C.MODEL_ORDER, 2):
    A, B, z, p = E.delong_test(y, P_oof[a].values, P_oof[b].values)
    rows.append({"Model A": a, "Model B": b, "AUC (A)": round(A, 4), "AUC (B)": round(B, 4),
                 "Z": round(z, 3), "p": p})
    bb, cc, chi, pm, kind = E.mcnemar(y, P_oof[a].values >= .5, P_oof[b].values >= .5)
    rows_m.append({"Model A": a, "Model B": b, "b (A✓ B✗)": bb, "c (A✗ B✓)": cc,
                   "χ² (cc)": "" if np.isnan(chi) else round(chi, 3), "Test": kind, "p": pm})
for rws, name, cap in [(rows, "P05_delong_oof", "DeLong tests on per-student out-of-fold probabilities (n = %d)." % N),
                       (rows_m, "P06_mcnemar_oof", "McNemar tests on per-student out-of-fold predictions (n = %d, threshold 0.5)." % N)]:
    t = pd.DataFrame(rws)
    t["p (BH-FDR)"] = E.bh_fdr(t.p)
    t["Sig."] = np.where(t["p (BH-FDR)"] < .05, "✓", "✗")
    t[["p", "p (BH-FDR)"]] = t[["p", "p (BH-FDR)"]].round(4)
    save_table(t, name, cap, note="Benjamini–Hochberg adjustment over 15 comparisons.")

# ---------------- Table P7 hyper-parameter stability ----------------------- #
bp = folds[folds.threshold == "0.5"].groupby(["model", "best_params"]).size().reset_index(name="n")
bp["Share of outer folds (%)"] = (100 * bp.n / bp.groupby("model").n.transform("sum")).round(1)
bp = bp.sort_values(["model", "n"], ascending=[True, False]).groupby("model").head(3)
save_table(bp.rename(columns={"model": "Model", "best_params": "Selected hyper-parameters", "n": "Outer folds"}),
           "P07_param_frequency", "Most frequently selected hyper-parameters across outer folds (top 3 per model).")

# ---------------- Fig P1 ROC / PR on OOF ----------------------------------- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
for name in C.MODEL_ORDER:
    p = P_oof[name].values
    fpr, tpr, _ = roc_curve(y, p)
    auc, lo, hi = E.delong_ci(y, p)
    ax1.plot(fpr, tpr, color=P.MODEL_COLORS[name], ls=P.MODEL_LS[name],
             label=f"{C.MODEL_SHORT[name]}  {auc:.3f} [{lo:.3f}–{hi:.3f}]")
    pr, rc, _ = precision_recall_curve(y, p)
    ap = E.all_metrics(y, p)["pr_auc"]
    ax2.step(rc, pr, where="post", color=P.MODEL_COLORS[name], ls=P.MODEL_LS[name],
             label=f"{C.MODEL_SHORT[name]}  {ap:.3f}")
ax1.plot([0, 1], [0, 1], color=P.MUTED, lw=0.8, ls=":")
ax1.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
ax1.set_title("(a) ROC — out-of-fold, all students", loc="left")
ax1.legend(title="Model   AUC [DeLong 95% CI]", loc="lower right", fontsize=6.8, title_fontsize=7.2)
ax2.axhline(y.mean(), color=P.MUTED, lw=0.8, ls=":")
ax2.text(0.98, y.mean() + 0.02, f"prevalence = {y.mean():.3f}", ha="right", fontsize=7, color=P.MUTED)
ax2.set(xlabel="Recall", ylabel="Precision", xlim=(-0.01, 1.01), ylim=(-0.01, 1.03))
ax2.set_title("(b) Precision–recall", loc="left")
ax2.legend(title="Model   AP", loc="lower left", bbox_to_anchor=(0.0, 0.09), fontsize=6.8, title_fontsize=7.2)
fig.tight_layout()
P.save(fig, "FigP1_roc_pr_oof")

# ---------------- Fig P3 fold distributions -------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
for ax, metric in zip(axes, ["roc_auc", "pr_auc", "mcc"]):
    data = [f5[f5.model == n][metric].values for n in C.MODEL_ORDER]
    vp = ax.violinplot(data, showextrema=False, widths=0.75)
    for body, n in zip(vp["bodies"], C.MODEL_ORDER):
        body.set_facecolor(P.MODEL_COLORS[n]); body.set_alpha(0.35); body.set_edgecolor("none")
    ax.boxplot(data, widths=0.16, showfliers=False, medianprops=dict(color=P.INK, lw=1.2),
               boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    ax.set_xticks(range(1, 7), [C.MODEL_SHORT[n] for n in C.MODEL_ORDER], fontsize=7.5)
    ax.set_title(E.METRIC_LABELS[metric], loc="left", fontsize=9.5)
fig.suptitle(f"Distribution over {len(f5) // 6} outer test folds (nested repeated CV)", x=0.01, ha="left",
             fontweight="bold", fontsize=10)
fig.tight_layout()
P.save(fig, "FigP3_fold_distributions")
log.info("done")
