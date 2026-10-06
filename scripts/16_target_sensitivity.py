"""
16 - Sensitivity to the At-Risk threshold τ  (NEW)
==================================================
The draft notes the 50% cut-off is institution-dependent. This script shows
how prevalence, discrimination and the PREX-Edu explanation change for
τ ∈ {40, 45, 50, 55, 60} so the conclusions are not an artefact of one cut-off.

Outputs
  Table S26  performance + explanation stability by τ   S26_threshold_sensitivity
  Fig S22    sensitivity plot                           FigS22_threshold_sensitivity
"""
from _common import C, get_logger

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold

from prexedu import data as D
from prexedu import evaluation as E
from prexedu import models as M
from prexedu import plotting as P
from prexedu import prex, xai
from prexedu.tables import mean_sd, save_table

log = get_logger("16_target_sensitivity")
TAUS = [40, 45, 50, 55, 60]
REP = 2 if C.FAST else 5
MODELS = ["Logistic Regression", "XGBoost"]
best = {}
for m in MODELS:
    est = joblib.load(C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[m]}.joblib").named_steps["clf"]
    best[m] = {k: est.get_params()[k.replace("clf__", "")] for k in M.param_grids()[m]}


def one(m, X, y, tr, te):
    pp = M.make_pipeline(m, X.columns, C.PRIMARY_STRATEGY, params=best[m]).fit(X.iloc[tr], y[tr])
    return E.all_metrics(y[te], pp.predict_proba(X.iloc[te])[:, 1])


rows, ref_imp, curves = [], None, {}
for tau in TAUS:
    df, X, y = D.load_xy(threshold=tau)
    splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=REP, random_state=C.SEED).split(X, y))
    res = {m: pd.DataFrame(Parallel(n_jobs=C.N_JOBS)(delayed(one)(m, X, y, tr, te) for tr, te in splits)) for m in MODELS}
    # explanation on a hold-out split
    Xtr, Xte, ytr, yte = D.holdout_split(X, y)
    pl = M.make_pipeline("Logistic Regression", X.columns, C.PRIMARY_STRATEGY, params=best["Logistic Regression"]).fit(Xtr, ytr)
    S, _, _ = xai.compute_shap(pl, Xtr, Xte, "Logistic Regression", n_bg=len(Xtr))
    imp = S.abs().mean()
    if tau == 50:
        ref_imp = imp
    R = prex.prex_scores(S)
    pr = pl.predict_proba(Xte)[:, 1]
    dom = R.idxmax(axis=1)[pr >= np.quantile(pr, 1 - y.mean())]      # top-risk students (same share as prevalence)
    curves[tau] = (res, imp, R.mean())
    row = {"τ (last-term % ≤)": tau, "At-Risk n": int(y.sum()), "Prevalence": f"{y.mean():.1%}"}
    for m in MODELS:
        row[f"{C.MODEL_SHORT[m]} ROC-AUC"] = mean_sd(res[m].roc_auc)
        row[f"{C.MODEL_SHORT[m]} PR-AUC"] = mean_sd(res[m].pr_auc)
        row[f"{C.MODEL_SHORT[m]} PR-AUC lift"] = round(res[m].pr_auc.mean() / y.mean(), 1)
    row["Top-3 SHAP features"] = ", ".join(C.LABELS[f] for f in imp.nlargest(3).index)
    row["Dominant dimension (top-risk students)"] = "; ".join(f"{k} {v:.0%}" for k, v in dom.value_counts(normalize=True).items())
    rows.append(row)
    log.info(f"τ={tau}: prevalence {y.mean():.3f}")
for row, tau in zip(rows, TAUS):
    row["Global SHAP rank ρ vs τ=50"] = round(stats.spearmanr(curves[tau][1], ref_imp)[0], 3)
save_table(pd.DataFrame(rows), "S26_threshold_sensitivity",
           "Sensitivity of discrimination and explanations to the At-Risk threshold τ.",
           note=f"{REP}×10 repeated stratified CV (hyper-parameters fixed at the τ = 50 selection). PR-AUC lift = PR-AUC / "
                f"prevalence (random classifier = 1). Explanations from the hold-out LR model at each τ.")

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
fig.suptitle("Error bars: 95% CI of the mean over CV folds", x=0.99, ha="right", fontsize=7.5, color=P.MUTED)
for m in MODELS:
    for ax, met in zip(axes[:2], ["roc_auc", "pr_auc"]):
        mu = [curves[t][0][m][met].mean() for t in TAUS]
        ci = [1.96 * curves[t][0][m][met].std() / np.sqrt(len(curves[t][0][m])) for t in TAUS]
        ax.errorbar(TAUS, mu, yerr=ci, color=P.MODEL_COLORS[m], marker=P.MODEL_MARK[m], ms=4, capsize=2,
                    ls=P.MODEL_LS[m], label=C.MODEL_SHORT[m])
for ax, t_ in zip(axes[:2], ["(a) ROC-AUC", "(b) PR-AUC"]):
    ax.set_title(t_, loc="left", fontsize=9.5)
    ax.set_xlabel("τ (last-term % ≤ τ)")
    ax.axvline(50, color=P.MUTED, ls=":", lw=0.8)
axes[0].legend(fontsize=7)
ax = axes[2]
for k in C.PREX_DIMENSIONS:
    ax.plot(TAUS, [curves[t][2][k] for t in TAUS], marker="o", ms=3.5, color=P.DIM_COLORS[k], label=k)
ax.set_title("(c) Mean R_k by dimension", loc="left", fontsize=9.5)
ax.set_xlabel("τ")
ax.axvline(50, color=P.MUTED, ls=":", lw=0.8)
ax.legend(fontsize=6.5)
fig.tight_layout()
P.save(fig, "FigS22_threshold_sensitivity")
log.info("done")
