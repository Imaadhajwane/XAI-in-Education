"""
11 - Class-imbalance strategy comparison  (NEW)
==============================================
Why: the first draft applied SMOTE but did not show that SMOTE was the right
choice, and reviewers in learning analytics increasingly know that
resampling can damage probability calibration (van den Goorbergh et al., 2022).
This script compares seven strategies under 5 x 10 repeated stratified CV on
ALL 1,208 students, with the resampler re-fitted inside every training fold.

Outputs
  Table S5  strategy x model performance (ROC-AUC, PR-AUC, recall, F1, MCC,
            Brier, ECE)                                       S05_imbalance_strategies
  Fig S6    strategy comparison (PR-AUC, recall, Brier)      FigS6_imbalance_strategies
"""
from _common import C, Timer, get_logger

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from prexedu import data as D
from prexedu import evaluation as E
from prexedu import models as M
from prexedu import plotting as P
from prexedu.tables import mean_sd, save_table

log = get_logger("11_imbalance")
df, X, y = D.load_xy()
MODELS = ["Logistic Regression", "Random Forest", "XGBoost"]
REPEATS = 2 if C.FAST else 5


def tuned_params(name):
    """Re-use the hyper-parameters selected in script 01 if available."""
    p = C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[name]}.joblib"
    if p.exists():
        est = joblib.load(p).named_steps["clf"]
        keys = M.param_grids()[name].keys()
        return {k: est.get_params()[k.replace("clf__", "")] for k in keys}
    return {}


def run(name, strat, tr, te, rep, fold):
    pipe = M.make_pipeline(name, X.columns, strat, params=tuned_params(name), seed=C.SEED + rep)
    pipe.fit(X.iloc[tr], y[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    d = E.all_metrics(y[te], p)
    d.update(model=name, strategy=strat, rep=rep, fold=fold)
    return d, (name, strat, rep, te, p)


splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=REPEATS, random_state=C.SEED).split(X, y))
jobs = [delayed(run)(m, s, tr, te, i // 10, i % 10) for m in MODELS for s in M.IMBALANCE_STRATEGIES
        for i, (tr, te) in enumerate(splits)]
with Timer(log, f"{len(jobs)} fits"):
    out = Parallel(n_jobs=C.N_JOBS)(jobs)
folds = pd.DataFrame([o[0] for o in out])
folds.to_csv(C.TAB_DIR / "_imbalance_fold_metrics.csv", index=False)

# pooled out-of-fold probabilities per repeat -> calibration (ECE) on all 1,208
ece = {}
for (m, s) in [(m, s) for m in MODELS for s in M.IMBALANCE_STRATEGIES]:
    vals = []
    for r in range(REPEATS):
        p = np.zeros(len(y))
        for (mm, ss, rr, te, pp) in [o[1] for o in out]:
            if mm == m and ss == s and rr == r:
                p[te] = pp
        vals.append(E.expected_calibration_error(y, p))
    ece[(m, s)] = vals

rows = []
for m in MODELS:
    for s in M.IMBALANCE_STRATEGIES:
        f = folds[(folds.model == m) & (folds.strategy == s)]
        rows.append({"Model": m, "Strategy": M.STRATEGY_LABELS[s],
                     "ROC-AUC": mean_sd(f.roc_auc), "PR-AUC": mean_sd(f.pr_auc),
                     "Recall (At-Risk)": mean_sd(f.recall), "Precision (At-Risk)": mean_sd(f.precision),
                     "F1 (At-Risk)": mean_sd(f.f1), "MCC": mean_sd(f.mcc),
                     "Balanced acc.": mean_sd(f.balanced_accuracy), "Brier": mean_sd(f.brier, 4),
                     "ECE": mean_sd(ece[(m, s)], 4)})
save_table(pd.DataFrame(rows), "S05_imbalance_strategies",
           f"Comparison of class-imbalance strategies ({REPEATS}×10 repeated stratified CV, all 1,208 students).",
           note="Resamplers are fitted inside each training fold only. Threshold = 0.5. ECE = expected "
                "calibration error on pooled out-of-fold probabilities (10 quantile bins), mean ± SD over repeats. "
                "Ranking metrics (ROC/PR-AUC) are threshold-free; recall/F1 depend on the 0.5 threshold.")

# ---------------- Fig S6 --------------------------------------------------- #
metrics = [("pr_auc", "PR-AUC (↑)"), ("recall", "Recall, At-Risk (↑)"), ("brier", "Brier score (↓)")]
strats = M.IMBALANCE_STRATEGIES
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True)
ypos = np.arange(len(strats))
off = {m: o for m, o in zip(MODELS, [-0.22, 0, 0.22])}
for ax, (met, lab) in zip(axes, metrics):
    for m in MODELS:
        mu = [folds[(folds.model == m) & (folds.strategy == s)][met].mean() for s in strats]
        sd = [folds[(folds.model == m) & (folds.strategy == s)][met].std() for s in strats]
        se = np.array(sd) / np.sqrt(len(splits))
        ax.errorbar(mu, ypos + off[m], xerr=1.96 * se, fmt=P.MODEL_MARK[m], color=P.MODEL_COLORS[m],
                    ms=4.5, lw=1, capsize=0, label=C.MODEL_SHORT[m])
    ax.set_title(lab, loc="left", fontsize=9.5)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ypos, [M.STRATEGY_LABELS[s] for s in strats])
axes[0].invert_yaxis()
axes[0].legend(loc="lower left", fontsize=7.5)
fig.suptitle("Effect of imbalance-handling strategy (mean ± 95% CI of fold mean)", x=0.01, ha="left",
             fontweight="bold", fontsize=10)
fig.tight_layout()
P.save(fig, "FigS6_imbalance_strategies")
log.info("done")
