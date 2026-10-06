"""
Cross-validation engines.

``cv_predict``           - plain stratified K-fold, returns per-fold metrics and
                           out-of-fold (OOF) probabilities.
``nested_repeated_cv``   - R x K repeated stratified CV with an INNER grid search
                           in every outer training fold. Every one of the 1,208
                           students (and all 66 At-Risk students) is scored
                           out-of-sample R times. This is the primary protocol
                           of the revised paper and directly answers the
                           "test set too small" criticism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from . import config as C
from .evaluation import all_metrics, classwise_metrics


def cv_predict(pipe, X, y, n_splits: int = C.CV_FOLDS, seed: int = C.SEED, thr: float = 0.5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows, oof = [], np.zeros(len(y))
    for k, (tr, te) in enumerate(skf.split(X, y)):
        m = clone(pipe).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        d = all_metrics(y[te], p, thr)
        d.update(classwise_metrics(y[te], p, thr))
        d["fold"] = k
        rows.append(d)
    return pd.DataFrame(rows), oof


def _outer_fold(model_name, make_pipe, grid, X, y, tr, te, rep, fold, inner, scoring, seed):
    pipe = make_pipe(model_name)
    inner_cv = StratifiedKFold(n_splits=inner, shuffle=True, random_state=seed + rep)
    gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner_cv, n_jobs=1, refit=True)
    gs.fit(X.iloc[tr], y[tr])
    p = gs.predict_proba(X.iloc[te])[:, 1]
    # threshold chosen on INNER out-of-fold predictions (no peeking at the test fold)
    best = gs.best_estimator_
    inner_oof = np.zeros(len(tr))
    for itr, ite in inner_cv.split(X.iloc[tr], y[tr]):
        mm = clone(best).fit(X.iloc[tr].iloc[itr], y[tr][itr])
        inner_oof[ite] = mm.predict_proba(X.iloc[tr].iloc[ite])[:, 1]
    thr_f1 = _best_f1_threshold(y[tr], inner_oof)
    return {"model": model_name, "rep": rep, "fold": fold, "test_idx": te, "prob": p,
            "best_params": gs.best_params_, "thr_f1": thr_f1}


def _best_f1_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    return float(th[np.nanargmax(f1[:-1])]) if len(th) else 0.5


def nested_repeated_cv(model_names, make_pipe, grids, X, y, repeats=C.CV_REPEATS,
                       folds=C.CV_FOLDS, inner=C.INNER_FOLDS, scoring=C.TUNING_SCORING,
                       seed=C.SEED, n_jobs=C.N_JOBS, verbose=0):
    """Returns (fold_df, oof_df).

    fold_df: one row per model x repeat x fold with all metrics at thr=0.5 and
             at the inner-CV F1-optimal threshold.
    oof_df:  long table (model, rep, idx, y, prob, prob-threshold) - all 1,208
             students x repeats x models.
    """
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    splits = list(rskf.split(X, y))
    jobs = []
    for m in model_names:
        for s, (tr, te) in enumerate(splits):
            jobs.append(delayed(_outer_fold)(m, make_pipe, grids[m], X, y, tr, te,
                                             s // folds, s % folds, inner, scoring, seed))
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
    rows, oof = [], []
    for r in res:
        te = r["test_idx"]
        for thr_name, thr in [("0.5", 0.5), ("tuned", r["thr_f1"])]:
            d = all_metrics(y[te], r["prob"], thr)
            d.update(classwise_metrics(y[te], r["prob"], thr))
            d.update({"model": r["model"], "rep": r["rep"], "fold": r["fold"],
                      "threshold": thr_name, "thr_value": thr,
                      "best_params": str(r["best_params"])})
            rows.append(d)
        oof.append(pd.DataFrame({"model": r["model"], "rep": r["rep"], "fold": r["fold"],
                                 "idx": te, "y": y[te], "prob": r["prob"],
                                 "thr_f1": r["thr_f1"]}))
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)
