"""
Metrics, bootstrap confidence intervals, and statistical model comparison.

Everything here is pure numpy/scipy so it is easy to audit.

Implemented tests
-----------------
* Bootstrap percentile CIs (stratified resampling keeps the 5 % minority
  represented in every replicate).
* DeLong (1988) test for two correlated ROC AUCs - fast O(n log n) version
  (Sun & Xu, 2014).
* McNemar test on paired hard predictions (exact binomial when b + c < 25,
  continuity-corrected chi-square otherwise).
* Welch's t-test (as used in the first draft) AND the Nadeau-Bengio corrected
  resampled t-test, which is the appropriate test for k-fold CV scores because
  CV folds share training data and are not independent.
* Friedman test + Nemenyi post-hoc + critical-difference values (Demsar, 2006).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision (At-Risk)", "recall": "Recall / Sensitivity (At-Risk)",
    "specificity": "Specificity", "f1": "F1 (At-Risk)", "mcc": "MCC", "kappa": "Cohen's κ",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC (Avg. precision)", "brier": "Brier score",
}


def all_metrics(y, prob, thr: float = 0.5) -> dict:
    y = np.asarray(y)
    prob = np.asarray(prob)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "kappa": cohen_kappa_score(y, pred),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, prob) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, prob),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    return out


def classwise_metrics(y, prob, thr: float = 0.5) -> dict:
    """Precision / recall / F1 for BOTH classes (Table 7)."""
    pred = (np.asarray(prob) >= thr).astype(int)
    out = {}
    for cls, name in [(0, "Not At-Risk"), (1, "At-Risk")]:
        out[f"{name}|precision"] = precision_score(y, pred, pos_label=cls, zero_division=0)
        out[f"{name}|recall"] = recall_score(y, pred, pos_label=cls, zero_division=0)
        out[f"{name}|f1"] = f1_score(y, pred, pos_label=cls, zero_division=0)
    return out


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_metrics(y, prob, n_boot: int = 2000, seed: int = 42, thr: float = 0.5,
                      metrics=("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                               "balanced_accuracy", "mcc", "brier")):
    """Stratified percentile bootstrap. Returns (point, lo, hi, samples)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    prob = np.asarray(prob)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    samples = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = all_metrics(y[idx], prob[idx], thr)
        for k in metrics:
            samples[k].append(m[k])
    point = all_metrics(y, prob, thr)
    res = {}
    for k in metrics:
        s = np.array(samples[k])
        res[k] = (point[k], np.nanpercentile(s, 2.5), np.nanpercentile(s, 97.5))
    return res, {k: np.array(v) for k, v in samples.items()}


def fmt_ci(t, nd: int = 3) -> str:
    p, lo, hi = t
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# --------------------------------------------------------------------------- #
# DeLong
# --------------------------------------------------------------------------- #
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2):
    """Two-sided DeLong test. Returns auc1, auc2, z, p."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")       # positives first
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1), np.asarray(p2)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return aucs[0], aucs[1], z, p


def delong_ci(y, p, alpha=0.05):
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    aucs, cov = _fast_delong(np.asarray(p)[None, order], m)
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return aucs[0], max(0, aucs[0] - z * se), min(1, aucs[0] + z * se)


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #
def mcnemar(y, pred_a, pred_b):
    ca = np.asarray(pred_a) == np.asarray(y)
    cb = np.asarray(pred_b) == np.asarray(y)
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    if b + c == 0:
        return b, c, 0.0, 1.0, "exact"
    if b + c < 25:
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        return b, c, np.nan, p, "exact"
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, chi2, stats.chi2.sf(chi2, 1), "chi2-cc"


# --------------------------------------------------------------------------- #
# Comparing CV score vectors
# --------------------------------------------------------------------------- #
def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = cohens_d(a, b)
    return t, df, p, d


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def corrected_resampled_t(a, b, n_train: int, n_test: int):
    """Nadeau & Bengio (2003) corrected repeated k-fold t-test.

    a, b: paired per-fold scores of two models on the SAME folds.
    """
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, J - 1, 1.0
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return t, J - 1, p


def friedman_nemenyi(score_table: pd.DataFrame, higher_is_better: bool = True):
    """score_table: rows = folds, cols = models.

    Returns (chi2, p, mean_ranks Series, CD at alpha=.05, Nemenyi p-matrix)."""
    S = score_table.values
    chi2, p = stats.friedmanchisquare(*[S[:, j] for j in range(S.shape[1])])
    ranks = pd.DataFrame(S).rank(axis=1, ascending=not higher_is_better).values
    mean_ranks = pd.Series(ranks.mean(axis=0), index=score_table.columns).sort_values()
    k, N = S.shape[1], S.shape[0]
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164}[k]
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(S)
        nem.index = nem.columns = score_table.columns
    except Exception:
        nem = None
    return chi2, p, mean_ranks, cd, nem


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (for the many pairwise tests)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1)
    return out


# --------------------------------------------------------------------------- #
# Calibration & decision-curve analysis
# --------------------------------------------------------------------------- #
def expected_calibration_error(y, prob, n_bins: int = 10, strategy: str = "quantile"):
    y, prob = np.asarray(y), np.asarray(prob)
    if strategy == "quantile":
        edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(prob, edges[1:-1]), 0, len(edges) - 2)
    ece = 0.0
    for b in np.unique(ids):
        m = ids == b
        ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return ece


def calibration_slope_intercept(y, prob):
    """Logistic recalibration: logit(p_obs) = a + b*logit(p_pred)."""
    import statsmodels.api as sm
    p = np.clip(np.asarray(prob), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        fit = sm.GLM(np.asarray(y), sm.add_constant(lp), family=sm.families.Binomial()).fit()
        b = fit.params[1]
        fit0 = sm.GLM(np.asarray(y), np.ones_like(lp), family=sm.families.Binomial(),
                      offset=lp).fit()
        a = fit0.params[0]
        return a, b
    except Exception:
        return np.nan, np.nan


def net_benefit(y, prob, thresholds):
    y, prob = np.asarray(y), np.asarray(prob)
    n = len(y)
    out = []
    for t in thresholds:
        pred = prob >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)
