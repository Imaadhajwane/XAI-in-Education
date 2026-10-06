"""
Explainability utilities: SHAP (model-appropriate explainer), LIME, PDP/ICE,
permutation importance. All functions take a fitted pipeline and the 13-column
base matrix and return attributions on the 13 ORIGINAL features.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap

from . import config as C
from .data import transformed_groups

warnings.filterwarnings("ignore", category=UserWarning)


def _split_pipeline(pipe):
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    return prep, clf


def _collapse(values: np.ndarray, groups: dict, columns) -> pd.DataFrame:
    out = np.zeros((values.shape[0], len(columns)))
    for j, c in enumerate(columns):
        out[:, j] = values[:, groups[c]].sum(axis=1)
    return pd.DataFrame(out, columns=list(columns))


def compute_shap(pipe, X_background: pd.DataFrame, X_explain: pd.DataFrame,
                 model_name: str, n_bg: int = 200, seed: int = C.SEED):
    """Return (shap_df [n x 13], base_value, output_space).

    * Logistic Regression -> exact interventional LinearExplainer, log-odds.
    * Tree models          -> interventional TreeExplainer (log-odds for GB/XGB,
                              probability for RF/DT).
    * SVM                  -> Permutation explainer on probability scale.
    One-hot SHAP columns are summed back onto their original feature (Shapley
    additivity makes this exact).
    """
    prep, clf = _split_pipeline(pipe)
    bg = X_background.sample(min(n_bg, len(X_background)), random_state=seed)
    cols = list(X_explain.columns)

    if model_name == "SVM":
        f = lambda A: pipe.predict_proba(pd.DataFrame(A, columns=cols))[:, 1]
        masker = shap.maskers.Independent(bg.values, max_samples=100)
        ex = shap.PermutationExplainer(f, masker)
        sv = ex(X_explain.values, max_evals=2 * len(cols) + 1, silent=True)
        return pd.DataFrame(sv.values, columns=cols), float(np.mean(sv.base_values)), "probability"

    Xb = prep.transform(bg)
    Xe = prep.transform(X_explain)
    groups = transformed_groups(prep)
    if model_name == "Logistic Regression":
        ex = shap.LinearExplainer(clf, Xb)
        vals = ex.shap_values(Xe)
        base = float(np.ravel(ex.expected_value)[0])
        space = "log-odds"
    else:
        out_space = "raw" if model_name in ("Gradient Boosting", "XGBoost") else "probability"
        try:
            ex = shap.TreeExplainer(clf, data=Xb, feature_perturbation="interventional", model_output=out_space)
            vals = ex.shap_values(Xe, check_additivity=False)
        except Exception:      # some XGBoost builds: fall back to path-dependent TreeSHAP (raw margin)
            ex = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
            vals = ex.shap_values(Xe, check_additivity=False)
            out_space = "raw"
        ev = ex.expected_value
        if isinstance(vals, list):
            vals = vals[1]
            ev = ev[1]
        elif vals.ndim == 3:
            vals = vals[:, :, 1]
            ev = np.ravel(ev)[-1]
        base = float(np.ravel(ev)[0]) if np.ndim(ev) else float(ev)
        space = "log-odds" if out_space == "raw" and model_name in ("Gradient Boosting", "XGBoost") else (
            "probability" if out_space == "probability" else "raw margin")
    return _collapse(np.asarray(vals), groups, cols), base, space


def global_importance(shap_df: pd.DataFrame) -> pd.Series:
    return shap_df.abs().mean().sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# LIME
# --------------------------------------------------------------------------- #
def make_lime_explainer(X_train: pd.DataFrame, seed: int = C.SEED):
    from lime.lime_tabular import LimeTabularExplainer
    from .data import categorical_idx
    cat_idx = categorical_idx(X_train.columns)
    cat_names = {}
    for j in cat_idx:
        c = X_train.columns[j]
        if c in C.ORDINAL:
            cat_names[j] = C.ORDINAL[c]
        elif c in C.NOMINAL:
            cat_names[j] = C.NOMINAL[c]
        else:
            cat_names[j] = ["No", "Yes"]
    return LimeTabularExplainer(
        X_train.values, feature_names=[C.LABELS.get(c, c) for c in X_train.columns], class_names=["Not At-Risk", "At-Risk"],
        categorical_features=cat_idx, categorical_names=cat_names, discretize_continuous=True,
        mode="classification", random_state=seed)


def lime_explain(explainer, pipe, x_row: np.ndarray, columns, num_samples: int = 5000):
    """Return (weights Series over ALL 13 features, explanation object)."""
    f = lambda A: pipe.predict_proba(pd.DataFrame(A, columns=list(columns)))
    exp = explainer.explain_instance(x_row, f, num_features=len(columns), labels=(1,),
                                     num_samples=num_samples)
    w = pd.Series(0.0, index=list(columns))
    for j, v in exp.as_map()[1]:
        w.iloc[j] = v
    return w, exp


# --------------------------------------------------------------------------- #
# PDP / ICE
# --------------------------------------------------------------------------- #
def pdp_ice(pipe, X: pd.DataFrame, feature: str, grid=None, n_grid: int = 40):
    if grid is None:
        lo, hi = np.percentile(X[feature], [1, 99])
        grid = np.linspace(lo, hi, n_grid)
    ice = np.zeros((len(X), len(grid)))
    for g, v in enumerate(grid):
        Xm = X.copy()
        Xm[feature] = v
        ice[:, g] = pipe.predict_proba(Xm)[:, 1]
    return grid, ice.mean(axis=0), ice


def masked_predict(pipe, X: pd.DataFrame, features, reference: pd.Series):
    """Predict with ``features`` replaced by a reference value (mean/mode)."""
    Xm = X.copy()
    for f in features:
        Xm[f] = reference[f]
    return pipe.predict_proba(Xm)[:, 1]


def reference_values(X_train: pd.DataFrame) -> pd.Series:
    """Mean for numeric, mode for categorical - used for 'feature removal'."""
    ref = X_train.mean()
    for c in C.CATEGORICAL:
        ref[c] = X_train[c].mode().iloc[0]
    return ref


def decision_threshold(model_name: str = C.FINAL_MODEL) -> float:
    """F1-optimal threshold chosen on training-partition out-of-fold predictions (script 01)."""
    import json
    f = C.MODEL_DIR / "decision_thresholds.json"
    return float(json.loads(f.read_text())[model_name]) if f.exists() else 0.5
