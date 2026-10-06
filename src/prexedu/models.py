"""
Model zoo, hyper-parameter grids and imbalance-handling strategies.

Every model is an ``imblearn.Pipeline``:

    [sampler] -> [round categorical codes] -> preprocess -> classifier

* The sampler runs ONLY during ``fit`` (imblearn guarantees this), so synthetic
  records never reach a validation / test fold.
* ``RoundCategorical`` snaps any fractional category code produced by an
  interpolating sampler back onto a valid level (no-op on real data).
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTENC, BorderlineSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as C
from .data import categorical_idx, make_preprocessor


class RoundCategorical(BaseEstimator, TransformerMixin):
    """Round + clip categorical columns to valid integer codes."""

    def __init__(self, idx=None, maxima=None):
        self.idx = idx
        self.maxima = maxima

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_df = hasattr(X, "iloc")
        A = X.to_numpy(dtype=float, copy=True) if is_df else np.array(X, dtype=float, copy=True)
        for j, m in zip(self.idx or [], self.maxima or []):
            A[:, j] = np.clip(np.round(A[:, j]), 0, m)
        if is_df:
            import pandas as pd
            return pd.DataFrame(A, columns=X.columns, index=X.index)
        return A


def _cat_maxima(columns):
    out = []
    for c in C.CATEGORICAL:
        if c in C.ORDINAL:
            out.append(len(C.ORDINAL[c]) - 1)
        elif c in C.NOMINAL:
            out.append(len(C.NOMINAL[c]) - 1)
        else:
            out.append(1)
    return out


# --------------------------------------------------------------------------- #
def base_estimators(seed: int = C.SEED, class_weight=None) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight=class_weight,
                                                  random_state=seed),
        "SVM": SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight=class_weight,
                                                n_jobs=1, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(class_weight=class_weight, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", n_jobs=1, random_state=seed,
                                 scale_pos_weight=1.0 if class_weight is None else 17.3),
    }


def param_grids(fast: bool = C.FAST) -> dict:
    if fast:
        return {
            "Logistic Regression": {"clf__C": [0.1, 1, 10]},
            "SVM": {"clf__C": [1, 10], "clf__gamma": ["scale"]},
            "Random Forest": {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 5]},
            "Decision Tree": {"clf__max_depth": [3, 5], "clf__min_samples_leaf": [5]},
            "Gradient Boosting": {"clf__learning_rate": [0.1], "clf__max_depth": [2, 3]},
            "XGBoost": {"clf__learning_rate": [0.1], "clf__max_depth": [3]},
        }
    return {
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        "SVM": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.1, 0.01]},
        "Random Forest": {"clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 5]},
        "Decision Tree": {"clf__max_depth": [3, 5, 7, None], "clf__min_samples_leaf": [1, 5, 10]},
        "Gradient Boosting": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                              "clf__max_depth": [2, 3]},
        "XGBoost": {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5]},
    }


# --------------------------------------------------------------------------- #
IMBALANCE_STRATEGIES = ["none", "class_weight", "random_over", "random_under",
                        "smotenc", "borderline_smote", "adasyn"]
STRATEGY_LABELS = {
    "none": "No resampling", "class_weight": "Cost-sensitive (class weights)",
    "random_over": "Random over-sampling", "random_under": "Random under-sampling",
    "smotenc": "SMOTE-NC", "borderline_smote": "Borderline-SMOTE", "adasyn": "ADASYN",
}


def make_sampler(strategy: str, columns, seed: int = C.SEED):
    cat = categorical_idx(columns)
    if strategy in ("none", "class_weight"):
        return None
    if strategy == "smotenc":
        return SMOTENC(categorical_features=cat, random_state=seed, k_neighbors=5)
    if strategy == "random_over":
        return RandomOverSampler(random_state=seed)
    if strategy == "random_under":
        return RandomUnderSampler(random_state=seed)
    if strategy == "borderline_smote":
        return BorderlineSMOTE(random_state=seed, k_neighbors=5)
    if strategy == "adasyn":
        return ADASYN(random_state=seed, n_neighbors=5)
    raise ValueError(strategy)


def make_pipeline(model_name: str, columns, strategy: str = "smotenc",
                  seed: int = C.SEED, params: dict | None = None) -> Pipeline:
    cw = "balanced" if strategy == "class_weight" else None
    est = clone(base_estimators(seed, class_weight=cw)[model_name])
    steps = []
    sampler = make_sampler(strategy, columns, seed)
    if sampler is not None:
        steps.append(("sampler", sampler))
        steps.append(("round", RoundCategorical(categorical_idx(columns), _cat_maxima(columns))))
    steps.append(("prep", make_preprocessor(columns)))
    steps.append(("clf", est))
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**params)
    return pipe
