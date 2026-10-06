"""
Data loading, validation, target construction and encoding.

Design
------
All downstream code works on a *base matrix* ``X`` with exactly the 13
original predictors as columns, every column numeric:

* numeric features      -> as recorded
* binary  (Yes/No)      -> 1/0
* ordinal categories    -> 0..k-1 in their natural order
* nominal categories    -> integer codes 0..k-1 (one-hot encoded *inside* the
                           model pipeline, see ``make_preprocessor``)

Keeping one column per original feature means SHAP, LIME, PDP, permutation
importance, ablation and PREX-Edu all speak the same 13-feature language,
while the models still receive a statistically correct one-hot encoding for
nominal variables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from . import config as C


# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    """Read the primary dataset exactly as collected (xlsx or csv; see PREX_DATA)."""
    if not C.DATA_RAW.exists():
        raise FileNotFoundError(
            f"{C.DATA_RAW} not found. The real survey data are available from the authors on request; "
            "to run the pipeline on the synthetic replica set "
            "PREX_DATA=data/synthetic/PREX-Edu_SYNTHETIC_n5000.csv")
    if C.DATA_RAW.suffix.lower() == ".csv":
        return pd.read_csv(C.DATA_RAW)
    return pd.read_excel(C.DATA_RAW)


def data_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level audit: type, missing, unique, range. Saved as a table."""
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "Variable": c,
            "Type": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical",
            "Missing (n)": int(s.isna().sum()),
            "Missing (%)": round(100 * s.isna().mean(), 2),
            "Unique": int(s.nunique()),
            "Min": s.min() if pd.api.types.is_numeric_dtype(s) else "",
            "Max": s.max() if pd.api.types.is_numeric_dtype(s) else "",
            "Levels": "" if pd.api.types.is_numeric_dtype(s) else
                      "; ".join(map(str, s.dropna().unique()[:8])),
        })
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame, threshold: float = C.RISK_THRESHOLD) -> pd.DataFrame:
    """Impute, normalise labels and build the binary target."""
    df = df.copy()
    # 124 records have no DifficultSubject (respondent left it blank). Treat
    # as an explicit category rather than imputing a subject.
    df["DifficultSubject"] = df["DifficultSubject"].fillna("Unknown")
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    if C.TARGET_SOURCE in df.columns:
        df[C.TARGET] = (df[C.TARGET_SOURCE] <= threshold).astype(int)
    elif C.TARGET not in df.columns:
        raise KeyError(f"Need either {C.TARGET_SOURCE} or a binary {C.TARGET} column")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 13-column numeric base matrix (see module docstring)."""
    X = pd.DataFrame(index=df.index)
    for c in C.FEATURES:
        if c in C.NUMERIC:
            X[c] = df[c].astype(float)
        elif c in C.BINARY:
            X[c] = (df[c] == "Yes").astype(float)
        elif c in C.ORDINAL:
            X[c] = df[c].map({v: i for i, v in enumerate(C.ORDINAL[c])}).astype(float)
        elif c in C.NOMINAL:
            X[c] = df[c].map({v: i for i, v in enumerate(C.NOMINAL[c])}).astype(float)
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Unmapped category values in {bad}")
    return X


def load_xy(threshold: float = C.RISK_THRESHOLD):
    """Convenience: (clean dataframe, X base matrix, y)."""
    df = clean(load_raw(), threshold)
    return df, encode(df), df[C.TARGET].values


def holdout_split(X, y, df=None, seed: int = C.SEED):
    """Legacy stratified 70:30 split (kept for continuity with the 1st draft)."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y, random_state=seed)
    out = (X.iloc[tr], X.iloc[te], y[tr], y[te])
    if df is not None:
        out = out + (df.iloc[tr], df.iloc[te])
    return out


# --------------------------------------------------------------------------- #
def nominal_idx(columns) -> list[int]:
    return [list(columns).index(c) for c in C.NOMINAL]


def categorical_idx(columns) -> list[int]:
    return [list(columns).index(c) for c in C.CATEGORICAL]


def make_preprocessor(columns, scale: bool = True) -> ColumnTransformer:
    """One-hot nominal codes; Min-Max scale the rest (as in the manuscript)."""
    cols = list(columns)
    nom = [c for c in cols if c in C.NOMINAL]
    rest = [c for c in cols if c not in C.NOMINAL]
    cats = [list(range(len(C.NOMINAL[c]))) for c in nom]
    return ColumnTransformer(
        [("nom", OneHotEncoder(categories=cats, handle_unknown="ignore",
                               sparse_output=False, drop=None), nom),
         ("num", MinMaxScaler() if scale else "passthrough", rest)],
        verbose_feature_names_out=False,
    )


def transformed_groups(preprocessor: ColumnTransformer) -> dict[str, list[int]]:
    """Map each original feature to its column indices after preprocessing.

    Used to collapse one-hot SHAP values back onto the original feature
    (valid because Shapley values are additive)."""
    names = list(preprocessor.get_feature_names_out())
    groups: dict[str, list[int]] = {}
    for j, n in enumerate(names):
        base = n
        for c in C.NOMINAL:
            if n.startswith(c + "_"):
                base = c
        groups.setdefault(base, []).append(j)
    return groups


def decode_value(feature: str, v: float) -> str:
    """Human-readable value for explanation cards."""
    if feature in C.BINARY:
        return "Yes" if v >= 0.5 else "No"
    if feature in C.ORDINAL:
        return C.ORDINAL[feature][int(round(v))]
    if feature in C.NOMINAL:
        return C.NOMINAL[feature][int(round(v))]
    return f"{v:g}"
