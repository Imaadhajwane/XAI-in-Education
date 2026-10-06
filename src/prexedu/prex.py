"""
PREX-Edu: Pedagogical Risk EXplainability for Education.

Implements Equation (3) of the manuscript

    R_k(x) = (1/|p_k|) * sum_{j in p_k} |phi_j(x)|

plus the naive (un-normalised) variant, a signed variant, ranking, dominant
dimension + margin, attribution coverage, and natural-language explanation
generation (Algorithm 1, steps 8-12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

INTERVENTIONS = {
    "Engagement": "structured study plan / mentoring to raise weekly study time and motivation",
    "Lifestyle": "sleep-hygiene and screen-time counselling with the family",
    "Cognitive Load": "counsellor referral for stress management; review of workload",
    "Participation": "attendance monitoring and follow-up with parents / class teacher",
}


def dimension_map(include_context: bool = False) -> dict:
    d = dict(C.PREX_DIMENSIONS)
    if include_context:
        d.update(C.CONTEXT_DIMENSION)
    return d


def prex_scores(shap_df: pd.DataFrame, normalize: bool | None = None, signed: bool = False,
                include_context: bool = False) -> pd.DataFrame:
    """Per-instance pedagogical risk magnitude R_k (rows = students).

    normalize=None -> follow C.PREX_AGGREGATION ("mean" -> Eq. 3, "sum" -> naive)."""
    if normalize is None:
        normalize = C.PREX_AGGREGATION == "mean"
    out = {}
    for k, feats in dimension_map(include_context).items():
        v = shap_df[feats] if signed else shap_df[feats].abs()
        s = v.sum(axis=1)
        out[k] = s / len(feats) if normalize else s
    return pd.DataFrame(out, index=shap_df.index)


def dominant(R: pd.DataFrame) -> pd.DataFrame:
    """Dominant dimension and margin over the runner-up (Table A8)."""
    arr = R.values
    order = np.argsort(-arr, axis=1)
    top = R.columns[order[:, 0]]
    second = R.columns[order[:, 1]]
    srt = -np.sort(-arr, axis=1)
    return pd.DataFrame({"Dominant": top, "Second": second,
                         "Margin": srt[:, 0] - srt[:, 1]}, index=R.index)


def coverage(shap_df: pd.DataFrame) -> pd.Series:
    """Share of total |SHAP| mass captured by the four PREX-Edu dimensions."""
    tot = shap_df.abs().sum(axis=1)
    feats = [f for v in C.PREX_DIMENSIONS.values() for f in v]
    return shap_df[feats].abs().sum(axis=1) / tot.replace(0, np.nan)


def feature_order_from_prex(R_mean: pd.Series, shap_global: pd.Series) -> list[str]:
    """Global removal order used in faithfulness ablation: dimensions by mean
    R_k, features within a dimension by mean |SHAP|; unmapped context features
    appended at the end by mean |SHAP|."""
    order = []
    for k in R_mean.sort_values(ascending=False).index:
        feats = C.PREX_DIMENSIONS[k]
        order += sorted(feats, key=lambda f: -shap_global[f])
    rest = [f for f in shap_global.sort_values(ascending=False).index if f not in order]
    return order + rest


def instance_feature_order(shap_row: pd.Series, R_row: pd.Series) -> list[str]:
    """Per-student removal order (dimension rank, then |phi| within)."""
    order = []
    for k in R_row.sort_values(ascending=False).index:
        order += sorted(C.PREX_DIMENSIONS[k], key=lambda f: -abs(shap_row[f]))
    rest = [f for f in shap_row.abs().sort_values(ascending=False).index if f not in order]
    return order + rest


def explain_text(R_row: pd.Series, signed_row: pd.Series | None = None, prob: float | None = None,
                 top_k: int = 2) -> str:
    """Natural-language PREX-Edu explanation (Algorithm 1, step 11)."""
    top = R_row.sort_values(ascending=False).index[:top_k]
    parts = []
    if prob is not None:
        parts.append(f"Predicted at-risk probability: {prob:.0%}.")
    for i, k in enumerate(top):
        direction = ""
        if signed_row is not None:
            direction = (" (pushing risk UP)" if signed_row[k] > 0 else " (protective)")
        parts.append(f"{'Primary' if i == 0 else 'Secondary'} concern: {k}{direction} "
                     f"→ suggested action: {INTERVENTIONS[k]}.")
    return " ".join(parts)
