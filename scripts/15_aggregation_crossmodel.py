"""
15 - Which aggregation operator should PREX-Edu use? + cross-model robustness  (NEW)
===================================================================================
A. Aggregation operators compared (per student, four PREX dimensions):
     mean|φ|   (Eq. 3, cardinality-normalised - the draft's choice)
     sum|φ|    (naive)
     |Σφ|      (absolute signed sum = |group SHAP|; exact group attribution
                for additive models)
     RMS(φ)    (root-mean-square, a compromise)
   Ground truths they are scored against (predicted At-Risk test students):
     G1  exact BASELINE GROUP-SHAPLEY over the 5 groups (4 dims + context),
         computed by enumerating all 32 coalitions on the log-odds scale
     G2  dimension ablation: Δlogit when the whole dimension is reset
     G3  intervention effect under an equal 1-SD-unit budget per dimension
   -> top-1 agreement and Kendall τ for each operator.
B. Cross-model consistency: PREX-Edu profiles from LR vs XGBoost, Random
   Forest and Gradient Boosting on the same test students.

Outputs
  Table S24  operator comparison                S24_aggregation_operators
  Table S25  cross-model PREX consistency       S25_crossmodel_prex
  Fig S20    operator agreement                 FigS20_aggregation_operators
  Fig S21    dimension shares by model          FigS21_crossmodel_dimensions
"""
from _common import C, get_logger

import itertools
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from prexedu import data as D
from prexedu import plotting as P
from prexedu import prex, xai
from prexedu.tables import save_table

log = get_logger("15_aggregation")
FM = C.FINAL_MODEL
df, X, y = D.load_xy()
split = joblib.load(C.MODEL_DIR / "holdout_split.joblib")
Xtr, Xte = X.loc[split["train_idx"]], X.loc[split["test_idx"]]
yte = y[split["test_idx"]]
pipe = joblib.load(C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[FM]}.joblib")
S = pd.read_csv(C.MODEL_DIR / f"shap_test_{C.MODEL_SHORT[FM]}.csv", index_col=0)
prob = pipe.predict_proba(Xte)[:, 1]
ref = xai.reference_values(Xtr)
dims = list(C.PREX_DIMENSIONS)
groups = prex.dimension_map(include_context=True)
gnames = list(groups)
SD = Xtr.std()
logit = lambda p_: np.log(np.clip(p_, 1e-9, 1 - 1e-9) / (1 - np.clip(p_, 1e-9, 1 - 1e-9)))


def operators(srow):
    out = {}
    for k in dims:
        v = srow[C.PREX_DIMENSIONS[k]].values
        out.setdefault("mean|φ| (cardinality-normalised)", {})[k] = np.abs(v).mean()
        out.setdefault("sum|φ| (PREX-Edu)", {})[k] = np.abs(v).sum()
        out.setdefault("|Σφ| (group SHAP)", {})[k] = abs(v.sum())
        out.setdefault("RMS(φ)", {})[k] = np.sqrt((v ** 2).mean())
    return {k: pd.Series(v) for k, v in out.items()}


def group_shapley(model, x: pd.Series):
    """Exact baseline Shapley over groups (log-odds), reference = train mean/mode."""
    G = len(gnames)
    coal = list(itertools.product([0, 1], repeat=G))
    rows = []
    for c in coal:
        z = ref.copy()
        for g, on in zip(gnames, c):
            if on:
                for f in groups[g]:
                    z[f] = x[f]
        rows.append(z)
    v = dict(zip(coal, logit(model.predict_proba(pd.DataFrame(rows)[X.columns].astype(float))[:, 1])))
    phi = {}
    for i, g in enumerate(gnames):
        tot = 0.0
        for c in coal:
            if c[i] == 1:
                continue
            s = sum(c)
            w = math.factorial(s) * math.factorial(G - s - 1) / math.factorial(G)
            c1 = list(c); c1[i] = 1
            tot += w * (v[tuple(c1)] - v[c])
        phi[g] = tot
    return pd.Series(phi)


THR = xai.decision_threshold(FM)
at = np.where(prob >= THR)[0]
recs = {op: {"G1 top-1": [], "G1 τ": [], "G2 top-1": [], "G2 τ": [], "G3 top-1": [], "G3 τ": []}
        for op in operators(S.iloc[0])}
gs_rows = []
for i in at:
    x = Xte.iloc[i]
    gsh = group_shapley(pipe, x)[dims]
    gs_rows.append(gsh)
    base_l = logit(prob[i])
    abl = pd.Series({k: base_l - logit(xai.masked_predict(pipe, x.to_frame().T, C.PREX_DIMENSIONS[k], ref)[0])
                     for k in dims})
    inter = {}
    for k in dims:
        xi = x.copy()
        for f in C.PREX_DIMENSIONS[k]:
            sgn, lo, hi = C.ACTIONABLE[f]
            dose = 1.0 / len(C.PREX_DIMENSIONS[k])
            xi[f] = min(1.0, xi[f] + dose) if f == "HighAttendance" else np.clip(xi[f] + sgn * dose * SD[f], lo, hi)
        inter[k] = prob[i] - pipe.predict_proba(xi.to_frame().T.astype(float))[:, 1][0]
    inter = pd.Series(inter)
    truths = {"G1": gsh.abs(), "G2": abl, "G3": inter}
    for op, sc in operators(S.iloc[i]).items():
        for gname, tr in truths.items():
            recs[op][f"{gname} top-1"].append(sc.idxmax() == tr.idxmax())
            recs[op][f"{gname} τ"].append(stats.kendalltau(sc.values, tr.values)[0])
rows = []
for op, r in recs.items():
    row = {"Operator": op, "Students": len(at)}
    for gname, lab in [("G1", "exact group-Shapley"), ("G2", "dimension ablation"), ("G3", "equal-budget intervention")]:
        row[f"Top-1 = {lab}"] = f"{np.mean(r[f'{gname} top-1']):.1%}"
        row[f"Mean τ vs {lab}"] = round(np.nanmean(r[f"{gname} τ"]), 3)
    rows.append(row)
s24 = pd.DataFrame(rows)
save_table(s24, "S24_aggregation_operators",
           "Empirical comparison of SHAP-to-dimension aggregation operators against three model-based ground truths.",
           note=f"Students = test students flagged At-Risk by the final model (p ≥ {THR:.3f}). G1: exact baseline Shapley value of each "
                "dimension treated as a single player (5 players incl. context; 32 coalitions; log-odds; reference = "
                "training mean/mode). G2: change in log-odds when the dimension is reset to the reference. G3: reduction in "
                "P(At-Risk) for an equal 1-SD-unit improvement budget per dimension. The operator with the highest agreement "
                "should be adopted as the PREX-Edu aggregation rule (or reported as a sensitivity analysis).")

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
ops = list(recs)
opcol = [P.OKABE_ITO[0], P.OKABE_ITO[1], P.OKABE_ITO[2], P.OKABE_ITO[3]]
for ax, (gname, lab) in zip(axes, [("G1", "vs exact group-Shapley"), ("G2", "vs dimension ablation"),
                                   ("G3", "vs equal-budget intervention")]):
    vals = [np.mean(recs[o][f"{gname} top-1"]) for o in ops]
    ax.barh(range(len(ops)), vals, color=opcol, height=0.6)
    for k_, v in enumerate(vals):
        ax.text(v + 0.01, k_, f"{v:.0%}", va="center", fontsize=7.5)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(lab, loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(ops)), ops, fontsize=8)
axes[0].invert_yaxis()
fig.supxlabel("Top-1 dimension agreement", fontsize=8.5)
fig.suptitle(f"Which aggregation operator recovers the model's dimension ranking? (n = {len(at)})", x=0.01,
             ha="left", fontweight="bold", fontsize=10)
fig.tight_layout()
P.save(fig, "FigS20_aggregation_operators")

# =========================================================================== #
# B. cross-model consistency
# =========================================================================== #
others = ["XGBoost", "Random Forest", "Gradient Boosting"]
R_lr = prex.prex_scores(S)
rows, shares = [], {FM: (R_lr.div(R_lr.sum(axis=1), axis=0)).mean()}
for m in others:
    pm = joblib.load(C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[m]}.joblib")
    Sm, _, space = xai.compute_shap(pm, Xtr, Xte, m, n_bg=200)
    Sm.index = Xte.index
    Rm = prex.prex_scores(Sm)
    shares[m] = (Rm.div(Rm.sum(axis=1).replace(0, np.nan), axis=0)).mean()
    taus = [stats.kendalltau(R_lr.iloc[i].values, Rm.iloc[i].values)[0] for i in range(len(Rm))]
    agree_all = (R_lr.idxmax(axis=1) == Rm.idxmax(axis=1)).mean()
    agree_at = (R_lr.idxmax(axis=1).values[prob >= THR] == Rm.idxmax(axis=1).values[prob >= THR]).mean()
    g_rho = stats.spearmanr(S.abs().mean(), Sm.abs().mean())[0]
    rows.append({"Comparison": f"{C.MODEL_SHORT[FM]} vs {C.MODEL_SHORT[m]}", "SHAP output space": space,
                 "Global feature-rank ρ": round(g_rho, 3),
                 "Mean per-student τ (dimension ranks)": round(np.nanmean(taus), 3),
                 "Top-1 dimension agreement (all test)": f"{agree_all:.1%}",
                 "Top-1 agreement (flagged At-Risk)": f"{agree_at:.1%}",
                 "Global dimension order": " > ".join(Rm.mean().sort_values(ascending=False).index)})
rows.insert(0, {"Comparison": f"{C.MODEL_SHORT[FM]} (reference)", "SHAP output space": "log-odds",
                "Global feature-rank ρ": 1.0, "Mean per-student τ (dimension ranks)": 1.0,
                "Top-1 dimension agreement (all test)": "100%", "Top-1 agreement (flagged At-Risk)": "100%",
                "Global dimension order": " > ".join(R_lr.mean().sort_values(ascending=False).index)})
save_table(pd.DataFrame(rows), "S25_crossmodel_prex",
           "Robustness of PREX-Edu explanations to the choice of predictive model.",
           note="Same 363 test students. High agreement means the pedagogical message does not hinge on the classifier.")

fig, ax = plt.subplots(figsize=(5.6, 2.9))
mlist = [FM] + others
left = np.zeros(len(mlist))
for k in dims:
    v = np.array([shares[m][k] for m in mlist])
    ax.barh(range(len(mlist)), v, left=left, color=P.DIM_COLORS[k], height=0.6, label=k,
            edgecolor="white", linewidth=1.5)
    for j, (l_, vv) in enumerate(zip(left, v)):
        if vv > 0.07:
            ax.text(l_ + vv / 2, j, f"{vv:.0%}", ha="center", va="center", fontsize=7.5, color="white")
    left += v
ax.set_yticks(range(len(mlist)), [C.MODEL_SHORT[m] for m in mlist])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Mean share of a student's PREX-Edu risk profile")
ax.grid(False)
ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
ax.set_title("PREX-Edu dimension shares across four classifiers", loc="left")
P.save(fig, "FigS21_crossmodel_dimensions")
log.info("done")
