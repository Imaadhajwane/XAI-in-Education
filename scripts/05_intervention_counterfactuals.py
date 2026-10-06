"""
05 - Actionability: intervention simulation & counterfactuals  (EXTENDED)
=========================================================================
Question: if a teacher acts on the dimension PREX-Edu ranks first, does the
model's predicted risk fall more than if they act on another dimension?

A. Intervention simulation (per predicted At-Risk student)
   For each dimension k, "improve" only its features by a standardised dose:
   +1 training SD in the beneficial direction (Likert features +/-1 point,
   attendance -> Yes), clipped to plausible bounds. Record Δ P(At-Risk).
   -> agreement between PREX top-ranked dimension and the most effective
      dimension, compared with chance (25%) and with naive aggregation.
B. Counterfactual explanations
   Greedy minimal-cost search over actionable features only (sociodemographic
   features are immutable) for the change that brings P(At-Risk) < 0.5.

Outputs
  Table S11 intervention simulation summary          S11_intervention_simulation
  Table S12 per-student intervention effects          S12_intervention_per_student
  Table S13 counterfactual summary                    S13_counterfactual_summary
  Table S14 counterfactual examples                   S14_counterfactual_examples
  Fig S11   Δ risk by PREX rank of targeted dimension FigS11_intervention_by_rank
  Fig S12   counterfactual feature-change frequency   FigS12_counterfactual_changes
"""
from _common import C, get_logger

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from prexedu import data as D
from prexedu import plotting as P
from prexedu import prex
from prexedu.tables import save_table

log = get_logger("05_actionability")
FM = C.FINAL_MODEL
df, X, y = D.load_xy()
split = joblib.load(C.MODEL_DIR / "holdout_split.joblib")
Xtr, Xte = X.loc[split["train_idx"]], X.loc[split["test_idx"]]
yte = y[split["test_idx"]]
pipe = joblib.load(C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[FM]}.joblib")
S = pd.read_csv(C.MODEL_DIR / f"shap_test_{C.MODEL_SHORT[FM]}.csv", index_col=0)
prob = pipe.predict_proba(Xte)[:, 1]
R, Rn = prex.prex_scores(S, normalize=True), prex.prex_scores(S, normalize=False)   # mean (Eq. 3 variant) vs sum
RP = Rn if C.PREX_AGGREGATION == "sum" else R                                      # PREX-Edu operator
SD = Xtr.std()
dims = list(C.PREX_DIMENSIONS)


def improve(x: pd.Series, feats, dose=1.0):
    """Move each feature `dose` SDs in its beneficial direction (clipped).
    Binary attendance moves to Yes with probability-free linear interpolation
    (dose >= 1 -> Yes; fractional dose -> partial value, evaluated by the model
    as an expected effect)."""
    x = x.copy()
    for f in feats:
        sgn, lo, hi = C.ACTIONABLE[f]
        if f == "HighAttendance":
            x[f] = min(1.0, x[f] + dose)
        else:
            x[f] = np.clip(x[f] + sgn * dose * SD[f], lo, hi)
    return x


# Two intervention designs:
#  "full"  : every feature of the dimension gets 1 SD  (total effort ∝ |p_k|)
#  "equal" : the dimension gets a fixed budget of 1 SD-unit, split equally
#            over its |p_k| features (total effort identical for every dimension)
DESIGNS = {"full": lambda k: 1.0, "equal": lambda k: 1.0 / len(C.PREX_DIMENSIONS[k])}


from prexedu import xai
THR = xai.decision_threshold(FM)
at = np.where(prob >= THR)[0]
log.info(f"{len(at)} flagged At-Risk test students (p >= {THR:.3f})")

# ---------------- A. Intervention simulation ------------------------------- #
summary_rows, by_rank_all, per_student = [], {}, []
for design, dose_of in DESIGNS.items():
    rows = []
    for i in at:
        x = Xte.iloc[i]
        p0 = prob[i]
        eff = {}
        for k in dims:
            xi = improve(x, C.PREX_DIMENSIONS[k], dose_of(k))
            eff[k] = p0 - pipe.predict_proba(xi.to_frame().T.astype(float))[:, 1][0]
        Rp = Rn if C.PREX_AGGREGATION == "sum" else R
        rank_norm = Rp.iloc[i].rank(ascending=False, method="first")
        best = max(eff, key=eff.get)
        row = {"Design": design, "Student ID": int(Xte.index[i]) + 1, "P0": round(p0, 3),
               "Top (mean)": R.iloc[i].idxmax(), "Top (sum)": Rn.iloc[i].idxmax(),
               "Most effective": best, "ΔP targeting PREX top": round(eff[Rp.iloc[i].idxmax()], 4)}
        for k in dims:
            row[f"ΔP {k}"] = round(eff[k], 4)
            row[f"rank {k}"] = int(rank_norm[k])
        rows.append(row)
    iv = pd.DataFrame(rows)
    per_student.append(iv)
    n = len(iv)
    a_norm = (iv["Top (mean)"] == iv["Most effective"]).mean()
    a_naive = (iv["Top (sum)"] == iv["Most effective"]).mean()
    a_prex = a_naive if C.PREX_AGGREGATION == "sum" else a_norm
    b_norm = stats.binomtest(int(round(a_prex * n)), n, 0.25, alternative="greater")
    by_rank = {r: [iv.loc[j, f"ΔP {k}"] for j in iv.index for k in dims if iv.loc[j, f"rank {k}"] == r]
               for r in [1, 2, 3, 4]}
    by_rank_all[design] = by_rank
    fr = stats.friedmanchisquare(*[np.array(by_rank[r]) for r in [1, 2, 3, 4]])
    w_rand = stats.wilcoxon(iv["ΔP targeting PREX top"], iv[[f"ΔP {k}" for k in dims]].mean(axis=1))
    label = {"full": "Full dose (1 SD per feature)", "equal": "Equal budget (1 SD-unit per dimension)"}[design]
    summary_rows += [
        {"Design": label, "Measure": "Flagged At-Risk students", "Value": n},
        {"Design": label, "Measure": "Mean-|SHAP| top = most effective", "Value": f"{a_norm:.1%}"},
        {"Design": label, "Measure": "Sum-|SHAP| top = most effective", "Value": f"{a_naive:.1%}"},
        {"Design": label, "Measure": "Binomial test vs chance 25% (PREX-Edu operator)", "Value": f"p = {b_norm.pvalue:.2e}"},
        {"Design": label, "Measure": "Mean ΔP rank 1 / 2 / 3 / 4",
         "Value": " / ".join(f"{np.mean(by_rank[r]):.4f}" for r in [1, 2, 3, 4])},
        {"Design": label, "Measure": "Mean ΔP random dimension",
         "Value": f"{iv[[f'ΔP {k}' for k in dims]].values.mean():.4f}"},
        {"Design": label, "Measure": "Wilcoxon rank-1 vs random dimension",
         "Value": f"W = {w_rand.statistic:.1f}, p = {w_rand.pvalue:.2e}"},
        {"Design": label, "Measure": "Friedman across ranks 1–4", "Value": f"χ² = {fr.statistic:.2f}, p = {fr.pvalue:.2e}"},
    ]
save_table(pd.concat(per_student, ignore_index=True), "S12_intervention_per_student",
           "Simulated intervention effect per dimension for each flagged At-Risk test student.",
           note="ΔP = reduction in predicted At-Risk probability after improving only that dimension's features.")
save_table(pd.DataFrame(summary_rows), "S11_intervention_simulation",
           "Intervention simulation: does acting on the top-ranked PREX-Edu dimension produce the largest risk reduction?",
           note="Full dose: each feature of the targeted dimension improves by 1 SD, so two-feature dimensions receive twice "
                "the total effort. Equal budget: every dimension receives the same total effort (1 SD-unit split across its "
                "features). Ranks 1–4 and the random comparison use the PREX-Edu operator set in config. This is a computational "
                "coherence check, not a causal estimate.")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, design in zip(axes, DESIGNS):
    data = [by_rank_all[design][r] for r in [1, 2, 3, 4]]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=P.INK))
    for patch, col in zip(bp["boxes"], [P.OKABE_ITO[0], P.OKABE_ITO[5], "#9ca3af", "#d1d5db"]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    for r, d_ in enumerate(data):
        ax.scatter(np.full(len(d_), r + 1) + np.random.default_rng(r).uniform(-0.1, 0.1, len(d_)), d_, s=8,
                   color=P.INK, alpha=0.5, zorder=3)
    ax.set_xticks([1, 2, 3, 4], ["Rank 1\n(PREX top)", "Rank 2", "Rank 3", "Rank 4"])
    ax.set_title("(a) Full dose per feature" if design == "full" else "(b) Equal budget per dimension", loc="left")
axes[0].set_ylabel("Reduction in P(At-Risk)")
fig.suptitle(f"Simulated intervention effect by PREX-Edu rank (n = {len(at)} flagged At-Risk)", x=0.01,
             ha="left", fontweight="bold", fontsize=10)
fig.tight_layout()
P.save(fig, "FigS11_intervention_by_rank")

# ---------------- B. Counterfactuals --------------------------------------- #
STEP = {f: (1.0 if f in ("Motivation(1-5)", "Stress(1-5)", "HighAttendance") else 0.25 * SD[f]) for f in C.ACTIONABLE}
COST = {f: STEP[f] / (SD[f] if f not in ("HighAttendance",) else 1.0) for f in C.ACTIONABLE}
COST["HighAttendance"] = 1.0
FEAT_DIM = {f: k for k, v in C.PREX_DIMENSIONS.items() for f in v}


def counterfactual(x: pd.Series, target=None, max_steps=60):
    target = THR if target is None else target
    x = x.copy().astype(float)
    p = pipe.predict_proba(x.to_frame().T)[:, 1][0]
    path, cost = [], 0.0
    for _ in range(max_steps):
        if p < target:
            break
        best, best_gain, best_x = None, 0, None
        for f, (sgn, lo, hi) in C.ACTIONABLE.items():
            xn = x.copy()
            xn[f] = np.clip(x[f] + sgn * STEP[f], lo, hi)
            if xn[f] == x[f]:
                continue
            pn = pipe.predict_proba(xn.to_frame().T)[:, 1][0]
            gain = (p - pn) / COST[f]
            if gain > best_gain:
                best, best_gain, best_x, best_p = f, gain, xn, pn
        if best is None:
            break
        x, p = best_x, best_p
        cost += COST[best]
        path.append(best)
    return x, p, cost, path


rows, ex_rows = [], []
for i in at:
    x0 = Xte.iloc[i].astype(float)
    xc, pc, cost, path = counterfactual(x0)
    changed = [f for f in C.ACTIONABLE if not np.isclose(xc[f], x0[f])]
    dim_cost = pd.Series({k: sum(COST[f] for f in path if FEAT_DIM[f] == k) for k in dims})
    rows.append({"Student ID": int(Xte.index[i]) + 1, "P0": prob[i], "P_cf": pc, "Valid (P_cf < thr)": pc < THR,
                 "Cost (SD units)": cost, "Features changed": len(changed),
                 "Main CF dimension": dim_cost.idxmax() if cost > 0 else "—", "PREX top": RP.iloc[i].idxmax(),
                 **{f"chg {f}": xc[f] - x0[f] for f in C.ACTIONABLE}})
cf = pd.DataFrame(rows)
valid = cf[cf["Valid (P_cf < thr)"]]
s13 = pd.DataFrame({"Measure": ["Students", "Valid counterfactuals found", "Median cost (SD units)",
                                "Median number of features changed", "Main CF dimension = PREX top dimension"],
                    "Value": [len(cf), f"{len(valid)} ({len(valid) / max(1, len(cf)):.0%})",
                              round(valid["Cost (SD units)"].median(), 2) if len(valid) else "n/a",
                              valid["Features changed"].median() if len(valid) else "n/a",
                              f"{(valid['Main CF dimension'] == valid['PREX top']).mean():.1%}" if len(valid) else "n/a"]})
save_table(s13, "S13_counterfactual_summary",
           "Actionable counterfactual explanations for flagged At-Risk students (greedy minimal-cost search).",
           note=f"Target: P(At-Risk) below the decision threshold {THR:.3f}. Only the six actionable features may change; demographics are immutable. Step sizes: 0.25 SD for continuous "
                "features, 1 point for Likert items, Yes for attendance. Cost = sum of standardised step sizes.")
ex = cf.head(10).copy()
for f in C.ACTIONABLE:
    ex[f"Δ {C.LABELS[f]}"] = ex.pop(f"chg {f}").round(2)
ex[["P0", "P_cf", "Cost (SD units)"]] = ex[["P0", "P_cf", "Cost (SD units)"]].round(3)
save_table(ex, "S14_counterfactual_examples", "Counterfactual examples (first 10 flagged At-Risk test students).",
           note=f"Δ columns: change required in each actionable feature to bring predicted risk below {THR:.3f}.")

fig, ax = plt.subplots(figsize=(4.8, 2.9))
freq = pd.Series({C.LABELS[f]: (np.abs(valid[f"chg {f}"]) > 1e-9).mean() for f in C.ACTIONABLE}).sort_values()
ax.barh(range(len(freq)), freq.values, color=[P.DIM_COLORS[FEAT_DIM[f]] for f in
                                                 sorted(C.ACTIONABLE, key=lambda f: freq[C.LABELS[f]])], height=0.6)
ax.set_yticks(range(len(freq)), freq.index)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Share of counterfactuals changing the feature")
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=P.DIM_COLORS[k], label=k) for k in dims], fontsize=7, loc="lower right")
ax.set_title(f"Which actionable features flip the prediction? (n = {len(valid)})", loc="left")
P.save(fig, "FigS12_counterfactual_changes")
log.info("done")
