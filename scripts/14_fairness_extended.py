"""
14 - Extended fairness audit on all 1,208 students  (NEW)
=========================================================
The first draft's fairness analysis used a test set with only 19-20 At-Risk
students, which a reviewer will (rightly) call under-powered. Here every
student has an out-of-fold prediction (script 10), so subgroup metrics use
all 66 At-Risk students, with stratified bootstrap 95% CIs.

Metrics per subgroup: selection rate, TPR (equal opportunity), FPR,
PPV (predictive parity), ROC-AUC, calibration-in-the-large (mean predicted -
observed). Gaps vs reference group with bootstrap CIs; equalized-odds gap.

Outputs
  Table S22  subgroup metrics with 95% CIs        S22_fairness_subgroups
  Table S23  fairness gaps vs reference           S23_fairness_gaps
  Fig S19    subgroup forest plot                 FigS19_fairness_forest
"""
from _common import C, get_logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prexedu import data as D
from prexedu import plotting as P
from prexedu.tables import save_table

log = get_logger("14_fairness_ext")
df, X, y = D.load_xy()
oof = pd.read_csv(C.DATA_PROCESSED / "oof_predictions.csv", index_col=0)
long = pd.read_csv(C.DATA_PROCESSED / "oof_predictions_long.csv")
FM = C.FINAL_MODEL
p = oof.loc[df.index, FM].values
THR = float(long[long.model == FM].thr_f1.median())
pred = (p >= THR).astype(int)
log.info(f"{FM}: tuned threshold (median inner-CV F1-optimal) = {THR:.3f}")
ATTR = {"Gender": ["Female", "Male", "Prefer not to say"], "IncomeRange": ["High", "Medium", "Low"],
        "ParentEducation": ["Postgrad", "College", "High school", "No formal"]}
B = 300 if C.FAST else 2000
rng = np.random.default_rng(C.SEED)


def gm(yy, pp, pr):
    pos, neg = yy == 1, yy == 0
    return {"Selection rate": pr.mean(),
            "TPR (equal opportunity)": pr[pos].mean() if pos.any() else np.nan,
            "FPR": pr[neg].mean() if neg.any() else np.nan,
            "PPV (predictive parity)": yy[pr == 1].mean() if pr.any() else np.nan,
            "ROC-AUC": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) and yy.sum() >= 3 else np.nan,
            "Calibration-in-the-large": pp.mean() - yy.mean()}


rows, gap_rows, boots = [], [], {}
for a, levels in ATTR.items():
    g = df[a].values
    for lvl in levels:
        m = g == lvl
        idx = np.where(m)[0]
        pos_i, neg_i = idx[y[idx] == 1], idx[y[idx] == 0]
        point = gm(y[idx], p[idx], pred[idx])
        bs = []
        for _ in range(B):
            bi = np.concatenate([rng.choice(pos_i, len(pos_i)) if len(pos_i) else [], rng.choice(neg_i, len(neg_i))]).astype(int)
            bs.append(gm(y[bi], p[bi], pred[bi]))
        bs = pd.DataFrame(bs)
        boots[(a, lvl)] = bs
        row = {"Attribute": C.LABELS.get(a, a), "Subgroup": lvl, "n": int(m.sum()), "At-Risk n": int(y[m].sum()),
               "Prevalence": round(y[m].mean(), 3)}
        for k, v in point.items():
            lo, hi = np.nanpercentile(bs[k], [2.5, 97.5]) if bs[k].notna().any() else (np.nan, np.nan)
            row[k] = f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if not np.isnan(v) else "n/a"
        rows.append(row)
    ref = levels[0]
    for lvl in levels[1:]:
        r = {"Attribute": C.LABELS.get(a, a), "Comparison": f"{lvl} − {ref}"}
        for k in ["Selection rate", "TPR (equal opportunity)", "FPR", "PPV (predictive parity)"]:
            d = boots[(a, lvl)][k].values - boots[(a, ref)][k].values
            pt = gm(y[df[a].values == lvl], p[df[a].values == lvl], pred[df[a].values == lvl])[k] - \
                 gm(y[df[a].values == ref], p[df[a].values == ref], pred[df[a].values == ref])[k]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            r[f"Δ {k}"] = f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" *" if (lo > 0 or hi < 0) else "")
        sel_ref = gm(y[df[a].values == ref], p[df[a].values == ref], pred[df[a].values == ref])["Selection rate"]
        sel_l = gm(y[df[a].values == lvl], p[df[a].values == lvl], pred[df[a].values == lvl])["Selection rate"]
        r["Disparate-impact ratio"] = round(sel_l / sel_ref, 3) if sel_ref > 0 else np.nan
        tpr_d = boots[(a, lvl)]["TPR (equal opportunity)"] - boots[(a, ref)]["TPR (equal opportunity)"]
        fpr_d = boots[(a, lvl)]["FPR"] - boots[(a, ref)]["FPR"]
        r["Equalized-odds gap (max |ΔTPR|,|ΔFPR|), median [95% CI]"] = \
            "{:.3f} [{:.3f}, {:.3f}]".format(*np.nanpercentile(np.maximum(tpr_d.abs(), fpr_d.abs()), [50, 2.5, 97.5]))
        gap_rows.append(r)

save_table(pd.DataFrame(rows), "S22_fairness_subgroups",
           f"Subgroup performance of the final model on out-of-fold predictions for all {len(y)} students.",
           note=f"Threshold = {THR:.3f} (median inner-CV F1-optimal threshold). Values: point estimate [stratified bootstrap "
                f"95% CI, {B} replicates]. Calibration-in-the-large = mean predicted − observed prevalence.")
save_table(pd.DataFrame(gap_rows), "S23_fairness_gaps",
           "Fairness gaps relative to the reference subgroup (first level of each attribute).",
           note="* = 95% CI excludes 0. Disparate-impact ratio outside [0.8, 1.25] breaches the four-fifths rule. Small "
                "subgroups (e.g., 'Prefer not to say', n = 54) produce wide intervals; absence of significance is not "
                "evidence of fairness.")

# ---------------- Fig S19 forest ------------------------------------------- #
mets = ["Selection rate", "TPR (equal opportunity)", "FPR", "PPV (predictive parity)"]
fig, axes = plt.subplots(1, 4, figsize=(7.6, 4.2), sharey=True)
labels, ypos, k = [], [], 0
for a, levels in ATTR.items():
    for lvl in levels:
        labels.append(f"{C.LABELS.get(a, a)}: {lvl}")
        ypos.append(k); k += 1
    k += 0.6
for ax, met in zip(axes, mets):
    j = 0
    for a, levels in ATTR.items():
        col = {"Gender": P.OKABE_ITO[0], "IncomeRange": P.OKABE_ITO[2], "ParentEducation": P.OKABE_ITO[3]}[a]
        for lvl in levels:
            bs = boots[(a, lvl)][met]
            m = df[a].values == lvl
            pt = gm(y[m], p[m], pred[m])[met]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if bs.notna().any() else (np.nan, np.nan)
            ax.errorbar(pt, ypos[j], xerr=[[pt - lo], [hi - pt]], fmt="o", color=col, ms=4, lw=1.1, capsize=0)
            j += 1
    overall = gm(y, p, pred)[met]
    ax.axvline(overall, color=P.RISK, ls="--", lw=0.8)
    ax.set_title(met.replace(" (", "\n("), loc="left", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=7)
axes[0].set_yticks(ypos, labels, fontsize=7.5)
axes[0].invert_yaxis()
fig.suptitle(f"Subgroup fairness of {FM} (out-of-fold, n = {len(y)}; dashed = overall)", x=0.01, ha="left",
             fontweight="bold", fontsize=10)
fig.tight_layout()
P.save(fig, "FigS19_fairness_forest")
log.info("done")
