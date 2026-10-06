"""
13 - Calibration, threshold choice and decision-curve analysis  (NEW)
====================================================================
Uses the per-student out-of-fold probabilities from script 10 (n = 1,208).

Why: an early-warning system is judged by (a) whether its probabilities can
be trusted (calibration), (b) which operating threshold a school should use,
and (c) whether using the model beats "flag everyone" / "flag no one"
(net benefit; Vickers & Elkin, 2006). Reviewers in LA increasingly ask for these.

Outputs
  Table S20  calibration metrics (Brier, ECE, slope, intercept)   S20_calibration
  Table S21  operating points (recall-targeted thresholds)        S21_operating_points
  Fig S16    reliability diagrams                                 FigS16_calibration
  Fig S17    decision curves (net benefit)                        FigS17_decision_curve
  Fig S18    threshold trade-off for the final model              FigS18_threshold_tradeoff
"""
from _common import C, get_logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from prexedu import evaluation as E
from prexedu import plotting as P
from prexedu.tables import save_table

log = get_logger("13_calibration")
oof = pd.read_csv(C.DATA_PROCESSED / "oof_predictions.csv", index_col=0)
y = oof.pop("y").values
models = [m for m in C.MODEL_ORDER if m in oof.columns]

# ---------------- Table S20 ------------------------------------------------ #
rows = []
for m in models:
    p = oof[m].values
    a, b = E.calibration_slope_intercept(y, p)
    rows.append({"Model": m, "Brier": round(E.all_metrics(y, p)["brier"], 4),
                 "Brier skill score": round(1 - E.all_metrics(y, p)["brier"] / (y.mean() * (1 - y.mean())), 3),
                 "ECE (10 quantile bins)": round(E.expected_calibration_error(y, p), 4),
                 "Calibration intercept (ideal 0)": round(a, 3), "Calibration slope (ideal 1)": round(b, 3),
                 "Mean predicted": round(p.mean(), 4), "Observed prevalence": round(y.mean(), 4)})
save_table(pd.DataFrame(rows), "S20_calibration",
           "Calibration of out-of-fold predicted probabilities (nested repeated CV, n = 1,208).",
           note="Slope < 1 indicates over-confident (too extreme) probabilities; intercept ≠ 0 indicates systematic over/under-"
                "estimation (calibration-in-the-large).")

# ---------------- Fig S16 reliability -------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax = axes[0]
ax.plot([0, 1], [0, 1], color=P.MUTED, ls=":", lw=0.9, label="Perfect calibration")
for m in models:
    fr, mp = calibration_curve(y, oof[m], n_bins=10, strategy="quantile")
    ax.plot(mp, fr, marker=P.MODEL_MARK[m], ms=3.5, color=P.MODEL_COLORS[m], ls=P.MODEL_LS[m], label=C.MODEL_SHORT[m])
ax.set(xlabel="Mean predicted probability", ylabel="Observed At-Risk fraction", xlim=(0, 1), ylim=(0, 1))
ax.legend(fontsize=7, loc="upper left")
ax.set_title("(a) Reliability diagram (quantile bins)", loc="left")
ax = axes[1]
fm = C.FINAL_MODEL
ax.hist(oof.loc[y == 0, fm], bins=40, color="#9ca3af", alpha=0.8, label="Not At-Risk", log=True)
ax.hist(oof.loc[y == 1, fm], bins=40, color=P.RISK, alpha=0.8, label="At-Risk", log=True)
ax.set(xlabel=f"Predicted probability ({C.MODEL_SHORT[fm]})", ylabel="Students (log scale)")
ax.legend(fontsize=7)
ax.set_title("(b) Score distribution by class", loc="left")
fig.tight_layout()
P.save(fig, "FigS16_calibration")

# ---------------- Fig S17 decision curve ----------------------------------- #
th = np.linspace(0.01, 0.6, 120)
fig, ax = plt.subplots(figsize=(5.4, 3.3))
nb_all = y.mean() - (1 - y.mean()) * th / (1 - th)
ax.plot(th, nb_all, color=P.MUTED, ls="--", lw=1, label="Flag all students")
ax.axhline(0, color=P.INK, lw=0.8, label="Flag none")
for m in models:
    ax.plot(th, E.net_benefit(y, oof[m].values, th), color=P.MODEL_COLORS[m], ls=P.MODEL_LS[m], label=C.MODEL_SHORT[m])
ax.set(xlabel="Threshold probability (risk at which a teacher would intervene)", ylabel="Net benefit",
       ylim=(-0.01, y.mean() * 1.08), xlim=(0, 0.6))
ax.legend(fontsize=7, ncol=2)
ax.set_title("Decision-curve analysis (out-of-fold, n = 1,208)", loc="left")
P.save(fig, "FigS17_decision_curve")

# ---------------- Operating points ----------------------------------------- #
rows = []
for m in models:
    p = oof[m].values
    pr, rc, t = precision_recall_curve(y, p)
    for target in [0.95, 0.90, 0.80, 0.70]:
        ok = np.where(rc[:-1] >= target)[0]
        j = ok[-1] if len(ok) else 0              # highest threshold achieving the target recall
        thr = t[j]
        mm = E.all_metrics(y, p, thr)
        rows.append({"Model": m, "Target recall": f"≥ {target:.0%}", "Threshold": round(thr, 4),
                     "Recall": round(mm["recall"], 3), "Precision": round(mm["precision"], 3),
                     "Specificity": round(mm["specificity"], 3), "Students flagged": int(mm["tp"] + mm["fp"]),
                     "Flagged per true At-Risk": round((mm["tp"] + mm["fp"]) / max(1, mm["tp"]), 2),
                     "Missed At-Risk": int(mm["fn"]),
                     "Net benefit": round(E.net_benefit(y, p, [thr])[0], 4)})
save_table(pd.DataFrame(rows), "S21_operating_points",
           "Operating points: thresholds needed to reach target sensitivity, and the resulting workload.",
           note="Computed on out-of-fold probabilities for all 1,208 students (66 At-Risk). 'Flagged per true At-Risk' is "
                "the number of students a school must follow up per correctly identified At-Risk student. Note these "
                "thresholds are chosen post hoc on the pooled OOF predictions — use for planning, not as unbiased estimates.")

# ---------------- Fig S18 threshold trade-off ------------------------------ #
p = oof[fm].values
ths = np.linspace(0.01, 0.9, 200)
rec, prec, spec, flag = [], [], [], []
for t_ in ths:
    mm = E.all_metrics(y, p, t_)
    rec.append(mm["recall"]); prec.append(mm["precision"]); spec.append(mm["specificity"])
    flag.append((mm["tp"] + mm["fp"]) / len(y))
fig, ax = plt.subplots(figsize=(5.4, 3.2))
for vals, lab, col, ls in [(rec, "Recall (sensitivity)", P.OKABE_ITO[0], "-"), (prec, "Precision (PPV)", P.OKABE_ITO[1], "--"),
                           (spec, "Specificity", P.OKABE_ITO[2], "-."), (flag, "Share of students flagged", P.MUTED, ":")]:
    ax.plot(ths, vals, color=col, ls=ls, label=lab)
ax.axvline(0.5, color=P.INK, lw=0.7, ls=":")
ax.set(xlabel="Decision threshold", ylabel="Value", ylim=(0, 1.02))
ax.legend(fontsize=7, loc="center right")
ax.set_title(f"Threshold trade-off — {fm} (out-of-fold)", loc="left")
P.save(fig, "FigS18_threshold_tradeoff")
log.info("done")
