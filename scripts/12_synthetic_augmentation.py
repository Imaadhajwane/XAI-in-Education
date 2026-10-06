"""
12 - Synthetic data: fidelity, privacy, utility and augmentation  (NEW)
=======================================================================
Goal: use generative models to (i) test whether synthetic minority / full-
distribution augmentation helps At-Risk detection and (ii) release a
privacy-preserving synthetic copy of the primary dataset for open science -
WITHOUT ever inflating the evaluation sample.

Rules that keep this defensible for reviewers
  1. Every reported performance number is computed on REAL students only.
  2. Generators are fitted on training data only (inside each CV fold for
     the augmentation experiment), so no information leaks from test folds.
  3. Synthetic data quality is reported (fidelity + privacy) before use.
  4. Synthetic records are never described as additional participants.

Generators (class-conditional: one model per class, sampled in the desired
class mix - avoids the minority collapse seen when fitting jointly)
  * Gaussian Copula   (SDV)
  * TVAE              (SDV, variational auto-encoder)
  * CTGAN             (SDV, conditional GAN) - fidelity/privacy audit only by
                       default (CPU-expensive); set PREX_CTGAN_CV=1 to include
                       it in the CV experiment.
  * SMOTE-NC          (interpolation baseline)

Outputs
  Table S18  fidelity & privacy of each generator      S18_synthetic_fidelity_privacy
  Table S19  augmentation utility under repeated CV    S19_augmentation_utility
  Fig S14    fidelity diagnostics                      FigS14_synthetic_fidelity
  Fig S15    augmentation effect vs baseline           FigS15_augmentation_effect
  Data       data/synthetic/PREX-Edu_SYNTHETIC_n5000.csv  + DATASHEET.md
"""
from _common import C, Timer, get_logger

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from prexedu import data as D
from prexedu import evaluation as E
from prexedu import models as M
from prexedu import plotting as P
from prexedu.tables import mean_sd, save_table

warnings.filterwarnings("ignore")
log = get_logger("12_synthetic")
df, X, y = D.load_xy()
split = joblib.load(C.MODEL_DIR / "holdout_split.joblib")
EPOCHS = 50 if C.FAST else 300
CTGAN_CV = os.environ.get("PREX_CTGAN_CV", "0") == "1"
INT_COLS = ["Age", "Motivation(1-5)", "Stress(1-5)"] + list(C.ORDINAL) + list(C.NOMINAL) + C.BINARY


# --------------------------------------------------------------------------- #
def make_metadata(frame):
    from sdv.metadata import Metadata
    md = Metadata.detect_from_dataframe(frame)
    for c in frame.columns:
        if c in C.CATEGORICAL or c in ("Motivation(1-5)", "Stress(1-5)", "AtRisk"):
            md.update_column(column_name=c, sdtype="categorical")
        else:
            md.update_column(column_name=c, sdtype="numerical")
    return md


def fit_sample(kind, Xtr, ytr, n_per_class: dict, seed=C.SEED):
    """Class-conditional generation. Returns (X_syn, y_syn)."""
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
    outX, outy = [], []
    for cls, n in n_per_class.items():
        if n <= 0:
            continue
        part = Xtr[ytr == cls].reset_index(drop=True)
        if kind == "smotenc":
            continue
        md = make_metadata(part)
        if kind == "copula":
            g = GaussianCopulaSynthesizer(md, default_distribution="beta")
        elif kind == "tvae":
            g = TVAESynthesizer(md, epochs=EPOCHS, batch_size=min(500, max(20, len(part) // 2 * 2)))
        elif kind == "ctgan":
            bs = min(500, max(20, (len(part) // 20) * 10))
            g = CTGANSynthesizer(md, epochs=EPOCHS, batch_size=bs, pac=10 if bs % 10 == 0 else 1)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:
            pass
        g.fit(part)
        s = g.sample(n)
        outX.append(s)
        outy.append(np.full(n, cls))
    if kind == "smotenc":
        n_min = int((ytr == 1).sum() + n_per_class.get(1, 0))
        sm = M.make_sampler("smotenc", Xtr.columns, seed)
        sm.set_params(sampling_strategy={1: n_min})
        Xr, yr = sm.fit_resample(Xtr, ytr)
        new = Xr.iloc[len(Xtr):]
        return new.reset_index(drop=True), yr[len(Xtr):]
    Xs = pd.concat(outX, ignore_index=True)[Xtr.columns]
    for c in INT_COLS:
        Xs[c] = np.round(Xs[c]).clip(Xtr[c].min(), Xtr[c].max())
    for c in ["StudyHoursPerWeek", "SleepHoursPerNight", "ScreenTimeDaily"]:
        Xs[c] = Xs[c].clip(Xtr[c].min(), Xtr[c].max()).round(1)
    return Xs.astype(float), np.concatenate(outy)


def gower_matrix(A, B, ranges, cat_cols, cols):
    """Gower distance (mixed data) from every row of A to every row of B."""
    A, B = A[cols].values, B[cols].values
    is_cat = np.array([c in cat_cols for c in cols])
    D_ = np.zeros((len(A), len(B)))
    for j in range(len(cols)):
        if is_cat[j]:
            D_ += (A[:, [j]] != B[None, :, j]).astype(float)
        else:
            D_ += np.abs(A[:, [j]] - B[None, :, j]) / (ranges[j] if ranges[j] > 0 else 1)
    return D_ / len(cols)


# =========================================================================== #
# PART A - fidelity, privacy, TSTR (generators fitted on the hold-out TRAIN set)
# =========================================================================== #
Xtr, Xte = X.loc[split["train_idx"]].reset_index(drop=True), X.loc[split["test_idx"]].reset_index(drop=True)
ytr, yte = y[split["train_idx"]], y[split["test_idx"]]
cols = list(X.columns)
ranges = (X.max() - X.min()).values
cat_cols = set(C.CATEGORICAL)
mix = {0: int((ytr == 0).sum()), 1: int((ytr == 1).sum())}
gens = ["copula", "tvae", "ctgan"]
GLABEL = {"copula": "Gaussian Copula", "tvae": "TVAE", "ctgan": "CTGAN", "smotenc": "SMOTE-NC"}
syn = {}
for g in gens:
    with Timer(log, f"fit {g} on train partition"):
        syn[g] = fit_sample(g, Xtr, ytr, mix)
# SMOTE-NC "synthetic train" = SMOTE-generated minority only (for privacy reference)
syn["smotenc"] = fit_sample("smotenc", Xtr, ytr, {1: mix[1] * 4})

# Reference DCR: real test -> real train
dcr_ref = gower_matrix(Xte, Xtr, ranges, cat_cols, cols).min(axis=1)
q05 = np.percentile(dcr_ref, 5)

rows, dcrs = [], {"Real hold-out → train": dcr_ref}
lr = lambda: M.make_pipeline("Logistic Regression", cols, "none", params={"clf__C": 1.0})
trtr_auc = roc_auc_score(yte, lr().fit(Xtr, ytr).predict_proba(Xte)[:, 1])
real_corr = Xtr.corr(method="spearman")
for g, (Xs, ys) in syn.items():
    row = {"Generator": GLABEL[g], "Synthetic rows": len(Xs), "Synthetic At-Risk (%)": round(100 * ys.mean(), 1)}
    # --- fidelity
    try:
        from sdv.evaluation.single_table import evaluate_quality
        real_t = Xtr.assign(AtRisk=ytr) if g != "smotenc" else Xtr[ytr == 1].assign(AtRisk=1)
        syn_t = Xs.assign(AtRisk=ys)
        q = evaluate_quality(real_t, syn_t, make_metadata(real_t), verbose=False)
        props = q.get_properties().set_index("Property")["Score"]
        row["SDMetrics column shapes"] = round(props.get("Column Shapes", np.nan), 3)
        row["SDMetrics pair trends"] = round(props.get("Column Pair Trends", np.nan), 3)
        row["SDMetrics overall"] = round(q.get_score(), 3)
    except Exception as e:
        log.info(f"quality report failed for {g}: {e}")
    ref_block = Xtr if g != "smotenc" else Xtr[ytr == 1]
    row["Mean |Δ Spearman ρ|"] = round(np.nanmean(np.abs(ref_block.corr(method="spearman").values -
                                                          Xs.corr(method="spearman").values)), 3)
    # discriminator AUC (real vs synthetic) - 0.5 = indistinguishable
    Z = pd.concat([ref_block, Xs], ignore_index=True)
    lab = np.r_[np.zeros(len(ref_block)), np.ones(len(Xs))]
    from xgboost import XGBClassifier
    pdisc = cross_val_predict(XGBClassifier(n_estimators=200, max_depth=3, eval_metric="logloss", n_jobs=1),
                              Z, lab, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED), method="predict_proba")[:, 1]
    row["Discriminator AUC (↓0.5)"] = round(roc_auc_score(lab, pdisc), 3)
    # --- privacy
    d = gower_matrix(Xs, Xtr, ranges, cat_cols, cols).min(axis=1)
    dcrs[GLABEL[g]] = d
    row["Exact copies of real rows (%)"] = round(100 * np.mean(d < 1e-9), 2)
    row["Median DCR"] = round(np.median(d), 4)
    row["DCR < 5th pct of real hold-out (%)"] = round(100 * np.mean(d < q05), 1)
    # --- utility: TSTR
    if g != "smotenc" and ys.sum() >= 5:
        row["TSTR ROC-AUC (real test)"] = round(roc_auc_score(yte, lr().fit(Xs, ys).predict_proba(Xte)[:, 1]), 4)
    rows.append(row)
s18 = pd.DataFrame(rows)
save_table(s18, "S18_synthetic_fidelity_privacy",
           "Fidelity, privacy and utility of synthetic data generated from the training partition.",
           note=f"Generators fitted per class on the hold-out training partition (n = {len(Xtr)}). Fidelity: SDMetrics "
                f"quality scores (1 = perfect), mean absolute difference of Spearman correlation matrices, and a 5-fold "
                f"XGBoost discriminator (AUC 0.5 = synthetic indistinguishable from real). Privacy: Gower distance to the "
                f"closest real training record (DCR); reference = real hold-out students (5th percentile = {q05:.4f}). "
                f"A synthetic set is privacy-safe when its DCR distribution is not shifted below the reference. Utility: "
                f"train on synthetic, test on real (TSTR) vs train-real-test-real ROC-AUC = {trtr_auc:.4f}. SMOTE-NC row "
                f"refers to its interpolated minority records only.")

# ---------------- Fig S14 -------------------------------------------------- #
fig = plt.figure(figsize=(7.4, 5.6))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45)
feat4 = ["StudyHoursPerWeek", "SleepHoursPerNight", "ScreenTimeDaily", "Motivation(1-5)"]
gcol = {"copula": P.OKABE_ITO[0], "tvae": P.OKABE_ITO[2], "ctgan": P.OKABE_ITO[1]}
for j, f in enumerate(feat4):
    ax = fig.add_subplot(gs[0, j])
    bins = np.arange(0.5, 6) if f == "Motivation(1-5)" else 25
    ax.hist(Xtr[f], bins=bins, density=True, color="#9ca3af", alpha=0.6, label="Real")
    for g in gens:
        ax.hist(syn[g][0][f], bins=bins, density=True, histtype="step", lw=1.3, color=gcol[g], label=GLABEL[g])
    ax.set_title(C.LABELS[f], loc="left", fontsize=8.5)
    ax.tick_params(labelsize=7)
    if j == 0:
        ax.legend(fontsize=6.3, loc="upper left")
ax = fig.add_subplot(gs[1, :2])
for name, d in dcrs.items():
    col = "#111827" if name.startswith("Real") else gcol.get({v: k for k, v in GLABEL.items()}.get(name, ""), P.OKABE_ITO[3])
    ax.hist(d, bins=40, density=True, histtype="step", lw=1.4 if name.startswith("Real") else 1.1,
            ls="--" if name.startswith("Real") else "-", color=col, label=name)
ax.axvline(q05, color=P.RISK, ls=":", lw=0.9)
ax.set(xlabel="Distance to closest real training record (Gower)", ylabel="Density")
ax.legend(fontsize=6.5)
ax.set_title("(b) Privacy: distance to closest record", loc="left", fontsize=9)
ax = fig.add_subplot(gs[1, 2:])
names = [GLABEL[g] for g in gens]
disc = s18.set_index("Generator").loc[names, "Discriminator AUC (↓0.5)"].values
ax.barh(range(len(names)), disc, color=[gcol[g] for g in gens], height=0.55)
for k_, v in enumerate(disc):
    ax.text(v + 0.01, k_, f"{v:.3f}", va="center", fontsize=7.5)
ax.axvline(0.5, color=P.INK, ls="--", lw=0.9)
ax.text(0.505, -0.62, "← indistinguishable", fontsize=7, color=P.MUTED, va="center")
ax.set_ylim(-0.8, len(names) - 0.5)
ax.set_yticks(range(len(names)), names, fontsize=8)
ax.set_xlim(0.4, 1.02)
ax.set_xlabel("Real-vs-synthetic discriminator ROC-AUC (lower = better)", fontsize=8)
ax.grid(axis="y", visible=False)
ax.set_title("(c) Fidelity: can a classifier tell them apart?", loc="left", fontsize=9)
fig.text(0.01, 0.97, "(a) Marginal distributions: real vs synthetic (training partition)", fontsize=9, fontweight="bold")
P.save(fig, "FigS14_synthetic_fidelity")

# =========================================================================== #
# PART B - augmentation utility under repeated CV (real-only test folds)
# =========================================================================== #
REP, K = (1, 5) if C.FAST else (2, 5)
splits = list(RepeatedStratifiedKFold(n_splits=K, n_repeats=REP, random_state=C.SEED).split(X, y))
CONFIGS = [("baseline", None, None), ("smotenc", "smotenc", "minority"), ("copula-min", "copula", "minority"),
           ("tvae-min", "tvae", "minority"), ("copula-x2", "copula", "x2"), ("tvae-x2", "tvae", "x2"),
           ("copula-x5", "copula", "x5")]
if CTGAN_CV:
    CONFIGS += [("ctgan-min", "ctgan", "minority"), ("ctgan-x2", "ctgan", "x2")]
CLABEL = {"baseline": "Real data only", "smotenc": "SMOTE-NC → 20% minority",
          "copula-min": "Copula minority → 20%", "tvae-min": "TVAE minority → 20%",
          "ctgan-min": "CTGAN minority → 20%", "copula-x2": "Copula ×2 (both classes)",
          "tvae-x2": "TVAE ×2 (both classes)", "ctgan-x2": "CTGAN ×2 (both classes)",
          "copula-x5": "Copula ×5 (both classes)"}
MODELS = ["Logistic Regression", "XGBoost"]
best_params = {}
for m in MODELS:
    est = joblib.load(C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[m]}.joblib").named_steps["clf"]
    best_params[m] = {k: est.get_params()[k.replace("clf__", "")] for k in M.param_grids()[m]}

res = []
with Timer(log, f"augmentation CV: {len(splits)} folds x {len(CONFIGS)} configs"):
    for s_i, (tr, te) in enumerate(splits):
        Xa, ya = X.iloc[tr].reset_index(drop=True), y[tr]
        n0, n1 = int((ya == 0).sum()), int((ya == 1).sum())
        cache = {}
        for name, gen, mode in CONFIGS:
            if gen is None:
                Xaug, yaug = Xa, ya
            else:
                if mode == "minority":
                    need = {1: int(np.ceil(0.25 * n0)) - n1}        # minority -> 20% of total
                else:
                    mult = 1 if mode == "x2" else 4
                    need = {0: n0 * mult, 1: n1 * mult}
                key = (gen, mode)
                Xs, ys = fit_sample(gen, Xa, ya, need, seed=C.SEED + s_i)
                Xaug = pd.concat([Xa, Xs], ignore_index=True)
                yaug = np.r_[ya, ys]
            for m in MODELS:
                pipe = M.make_pipeline(m, X.columns, "none", params=best_params[m])
                pipe.fit(Xaug, yaug)
                p = pipe.predict_proba(X.iloc[te])[:, 1]
                d = E.all_metrics(y[te], p)
                d.update(config=name, model=m, fold=s_i, n_train=len(yaug), syn_rows=len(yaug) - len(ya))
                res.append(d)
        log.info(f"fold {s_i + 1}/{len(splits)} done")
R = pd.DataFrame(res)
R.to_csv(C.TAB_DIR / "_augmentation_fold_metrics.csv", index=False)

n_tr = int(len(y) * (K - 1) / K)
rows = []
for m in MODELS:
    base = R[(R.model == m) & (R.config == "baseline")].sort_values("fold")
    for name, _, _ in CONFIGS:
        r = R[(R.model == m) & (R.config == name)].sort_values("fold")
        row = {"Model": m, "Training data": CLABEL[name], "Synthetic rows / fold": int(r.syn_rows.mean()),
               "ROC-AUC": mean_sd(r.roc_auc), "PR-AUC": mean_sd(r.pr_auc), "Recall@0.5": mean_sd(r.recall),
               "F1@0.5": mean_sd(r.f1), "Brier": mean_sd(r.brier, 4)}
        if name != "baseline":
            for met in ["roc_auc", "pr_auc"]:
                t, _, p = E.corrected_resampled_t(r[met].values, base[met].values, n_tr, len(y) - n_tr)
                row[f"Δ{E.METRIC_LABELS[met].split(' ')[0]} vs real (p)"] = f"{(r[met].values - base[met].values).mean():+.4f} ({p:.3f})"
        rows.append(row)
save_table(pd.DataFrame(rows), "S19_augmentation_utility",
           f"Does synthetic augmentation improve At-Risk detection? ({REP}×{K}-fold repeated CV, evaluation on real students only)",
           note="Generators are fitted on the training fold only; synthetic rows never enter a test fold. Δ = mean paired "
                "difference vs real-data-only training; p from the Nadeau–Bengio corrected resampled t-test.")

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
names = [c[0] for c in CONFIGS if c[0] != "baseline"]
for ax, met in zip(axes, ["roc_auc", "pr_auc"]):
    for k_, m in enumerate(MODELS):
        base = R[(R.model == m) & (R.config == "baseline")].sort_values("fold")[met].values
        for j, name in enumerate(names):
            d = R[(R.model == m) & (R.config == name)].sort_values("fold")[met].values - base
            se = d.std(ddof=1) * np.sqrt(1 / len(d) + (len(y) - n_tr) / n_tr)
            ax.errorbar(d.mean(), j + (k_ - 0.5) * 0.3, xerr=1.96 * se, fmt=P.MODEL_MARK[m], color=P.MODEL_COLORS[m],
                        ms=4.5, capsize=0, lw=1, label=C.MODEL_SHORT[m] if j == 0 else None)
    ax.axvline(0, color=P.INK, lw=0.8)
    ax.set_title(f"Δ {E.METRIC_LABELS[met]} vs real-only", loc="left", fontsize=9.5)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(names)), [CLABEL[n] for n in names], fontsize=7.8)
axes[0].invert_yaxis()
axes[0].legend(fontsize=7.5, loc="lower left")
fig.suptitle("Effect of synthetic augmentation (mean ± 95% corrected CI; real test folds)", x=0.01, ha="left",
             fontweight="bold", fontsize=10)
fig.tight_layout()
P.save(fig, "FigS15_augmentation_effect")

# =========================================================================== #
# PART C - privacy-preserving synthetic release (fitted on ALL real data)
# =========================================================================== #
best_gen = s18[s18.Generator != "SMOTE-NC"].sort_values("Discriminator AUC (↓0.5)").iloc[0]["Generator"]
gkey = {v: k for k, v in GLABEL.items()}[best_gen]
N_REL = 5000
with Timer(log, f"release dataset with {best_gen}"):
    n1 = int(round(N_REL * y.mean()))
    Xs, ys = fit_sample(gkey, X.reset_index(drop=True), y, {0: N_REL - n1, 1: n1}, seed=2026)
dec = pd.DataFrame({c: [D.decode_value(c, v) for v in Xs[c]] if c in C.CATEGORICAL else Xs[c] for c in X.columns})
dec["AtRisk"] = ys
d_all = gower_matrix(Xs, X, ranges, cat_cols, cols).min(axis=1)
dec.to_csv(C.DATA_SYNTH / f"PREX-Edu_SYNTHETIC_n{N_REL}.csv", index=False)
(C.DATA_SYNTH / "DATASHEET.md").write_text(f"""# PREX-Edu synthetic dataset — datasheet

**This file contains NO real students.** It was generated by a class-conditional {best_gen}
model (SDV) fitted on the 1,208-record PREX-Edu primary dataset, and is intended for code
sharing, teaching and reproducibility checks only. It must never be pooled with the real
data or used to report model performance.

* Rows: {N_REL} (At-Risk = {ys.mean():.1%}, matching the real prevalence {y.mean():.1%})
* Columns: the 13 model predictors + binary `AtRisk` (LastTermPercentage and Grade are withheld)
* Generator chosen by lowest real-vs-synthetic discriminator AUC on the training partition (Table S18)
* Privacy: exact copies of real records = {100 * np.mean(d_all < 1e-9):.2f}%;
  median Gower distance to closest real record = {np.median(d_all):.4f}
* Recommended citation text: "A synthetic version of the dataset, generated with {best_gen}
  (Patki et al., 2016) and audited for fidelity and privacy, is available at <repository URL>."
""", encoding="utf-8")
log.info("done")
