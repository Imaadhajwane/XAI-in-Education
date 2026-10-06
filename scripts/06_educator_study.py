"""
06 - Educator validation study: analysis, power, and study materials
====================================================================
A. ANALYSIS of the pilot (Tables 14-15, Fig 13)
   Needs the raw ratings in  data/educator_pilot/ratings.csv  (long format,
   see ratings_TEMPLATE.csv written by this script). The raw pilot responses
   are NOT reconstructed here - only real responses may be analysed.
   Columns: rater, profile, format (PREX|SHAP), clarity, actionability,
            trust, alignment (1-5 Likert), top_factor (free text / category)
B. POWER ANALYSIS for the replication (Table S15, Fig S13)
   Monte-Carlo power of the paired Wilcoxon signed-rank test for a range of
   effect sizes and numbers of (rater x profile) pairs, plus a cluster-aware
   design effect for repeated ratings by the same teacher.
C. STUDY MATERIALS for the replication
   12 student explanation cards x 2 formats (raw SHAP vs PREX-Edu), blinded
   codes, randomised presentation order per rater, answer key.
   -> outputs/educator_study_materials/
"""
from _common import C, get_logger

import textwrap

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from prexedu import data as D
from prexedu import plotting as P
from prexedu import prex
from prexedu.tables import save_table

log = get_logger("06_educator")
CONSTRUCTS = ["clarity", "actionability", "trust", "alignment"]
CLABEL = {"clarity": "Clarity", "actionability": "Actionability", "trust": "Trust",
          "alignment": "Pedagogical alignment"}
FEATURE_TO_DIM = {**{f.lower(): k for k, v in C.PREX_DIMENSIONS.items() for f in v},
                  "study hours": "Engagement", "motivation": "Engagement", "sleep": "Lifestyle",
                  "screen time": "Lifestyle", "stress": "Cognitive Load", "attendance": "Participation"}

# --------------------------------------------------------------------------- #
# Template
# --------------------------------------------------------------------------- #
tmpl = pd.DataFrame([
    {"rater": "T01", "profile": "P01", "format": "PREX", "clarity": 4, "actionability": 4, "trust": 4,
     "alignment": 4, "top_factor": "Engagement"},
    {"rater": "T01", "profile": "P01", "format": "SHAP", "clarity": 3, "actionability": 3, "trust": 4,
     "alignment": 3, "top_factor": "Study hours"},
])
tmpl.to_csv(C.DATA_PILOT / "ratings_TEMPLATE.csv", index=False)
save_table(pd.DataFrame({
    "#": [1, 2, 3, 4, 5],
    "Item": ["This explanation is easy to understand without technical training.",
             "This explanation suggests a clear next step for intervention.",
             "I would be comfortable using this explanation to inform a decision about this student.",
             "This explanation matches how I naturally think about student risk.",
             "Which single factor do you consider most influential in this prediction?"],
    "Construct": ["Clarity", "Actionability", "Trust", "Pedagogical alignment", "Factor identification"],
    "Scale": ["1 (Strongly disagree) – 5 (Strongly agree)", "1–5", "1–5", "1–5", "Free text / category"]}),
    "A09_survey_instrument", "Educator survey instrument (administered per profile and format).",
    note="Blinded within-subjects design: each educator rates every profile in both formats (raw SHAP, PREX-Edu) in a "
         "randomised order; consent obtained; responses anonymous.")


def fleiss_kappa(table: np.ndarray) -> float:
    """table: subjects x categories counts."""
    N, k = table.shape
    n = table.sum(axis=1)[0]
    p = table.sum(axis=0) / (N * n)
    Pi = (np.sum(table ** 2, axis=1) - n) / (n * (n - 1))
    Pbar, Pe = Pi.mean(), np.sum(p ** 2)
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else np.nan


def to_dim(s: str) -> str:
    s = str(s).strip().lower()
    for k in C.PREX_DIMENSIONS:
        if k.lower() in s:
            return k
    for key, d in FEATURE_TO_DIM.items():
        if key in s:
            return d
    return "Outside scope"


# --------------------------------------------------------------------------- #
# A. Analysis (only if real data present)
# --------------------------------------------------------------------------- #
rp = C.DATA_PILOT / "ratings.csv"
if rp.exists():
    R = pd.read_csv(rp)
    R["format"] = R["format"].str.upper().str.strip()
    R["dim"] = R["top_factor"].map(to_dim)
    n_r, n_p = R.rater.nunique(), R.profile.nunique()
    log.info(f"pilot data: {n_r} raters x {n_p} profiles")
    # Table 14: Fleiss' kappa on coded factor
    rows = []
    cats = list(C.PREX_DIMENSIONS) + ["Outside scope"]
    for fmt in ["SHAP", "PREX"]:
        sub = R[R.format == fmt]
        tab = np.array([[np.sum(sub[sub.profile == p_].dim == c) for c in cats] for p_ in sorted(sub.profile.unique())])
        k_ = fleiss_kappa(tab)
        lvl = ("Poor (< 0)" if k_ < 0 else "Slight" if k_ <= .2 else "Fair" if k_ <= .4 else "Moderate"
               if k_ <= .6 else "Substantial" if k_ <= .8 else "Almost perfect")
        rows.append({"Explanation format": "Raw SHAP" if fmt == "SHAP" else "PREX-Edu", "Fleiss' κ": round(k_, 3),
                     "Interpretation (Landis & Koch, 1977)": lvl, "Raters": n_r, "Profiles": n_p,
                     "Responses outside scope": int((sub.dim == "Outside scope").sum())})
    save_table(pd.DataFrame(rows), "T14_fleiss_kappa", "Inter-rater agreement on the most influential factor, by explanation format.",
               note="Free-text answers coded to the parent PREX-Edu dimension before analysis.")
    # Table 15: Wilcoxon
    W = R.pivot_table(index=["rater", "profile"], columns="format", values=CONSTRUCTS)
    rows = []
    for c in CONSTRUCTS:
        a, b = W[(c, "PREX")], W[(c, "SHAP")]
        d = a - b
        res = stats.wilcoxon(a, b, zero_method="wilcox", method="approx")
        z = stats.norm.isf(res.pvalue / 2) * np.sign(d.median() if d.median() != 0 else d.mean())
        rows.append({"Construct": CLABEL[c], "Mean PREX-Edu": round(a.mean(), 2), "SD PREX": round(a.std(), 2),
                     "Mean SHAP": round(b.mean(), 2), "SD SHAP": round(b.std(), 2), "W": res.statistic,
                     "p": round(res.pvalue, 3), "Effect size r = Z/√N": round(z / np.sqrt(len(d)), 2),
                     "Cohen's d_z": round(d.mean() / d.std(ddof=1), 2) if d.std() > 0 else np.nan, "N pairs": len(d)})
    t15 = pd.DataFrame(rows)
    save_table(t15, "T15_wilcoxon_pilot", "Paired comparison of PREX-Edu vs raw SHAP ratings (Wilcoxon signed-rank).",
               note="Positive r favours PREX-Edu. Pairs are rater × profile; ratings by the same rater are not independent, "
                    "so p-values are optimistic — see the mixed-model analysis in the replication protocol.")
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    xs = np.arange(4)
    for j, (fmt, col, lab) in enumerate([("PREX", P.OKABE_ITO[0], "PREX-Edu"), ("SHAP", "#9ca3af", "Raw SHAP")]):
        mu = [W[(c, fmt)].mean() for c in CONSTRUCTS]
        sd = [W[(c, fmt)].std() for c in CONSTRUCTS]
        ax.bar(xs + (j - 0.5) * 0.36, mu, 0.34, yerr=sd, color=col, label=lab, error_kw=dict(lw=0.8, capsize=2))
    for x_, (_, r_) in zip(xs, t15.iterrows()):
        dz_, pv_ = r_["Cohen's d_z"], r_["p"]
        ax.text(x_, 5.25, f"d_z = {dz_:.2f}\np = {pv_:.2f}", ha="center", fontsize=7, color=P.MUTED)
    ax.set_xticks(xs, [CLABEL[c] for c in CONSTRUCTS])
    ax.set(ylim=(0, 5.9), ylabel="Mean rating (1–5) ± SD")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title(f"Educator ratings: PREX-Edu vs raw SHAP ({n_r} raters × {n_p} profiles)", loc="left")
    P.save(fig, "Fig13_educator_ratings")
else:
    log.info("No data/educator_pilot/ratings.csv - pilot tables skipped (template written). "
             "Enter the real pilot responses to regenerate Tables 14-15 and Fig 13.")

# --------------------------------------------------------------------------- #
# B. Power analysis
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(C.SEED)
effects = [0.3, 0.4, 0.5, 0.6, 0.8]
ns = [10, 15, 20, 30, 40, 50, 60, 80, 100]
SIMS = 400 if C.FAST else 2000


def sim_power(dz, n, icc=0.0, raters=None):
    """Paired Likert differences: latent normal -> rounded 1-5 ratings."""
    hits = 0
    for _ in range(SIMS):
        if icc > 0 and raters:
            per = int(np.ceil(n / raters))
            rater_eff = np.repeat(rng.normal(0, np.sqrt(icc), raters), per)[:n]
        else:
            rater_eff = 0
        base = rng.normal(3.4, 0.8, n)
        diff = dz * 1.0 + rater_eff + rng.normal(0, np.sqrt(1 - icc), n)
        a = np.clip(np.round(base + diff / 2), 1, 5)
        b = np.clip(np.round(base - diff / 2), 1, 5)
        if np.all(a == b):
            continue
        if stats.wilcoxon(a, b, zero_method="wilcox").pvalue < 0.05:
            hits += 1
    return hits / SIMS


rows = []
for dz in effects:
    for n in ns:
        rows.append({"Effect size d_z": dz, "N pairs": n, "Power (independent pairs)": sim_power(dz, n)})
pw = pd.DataFrame(rows)
need = pw[pw["Power (independent pairs)"] >= 0.8].groupby("Effect size d_z")["N pairs"].min()
icc = 0.2
design = []
for raters, profiles in [(3, 5), (8, 10), (10, 10), (12, 12), (15, 12), (20, 12)]:
    n = raters * profiles
    design.append({"Raters": raters, "Profiles": profiles, "Pairs": n,
                   "Design effect (ICC = 0.2)": round(1 + (profiles - 1) * icc, 2),
                   "Effective N": round(n / (1 + (profiles - 1) * icc), 1),
                   "Power d_z = 0.4": sim_power(0.4, n, icc, raters),
                   "Power d_z = 0.5": sim_power(0.5, n, icc, raters)})
save_table(pw.pivot(index="N pairs", columns="Effect size d_z", values="Power (independent pairs)").reset_index()
           .rename(columns=lambda c: f"d_z = {c}" if isinstance(c, float) else c), "S15_power_wilcoxon",
           "Monte-Carlo power of the paired Wilcoxon signed-rank test (α = .05, two-sided) on 1–5 Likert ratings.",
           note="Minimum pairs for 80% power: " + "; ".join(f"d_z={k}: {v}" for k, v in need.items()) +
                f". {SIMS} simulations per cell.")
save_table(pd.DataFrame(design), "S16_power_designs",
           "Power of candidate replication designs when ratings are clustered within raters (ICC = 0.2).",
           note=f"A 3 × 5 design has ≈ {100 * design[0]['Power d_z = 0.5']:.0f}% power for a medium effect (d_z = 0.5); "
                f"≥ 10 raters × 10 profiles gives ≥ 80% power for d_z = 0.4.")
fig, ax = plt.subplots(figsize=(4.8, 3.0))
for j, dz in enumerate(effects):
    s_ = pw[pw["Effect size d_z"] == dz]
    ax.plot(s_["N pairs"], s_["Power (independent pairs)"], marker="o", ms=3.5, color=P.OKABE_ITO[j],
            ls=["-", "--", "-.", ":", "-"][j], label=f"d_z = {dz}")
ax.axhline(0.8, color=P.RISK, ls=":", lw=0.9)
ax.axvline(15, color=P.MUTED, ls="--", lw=0.8)
ax.text(16, 0.05, "3×5 design\n(15 pairs)", fontsize=7, color=P.MUTED)
ax.set(xlabel="Number of paired ratings", ylabel="Power", ylim=(0, 1.02))
ax.legend(fontsize=7, loc="lower right")
ax.set_title("Power of the educator comparison (Wilcoxon)", loc="left")
P.save(fig, "FigS13_power_curves")

# --------------------------------------------------------------------------- #
# C. Study materials
# --------------------------------------------------------------------------- #
out = C.OUT / "educator_study_materials"
out.mkdir(exist_ok=True)
try:
    FM = C.FINAL_MODEL
    split = joblib.load(C.MODEL_DIR / "holdout_split.joblib")
    df, X, y = D.load_xy()
    Xte = X.loc[split["test_idx"]]
    pipe = joblib.load(C.MODEL_DIR / f"holdout_{C.MODEL_SHORT[FM]}.joblib")
    S = pd.read_csv(C.MODEL_DIR / f"shap_test_{C.MODEL_SHORT[FM]}.csv", index_col=0)
    prob = pipe.predict_proba(Xte)[:, 1]
    Rk, Rs = prex.prex_scores(S), prex.prex_scores(S, signed=True)
    dom = prex.dominant(Rk)["Dominant"].values
    # pick 12 profiles: predicted risk >= 0.3, balanced across dominant dimensions
    cand = np.where(prob >= 0.3)[0]
    picks = []
    for k in C.PREX_DIMENSIONS:
        picks += list(cand[dom[cand] == k][:4])
    picks = (picks + [i for i in np.argsort(-prob) if i not in picks])[:12]
    key = []
    for n_, i in enumerate(picks):
        pid = f"P{n_ + 1:02d}"
        # SHAP card
        s = S.iloc[i].reindex(S.iloc[i].abs().sort_values(ascending=False).index)[:8][::-1]
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.barh(range(len(s)), s.values, color=[P.RISK if v > 0 else P.PROTECT for v in s.values], height=0.6)
        ax.set_yticks(range(len(s)), [f"{C.LABELS[f]} = {D.decode_value(f, Xte.iloc[i][f])}" for f in s.index], fontsize=7.5)
        ax.axvline(0, color=P.MUTED, lw=0.8)
        ax.set_xlabel("Contribution to risk score (SHAP, log-odds)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Student {pid} — predicted risk {prob[i]:.0%}", loc="left", fontsize=9.5)
        fig.savefig(out / f"{pid}_format_A_shap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        # PREX card
        r_ = Rk.iloc[i].sort_values()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[1, 1.1]))
        a1.barh(range(4), r_.values, color=[P.DIM_COLORS[k] for k in r_.index], height=0.6)
        a1.set_yticks(range(4), r_.index, fontsize=8)
        a1.set_xlabel("Risk contribution of each area")
        a1.grid(axis="y", visible=False)
        a1.set_title(f"Student {pid} — predicted risk {prob[i]:.0%}", loc="left", fontsize=9.5)
        a2.axis("off")
        a2.text(0, 0.95, "\n".join(textwrap.wrap(prex.explain_text(Rk.iloc[i], Rs.iloc[i]), 50)), va="top",
                fontsize=7.8, bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec=P.GRID))
        fig.tight_layout()
        fig.savefig(out / f"{pid}_format_B_prexedu.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        key.append({"profile": pid, "test_row": int(Xte.index[i]), "predicted_risk": round(prob[i], 3),
                    "true_label": int(y[split["test_idx"]][i]), "PREX_dominant": dom[i],
                    "SHAP_top_feature": C.LABELS[S.iloc[i].abs().idxmax()]})
    pd.DataFrame(key).to_csv(out / "answer_key.csv", index=False)
    # randomised, counter-balanced order for 20 raters
    orders = []
    for r_i in range(1, 21):
        seq = [(p_["profile"], f) for p_ in key for f in ("A_shap", "B_prexedu")]
        rr = np.random.default_rng(1000 + r_i)
        rr.shuffle(seq)
        for pos_, (p_, f) in enumerate(seq, 1):
            orders.append({"rater": f"T{r_i:02d}", "position": pos_, "card": f"{p_}_format_{f}.png"})
    pd.DataFrame(orders).to_csv(out / "presentation_order_by_rater.csv", index=False)
    (out / "README.md").write_text(
        "# Educator study materials (PREX-Edu replication)\n\n"
        "* `Pxx_format_A_shap.png` – raw SHAP explanation card; `Pxx_format_B_prexedu.png` – PREX-Edu card.\n"
        "* Present cards in the order given in `presentation_order_by_rater.csv` (randomised per rater).\n"
        "* After each card collect the 4 Likert items + most-influential factor (Appendix D, Table A9).\n"
        "* Enter responses in `data/educator_pilot/ratings.csv` using `ratings_TEMPLATE.csv` columns, then "
        "run `python scripts/06_educator_study.py` to regenerate Tables 14–15 and Figure 13.\n"
        "* `answer_key.csv` is for the analyst only – do not show it to raters.\n", encoding="utf-8")
    log.info(f"study materials -> {out}")
except FileNotFoundError:
    log.info("run scripts 01-02 first to create study materials")
log.info("done")
