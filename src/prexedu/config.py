"""
Central configuration for the PREX-Edu project.

Every script imports from here, so changing a path, the risk threshold, the
model grids or the evaluation protocol in ONE place propagates everywhere.

Environment variables
---------------------
PREX_FAST=1   -> quick smoke-test mode (fewer repeats / bootstrap iterations /
                 smaller grids). Use it to check that the pipeline runs; never
                 report numbers produced in FAST mode.
PREX_NJOBS=k  -> number of parallel workers (default: all cores).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
# Real survey data are NOT distributed. Set PREX_DATA to another file (e.g. the
# synthetic replica data/synthetic/PREX-Edu_SYNTHETIC_n5000.csv) to run the
# pipeline without it.
DATA_RAW = Path(os.environ.get("PREX_DATA", ROOT / "data" / "raw" / "student_performance_dataset.xlsx"))
if not DATA_RAW.is_absolute():
    DATA_RAW = ROOT / DATA_RAW          # relative paths are taken from the repository root
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SYNTH = ROOT / "data" / "synthetic"
DATA_PILOT = ROOT / "data" / "educator_pilot"
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
MODEL_DIR = OUT / "models"
LOG_DIR = OUT / "logs"
for _d in (DATA_PROCESSED, DATA_SYNTH, DATA_PILOT, FIG_DIR, TAB_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Run-mode switches
# --------------------------------------------------------------------------- #
FAST = os.environ.get("PREX_FAST", "0") == "1"
N_JOBS = int(os.environ.get("PREX_NJOBS", "-1"))
SEED = 42

# --------------------------------------------------------------------------- #
# Target definition
# --------------------------------------------------------------------------- #
# At-Risk = LastTermPercentage <= RISK_THRESHOLD  (66 / 1,208 = 5.46 %)
RISK_THRESHOLD = 50.0
TARGET = "AtRisk"
TARGET_SOURCE = "LastTermPercentage"
# Columns removed before modelling to prevent leakage:
#  - LastTermPercentage defines the target
#  - Grade is (almost) a deterministic function of Age and was dropped in the
#    original design (Algorithm 1, step 2)
LEAKAGE_COLS = ["LastTermPercentage", "Grade"]

# --------------------------------------------------------------------------- #
# Feature schema  (13 predictors after leakage removal)
# --------------------------------------------------------------------------- #
NUMERIC = ["Age", "StudyHoursPerWeek", "SleepHoursPerNight", "ScreenTimeDaily",
           "Motivation(1-5)", "Stress(1-5)"]
BINARY = ["StudySpace", "Extracurricular", "HighAttendance"]          # Yes/No -> 1/0
ORDINAL = {                                                           # ordered categories
    "ParentEducation": ["No formal", "High school", "College", "Postgrad"],
    "IncomeRange": ["Low", "Medium", "High"],
}
NOMINAL = {                                                           # one-hot inside the model
    "Gender": ["Female", "Male", "Prefer not to say"],
    "DifficultSubject": ["Math", "Science", "English", "History", "Geography", "Unknown"],
}
FEATURES = ["Age", "Gender", "ParentEducation", "IncomeRange", "StudySpace",
            "StudyHoursPerWeek", "SleepHoursPerNight", "ScreenTimeDaily",
            "Motivation(1-5)", "Stress(1-5)", "Extracurricular", "HighAttendance",
            "DifficultSubject"]
CATEGORICAL = list(ORDINAL) + list(NOMINAL) + BINARY                  # for SMOTE-NC / LIME

# Pretty labels for figures
LABELS = {
    "StudyHoursPerWeek": "Study hours / week",
    "SleepHoursPerNight": "Sleep hours / night",
    "ScreenTimeDaily": "Screen time / day",
    "Motivation(1-5)": "Motivation (1–5)",
    "Stress(1-5)": "Stress (1–5)",
    "HighAttendance": "High attendance",
    "ParentEducation": "Parent education",
    "IncomeRange": "Income range",
    "StudySpace": "Study space",
    "Extracurricular": "Extracurricular",
    "DifficultSubject": "Difficult subject",
    "Gender": "Gender",
    "Age": "Age",
}

# --------------------------------------------------------------------------- #
# PREX-Edu pedagogical mapping (Table 1)
# --------------------------------------------------------------------------- #
PREX_DIMENSIONS = {
    "Engagement": ["StudyHoursPerWeek", "Motivation(1-5)"],
    "Lifestyle": ["SleepHoursPerNight", "ScreenTimeDaily"],
    "Cognitive Load": ["Stress(1-5)"],
    "Participation": ["HighAttendance"],
}
# Aggregation operator for R_k (see script 15 / Table S24):
#   "mean" = Eq. 3 of the draft (cardinality-normalised mean |phi|)
#   "sum"  = sum |phi| (naive) - agreed best with exact group-Shapley in our data
PREX_AGGREGATION = os.environ.get("PREX_AGG", "sum")
# Non-actionable socio-demographic context. NOT part of the four PREX-Edu
# dimensions, but tracked so that we can report how much attribution mass the
# four dimensions cover (a question reviewers will ask).
CONTEXT_DIMENSION = {"Context (non-actionable)": [
    "Age", "Gender", "ParentEducation", "IncomeRange", "StudySpace",
    "Extracurricular", "DifficultSubject"]}

# Direction of a beneficial change for each actionable feature (+1 = increase
# lowers risk, -1 = decrease lowers risk) and plausible bounds for
# counterfactual / intervention simulation.
ACTIONABLE = {
    "StudyHoursPerWeek": (+1, 1.0, 30.0),
    "Motivation(1-5)": (+1, 1, 5),
    "SleepHoursPerNight": (+1, 4.0, 9.5),
    "ScreenTimeDaily": (-1, 0.5, 9.0),
    "Stress(1-5)": (-1, 1, 5),
    "HighAttendance": (+1, 0, 1),
}

# --------------------------------------------------------------------------- #
# Evaluation protocol
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.30                 # legacy hold-out (kept for continuity with the earlier draft)
CV_FOLDS = 10
CV_REPEATS = 2 if FAST else 10   # repeated stratified K-fold on ALL 1,208 records
INNER_FOLDS = 3 if FAST else 5   # nested hyper-parameter tuning
N_BOOT = 200 if FAST else 2000   # bootstrap iterations for CIs
N_RANDOM_ABLATION = 30 if FAST else 100
SHAP_SEEDS = [42, 7, 13, 99, 2024, 314, 777, 101, 55, 888]
N_LIME_INSTANCES = 50
TUNING_SCORING = "average_precision"   # PR-AUC is the right target under 5 % prevalence
# Imbalance handling used for the PRIMARY analysis (see script 11 for the
# comparison of all strategies that justifies this choice).
PRIMARY_STRATEGY = os.environ.get("PREX_STRATEGY", "none")
FINAL_MODEL = "Logistic Regression"     # explained model (Sections 5.3-5.5)

MODEL_ORDER = ["Logistic Regression", "SVM", "Random Forest", "Decision Tree",
               "Gradient Boosting", "XGBoost"]
MODEL_SHORT = {"Logistic Regression": "LR", "SVM": "SVM", "Random Forest": "RF",
               "Decision Tree": "DT", "Gradient Boosting": "GB", "XGBoost": "XGB"}
