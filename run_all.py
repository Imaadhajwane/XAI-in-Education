"""
Run the complete PREX-Edu pipeline in dependency order.

    python run_all.py                 # full run (≈ 1.5–3 h on a 4-core laptop)
    python run_all.py --fast          # smoke test (≈ 10–15 min) - numbers NOT reportable
    python run_all.py --only 02 03    # run selected steps
    python run_all.py --skip 12       # skip steps (e.g. the slow synthetic-data step)

Dependency graph
    00 ─► 01 ─► 02 ─► 03 ─► 04 ─► 05 ─► 06 ─► 15 ─► 16
                └──► 11          └──► 07, 08
          01 ─► 10 ─► 13 ─► 14
          01 ─► 12
    99 builds the Excel + Word compendium at the end.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _s in (sys.stdout, sys.stderr):           # Windows consoles: avoid UnicodeEncodeError on ✓ ▶ τ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
STEPS = [
    ("00", "00_data_audit_eda.py"), ("01", "01_model_comparison_holdout.py"),
    ("02", "02_xai_shap_lime_pdp.py"), ("03", "03_prexedu_framework.py"),
    ("04", "04_fairness_subgroups.py"), ("05", "05_intervention_counterfactuals.py"),
    ("06", "06_educator_study.py"), ("07", "07_xai_method_comparison.py"),
    ("08", "08_workflow_diagram.py"), ("08b", "08b_methodology_flowchart.py"), ("10", "10_nested_repeated_cv.py"),
    ("11", "11_imbalance_strategies.py"), ("12", "12_synthetic_augmentation.py"),
    ("13", "13_calibration_threshold_dca.py"), ("14", "14_fairness_extended.py"),
    ("15", "15_aggregation_crossmodel.py"), ("16", "16_target_sensitivity.py"), ("17", "17_paper_figures.py"),
    ("99", "99_build_compendium.py"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true")
ap.add_argument("--only", nargs="*")
ap.add_argument("--skip", nargs="*", default=[])
a = ap.parse_args()
env = dict(os.environ)
if a.fast:
    env["PREX_FAST"] = "1"
env.setdefault("PYTHONWARNINGS", "ignore")
reg = ROOT / "outputs" / "tables" / "_registry.json"
if not a.only and reg.exists():
    reg.unlink()                      # fresh table registry for a full run
t0 = time.time()
for code, script in STEPS:
    if (a.only and code not in a.only) or code in a.skip:
        continue
    print(f"\n{'=' * 70}\n▶ {script}\n{'=' * 70}", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, script], cwd=ROOT / "scripts", env=env)
    if r.returncode != 0:
        sys.exit(f"✗ {script} failed (exit {r.returncode})")
    print(f"✓ {script}  ({(time.time() - t) / 60:.1f} min)", flush=True)
print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Results: outputs/")
