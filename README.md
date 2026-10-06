# PREX-Edu

**From feature attributions to pedagogical levers in the early identification of at-risk secondary-school students**

Code, results and a privacy-checked synthetic dataset for the PREX-Edu study (submitted to the *Journal of Educational Data Mining*).

PREX-Edu groups SHAP attributions of an at-risk prediction model into four pedagogical dimensions: Engagement, Lifestyle, Cognitive Load and Participation. It then tests whether these grouped explanations are **faithful** to the model and **actionable** for students. The pipeline covers:

- nested 10×10 cross-validation of six classifiers;
- calibration and decision-curve analysis;
- SHAP, LIME and PDP explanations;
- the PREX-Edu aggregation and its validation (ablation, AOPC, exact group-Shapley, intervention simulation, counterfactuals);
- a fairness audit;
- class-imbalance and synthetic-data experiments.

---

## Data

| File | Status |
|---|---|
| `data/raw/student_performance_dataset.xlsx` | **Not distributed.** Primary survey of 1,208 secondary-school students; available from the authors on reasonable request, subject to ethics approval. |
| `data/synthetic/PREX-Edu_SYNTHETIC_n5000.csv` | **Open.** 5,000 synthetic records from a class-conditional Gaussian copula. No exact copies of real students; see `data/synthetic/DATASHEET.md`. |

The synthetic file contains the 13 predictors and the binary `AtRisk` label. It is meant for running and checking the code, not for reporting results.

## Quick start

```bash
git clone https://github.com/<user>/PREX-Edu.git
cd PREX-Edu
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # exact versions: requirements-lock.txt
```

Figure 1 also needs [Graphviz](https://graphviz.org/download/) on the PATH.

**Run on the synthetic data** (anyone):

```bash
# macOS / Linux
export PREX_DATA=data/synthetic/PREX-Edu_SYNTHETIC_n5000.csv
# Windows (cmd)
set PREX_DATA=data/synthetic/PREX-Edu_SYNTHETIC_n5000.csv

python run_all.py --fast --skip 00 12 16 17 99
```

Steps 00, 16 and 17 need the continuous last-term percentage, and step 12 refits the synthetic generators, so these are skipped.

**Run on the real data** (with the file in `data/raw/`):

```bash
python run_all.py                      # full run, about 1.5–3 h on a 4-core laptop
python run_all.py --only 10            # nested cross-validation only
python scripts/17_paper_figures.py     # redraw the manuscript figures from saved results
```

## Repository layout

```
src/prexedu/      configuration, data handling, models, CV, evaluation, XAI, PREX-Edu, plotting
scripts/          numbered analysis steps (00–17) and the methodology flowchart (08b)
run_all.py        runs the steps in order
outputs/tables/   every result table (CSV, LaTeX, Markdown) + PREX-Edu_all_tables.xlsx
outputs/figures/  every figure (PDF + 600-dpi PNG)
data/synthetic/   open synthetic replica + datasheet
```

## Script → manuscript map

| Manuscript item | Script | Output |
|---|---|---|
| Fig. 1 methodology | `08b_methodology_flowchart.py` | `FigM_methodology_flowchart` |
| Figs. 2, 3, 5–12 | `17_paper_figures.py` | `PF02`–`PF11` |
| Fig. 4, Table 3, Tables 9–10 (nested CV, tests, calibration) | `10_nested_repeated_cv.py`, `13_calibration_threshold_dca.py` | `P01`–`P07`, `S20`, `S21` |
| Table 1 (descriptives) | `00_data_audit_eda.py` | `S01`, `S02` |
| Table 4, Fig. 13–14 (SHAP, PFI, PDP, stability, LIME) | `02_xai_shap_lime_pdp.py` | `S06`, `T10`, `T11` |
| Table 5 (faithfulness) | `03_prexedu_framework.py` | `T12`, `S07`–`S09` |
| Aggregation operator, cross-model (Table 6) | `15_aggregation_crossmodel.py` | `S24`, `S25` |
| Intervention, counterfactuals | `05_intervention_counterfactuals.py` | `S11`–`S14` |
| Fairness | `14_fairness_extended.py` | `S22`, `S23` |
| Table 7 (class imbalance) | `11_imbalance_strategies.py` | `S05` |
| Synthetic data | `12_synthetic_augmentation.py` | `S18`, `S19` |
| Threshold sensitivity (Table 12) | `16_target_sensitivity.py` | `S26` |
| Teacher-study power (Tables 13–14) | `06_educator_study.py` | `S16` |
| Hold-out evaluation (Table 11) | `01_model_comparison_holdout.py` | `T06`–`T09` |

All settings (seed 42, folds, grids, the τ = 50 threshold, PREX-Edu dimensions, aggregation operator) are in `src/prexedu/config.py`.

## Citation

If you use this code or the synthetic data, please cite the paper (see `CITATION.cff`).

## License

Code is released under the MIT License. The synthetic dataset is released under CC BY 4.0.
