**Instance-level faithfulness: area over the perturbation curve (AOPC) for students flagged At-Risk.**

| Removal order   |   Students |   AOPC@6 mean |   AOPC@6 SD |   Mean prob. after 1 removal |   Mean prob. after 3 removals |
|:----------------|-----------:|--------------:|------------:|-----------------------------:|------------------------------:|
| PREX-Edu        |         21 |        0.6075 |      0.1903 |                       0.0704 |                        0.0021 |
| SHAP            |         21 |        0.6092 |      0.1899 |                       0.0704 |                        0.0022 |
| Random          |         21 |        0.3187 |      0.1125 |                       0.4997 |                        0.327  |

*Note.* Flagged = P(At-Risk) ≥ 0.339 (F1-optimal threshold from training-set CV). AOPC@6 = mean drop in P(At-Risk) over the first six removals; higher = removed features mattered more. Wilcoxon PREX vs random: W = 0.0, p = 9.54e-07; PREX vs SHAP: W = 51.0, p = 0.024. SHAP order is the per-instance upper bound; PREX-Edu constrains removal to whole dimensions.