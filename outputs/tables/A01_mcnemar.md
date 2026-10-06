**Pairwise McNemar tests on hold-out hard predictions.**

| Model A             | Model B           |   b (A✓ B✗) |   c (A✗ B✓) | χ² (cc)   | Test   |      p |   p (BH-FDR) | Sig. (FDR α=.05)   |
|:--------------------|:------------------|------------:|------------:|:----------|:-------|-------:|-------------:|:-------------------|
| Logistic Regression | SVM               |           5 |           2 |           | exact  | 0.4531 |       0.8464 | ✗                  |
| Logistic Regression | Random Forest     |          11 |           2 |           | exact  | 0.0225 |       0.3369 | ✗                  |
| Logistic Regression | Decision Tree     |           9 |           4 |           | exact  | 0.2668 |       0.6671 | ✗                  |
| Logistic Regression | Gradient Boosting |           5 |           1 |           | exact  | 0.2188 |       0.6671 | ✗                  |
| Logistic Regression | XGBoost           |           9 |           3 |           | exact  | 0.146  |       0.6671 | ✗                  |
| SVM                 | Random Forest     |           8 |           2 |           | exact  | 0.1094 |       0.6671 | ✗                  |
| SVM                 | Decision Tree     |           7 |           5 |           | exact  | 0.7744 |       0.968  | ✗                  |
| SVM                 | Gradient Boosting |           4 |           3 |           | exact  | 1      |       1      | ✗                  |
| SVM                 | XGBoost           |           6 |           3 |           | exact  | 0.5078 |       0.8464 | ✗                  |
| Random Forest       | Decision Tree     |           5 |           9 |           | exact  | 0.424  |       0.8464 | ✗                  |
| Random Forest       | Gradient Boosting |           4 |           9 |           | exact  | 0.2668 |       0.6671 | ✗                  |
| Random Forest       | XGBoost           |           6 |           9 |           | exact  | 0.6072 |       0.9109 | ✗                  |
| Decision Tree       | Gradient Boosting |           5 |           6 |           | exact  | 1      |       1      | ✗                  |
| Decision Tree       | XGBoost           |           7 |           6 |           | exact  | 1      |       1      | ✗                  |
| Gradient Boosting   | XGBoost           |           5 |           3 |           | exact  | 0.7266 |       0.968  | ✗                  |

*Note.* Exact binomial test when b + c < 25, otherwise continuity-corrected χ². p-values adjusted for 15 comparisons with Benjamini–Hochberg.