**Class-wise performance under nested repeated cross-validation.**

| Model               | Threshold   | Class       | Precision (%)   | Recall (%)    | F1 (%)        |   Support (per repeat) |
|:--------------------|:------------|:------------|:----------------|:--------------|:--------------|-----------------------:|
| Logistic Regression | 0.5         | Not At-Risk | 97.28 ± 1.10    | 98.68 ± 1.07  | 97.97 ± 0.74  |                   1142 |
| Logistic Regression | 0.5         | At-Risk     | 72.01 ± 20.11   | 52.19 ± 18.84 | 58.49 ± 16.33 |                     66 |
| Logistic Regression | tuned       | Not At-Risk | 97.67 ± 1.05    | 97.45 ± 1.96  | 97.54 ± 0.97  |                   1142 |
| Logistic Regression | tuned       | At-Risk     | 62.15 ± 19.45   | 59.45 ± 18.38 | 58.23 ± 13.61 |                     66 |
| SVM                 | 0.5         | Not At-Risk | 96.84 ± 1.09    | 99.06 ± 0.95  | 97.93 ± 0.64  |                   1142 |
| SVM                 | 0.5         | At-Risk     | 74.90 ± 23.83   | 43.79 ± 19.72 | 52.66 ± 18.96 |                     66 |
| SVM                 | tuned       | Not At-Risk | 97.72 ± 1.16    | 96.97 ± 1.98  | 97.33 ± 1.04  |                   1142 |
| SVM                 | tuned       | At-Risk     | 56.47 ± 19.11   | 60.48 ± 20.16 | 56.37 ± 15.58 |                     66 |
| Random Forest       | 0.5         | Not At-Risk | 94.60 ± 0.40    | 100.00 ± 0.00 | 97.22 ± 0.21  |                   1142 |
| Random Forest       | 0.5         | At-Risk     | 8.00 ± 27.27    | 1.14 ± 3.90   | 2.00 ± 6.82   |                     66 |
| Random Forest       | tuned       | Not At-Risk | 97.33 ± 1.10    | 97.08 ± 1.66  | 97.19 ± 0.91  |                   1142 |
| Random Forest       | tuned       | At-Risk     | 53.85 ± 18.03   | 53.62 ± 19.88 | 51.71 ± 15.25 |                     66 |
| Decision Tree       | 0.5         | Not At-Risk | 96.13 ± 1.00    | 97.84 ± 1.70  | 96.97 ± 0.93  |                   1142 |
| Decision Tree       | 0.5         | At-Risk     | 48.67 ± 27.72   | 31.62 ± 18.20 | 36.19 ± 18.58 |                     66 |
| Decision Tree       | tuned       | Not At-Risk | 96.77 ± 1.24    | 96.16 ± 2.79  | 96.44 ± 1.40  |                   1142 |
| Decision Tree       | tuned       | At-Risk     | 44.11 ± 22.06   | 44.07 ± 22.23 | 40.99 ± 17.13 |                     66 |
| Gradient Boosting   | 0.5         | Not At-Risk | 96.37 ± 0.93    | 98.92 ± 0.84  | 97.63 ± 0.63  |                   1142 |
| Gradient Boosting   | 0.5         | At-Risk     | 67.14 ± 24.03   | 35.57 ± 16.04 | 44.79 ± 16.80 |                     66 |
| Gradient Boosting   | tuned       | Not At-Risk | 97.44 ± 1.18    | 96.73 ± 2.28  | 97.06 ± 1.12  |                   1142 |
| Gradient Boosting   | tuned       | At-Risk     | 52.27 ± 14.78   | 55.83 ± 20.36 | 51.94 ± 13.92 |                     66 |
| XGBoost             | 0.5         | Not At-Risk | 96.79 ± 1.06    | 99.03 ± 0.90  | 97.89 ± 0.70  |                   1142 |
| XGBoost             | 0.5         | At-Risk     | 74.10 ± 22.57   | 43.12 ± 19.09 | 52.20 ± 18.41 |                     66 |
| XGBoost             | tuned       | Not At-Risk | 97.51 ± 1.11    | 96.82 ± 2.02  | 97.15 ± 1.01  |                   1142 |
| XGBoost             | tuned       | At-Risk     | 54.64 ± 16.97   | 57.14 ± 19.04 | 53.54 ± 13.39 |                     66 |

*Note.* Mean ± SD over outer folds. Support = number of students of that class, each scored once per repeat.