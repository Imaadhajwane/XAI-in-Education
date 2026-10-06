**Does synthetic augmentation improve At-Risk detection? (2×5-fold repeated CV, evaluation on real students only)**

| Model               | Training data            |   Synthetic rows / fold | ROC-AUC       | PR-AUC        | Recall@0.5    | F1@0.5        | Brier           | ΔROC-AUC vs real (p)   | ΔPR-AUC vs real (p)   |
|:--------------------|:-------------------------|------------------------:|:--------------|:--------------|:--------------|:--------------|:----------------|:-----------------------|:----------------------|
| Logistic Regression | Real data only           |                       0 | 0.964 ± 0.009 | 0.690 ± 0.089 | 0.529 ± 0.106 | 0.602 ± 0.087 | 0.0297 ± 0.0048 | nan                    | nan                   |
| Logistic Regression | SMOTE-NC → 20% minority  |                     176 | 0.952 ± 0.022 | 0.625 ± 0.111 | 0.642 ± 0.129 | 0.544 ± 0.062 | 0.0429 ± 0.0092 | -0.0126 (0.273)        | -0.0645 (0.175)       |
| Logistic Regression | Copula minority → 20%    |                     176 | 0.954 ± 0.013 | 0.637 ± 0.110 | 0.687 ± 0.145 | 0.515 ± 0.082 | 0.0459 ± 0.0067 | -0.0104 (0.024)        | -0.0525 (0.120)       |
| Logistic Regression | TVAE minority → 20%      |                     176 | 0.957 ± 0.015 | 0.632 ± 0.100 | 0.565 ± 0.150 | 0.541 ± 0.109 | 0.0363 ± 0.0070 | -0.0068 (0.211)        | -0.0577 (0.054)       |
| Logistic Regression | Copula ×2 (both classes) |                     966 | 0.959 ± 0.009 | 0.677 ± 0.089 | 0.445 ± 0.110 | 0.572 ± 0.112 | 0.0303 ± 0.0037 | -0.0049 (0.122)        | -0.0131 (0.458)       |
| Logistic Regression | TVAE ×2 (both classes)   |                     966 | 0.957 ± 0.013 | 0.639 ± 0.103 | 0.521 ± 0.109 | 0.534 ± 0.090 | 0.0344 ± 0.0067 | -0.0071 (0.056)        | -0.0509 (0.090)       |
| Logistic Regression | Copula ×5 (both classes) |                    3865 | 0.949 ± 0.016 | 0.619 ± 0.123 | 0.340 ± 0.125 | 0.456 ± 0.141 | 0.0334 ± 0.0058 | -0.0148 (0.029)        | -0.0710 (0.080)       |
| XGBoost             | Real data only           |                       0 | 0.945 ± 0.019 | 0.593 ± 0.095 | 0.401 ± 0.092 | 0.495 ± 0.099 | 0.0347 ± 0.0049 | nan                    | nan                   |
| XGBoost             | SMOTE-NC → 20% minority  |                     176 | 0.940 ± 0.026 | 0.588 ± 0.113 | 0.462 ± 0.118 | 0.531 ± 0.109 | 0.0356 ± 0.0069 | -0.0053 (0.493)        | -0.0049 (0.908)       |
| XGBoost             | Copula minority → 20%    |                     176 | 0.937 ± 0.022 | 0.533 ± 0.089 | 0.620 ± 0.111 | 0.516 ± 0.062 | 0.0484 ± 0.0058 | -0.0085 (0.409)        | -0.0604 (0.372)       |
| XGBoost             | TVAE minority → 20%      |                     176 | 0.938 ± 0.032 | 0.553 ± 0.134 | 0.462 ± 0.106 | 0.508 ± 0.083 | 0.0373 ± 0.0079 | -0.0071 (0.505)        | -0.0401 (0.515)       |
| XGBoost             | Copula ×2 (both classes) |                     966 | 0.934 ± 0.026 | 0.527 ± 0.107 | 0.295 ± 0.095 | 0.383 ± 0.101 | 0.0378 ± 0.0049 | -0.0115 (0.279)        | -0.0661 (0.345)       |
| XGBoost             | TVAE ×2 (both classes)   |                     966 | 0.939 ± 0.025 | 0.568 ± 0.124 | 0.453 ± 0.112 | 0.525 ± 0.098 | 0.0363 ± 0.0066 | -0.0057 (0.536)        | -0.0251 (0.576)       |
| XGBoost             | Copula ×5 (both classes) |                    3865 | 0.925 ± 0.026 | 0.485 ± 0.104 | 0.310 ± 0.101 | 0.383 ± 0.103 | 0.0423 ± 0.0066 | -0.0204 (0.207)        | -0.1087 (0.094)       |

*Note.* Generators are fitted on the training fold only; synthetic rows never enter a test fold. Δ = mean paired difference vs real-data-only training; p from the Nadeau–Bengio corrected resampled t-test.