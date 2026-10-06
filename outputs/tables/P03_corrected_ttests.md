**Pairwise model comparison with the Nadeau–Bengio corrected repeated-CV t-test.**

| Metric                  | Model A             | Model B           |   Mean diff (A−B) |      t |   df |      p |   p (BH-FDR) | Sig.   |
|:------------------------|:--------------------|:------------------|------------------:|-------:|-----:|-------:|-------------:|:-------|
| ROC-AUC                 | Logistic Regression | SVM               |            0.0052 |  1.698 |   99 | 0.0926 |       0.1157 | ✗      |
| ROC-AUC                 | Logistic Regression | Random Forest     |            0.0312 |  3.414 |   99 | 0.0009 |       0.0023 | ✓      |
| ROC-AUC                 | Logistic Regression | Decision Tree     |            0.1625 |  5.266 |   99 | 0      |       0      | ✓      |
| ROC-AUC                 | Logistic Regression | Gradient Boosting |            0.0247 |  2.413 |   99 | 0.0177 |       0.0331 | ✓      |
| ROC-AUC                 | Logistic Regression | XGBoost           |            0.0156 |  1.808 |   99 | 0.0736 |       0.1003 | ✗      |
| ROC-AUC                 | SVM                 | Random Forest     |            0.026  |  2.738 |   99 | 0.0073 |       0.0157 | ✓      |
| ROC-AUC                 | SVM                 | Decision Tree     |            0.1573 |  5.145 |   99 | 0      |       0      | ✓      |
| ROC-AUC                 | SVM                 | Gradient Boosting |            0.0195 |  1.987 |   99 | 0.0496 |       0.0827 | ✗      |
| ROC-AUC                 | SVM                 | XGBoost           |            0.0104 |  1.261 |   99 | 0.2102 |       0.2253 | ✗      |
| ROC-AUC                 | Random Forest       | Decision Tree     |            0.1313 |  4.575 |   99 | 0      |       0      | ✓      |
| ROC-AUC                 | Random Forest       | Gradient Boosting |           -0.0065 | -0.796 |   99 | 0.4281 |       0.4281 | ✗      |
| ROC-AUC                 | Random Forest       | XGBoost           |           -0.0156 | -1.879 |   99 | 0.0632 |       0.0948 | ✗      |
| ROC-AUC                 | Decision Tree       | Gradient Boosting |           -0.1378 | -4.796 |   99 | 0      |       0      | ✓      |
| ROC-AUC                 | Decision Tree       | XGBoost           |           -0.1468 | -5.072 |   99 | 0      |       0      | ✓      |
| ROC-AUC                 | Gradient Boosting   | XGBoost           |           -0.009  | -1.27  |   99 | 0.2072 |       0.2253 | ✗      |
| PR-AUC (Avg. precision) | Logistic Regression | SVM               |            0.0327 |  1.769 |   99 | 0.0799 |       0.109  | ✗      |
| PR-AUC (Avg. precision) | Logistic Regression | Random Forest     |            0.1302 |  2.954 |   99 | 0.0039 |       0.0084 | ✓      |
| PR-AUC (Avg. precision) | Logistic Regression | Decision Tree     |            0.3354 |  6.358 |   99 | 0      |       0      | ✓      |
| PR-AUC (Avg. precision) | Logistic Regression | Gradient Boosting |            0.113  |  3.143 |   99 | 0.0022 |       0.0055 | ✓      |
| PR-AUC (Avg. precision) | Logistic Regression | XGBoost           |            0.0653 |  1.823 |   99 | 0.0713 |       0.1069 | ✗      |
| PR-AUC (Avg. precision) | SVM                 | Random Forest     |            0.0975 |  2.246 |   99 | 0.0269 |       0.0505 | ✗      |
| PR-AUC (Avg. precision) | SVM                 | Decision Tree     |            0.3027 |  5.611 |   99 | 0      |       0      | ✓      |
| PR-AUC (Avg. precision) | SVM                 | Gradient Boosting |            0.0803 |  2.122 |   99 | 0.0363 |       0.0605 | ✗      |
| PR-AUC (Avg. precision) | SVM                 | XGBoost           |            0.0326 |  0.91  |   99 | 0.3651 |       0.3912 | ✗      |
| PR-AUC (Avg. precision) | Random Forest       | Decision Tree     |            0.2052 |  3.892 |   99 | 0.0002 |       0.0005 | ✓      |
| PR-AUC (Avg. precision) | Random Forest       | Gradient Boosting |           -0.0172 | -0.438 |   99 | 0.6621 |       0.6621 | ✗      |
| PR-AUC (Avg. precision) | Random Forest       | XGBoost           |           -0.0649 | -1.47  |   99 | 0.1446 |       0.1808 | ✗      |
| PR-AUC (Avg. precision) | Decision Tree       | Gradient Boosting |           -0.2224 | -4.442 |   99 | 0      |       0.0001 | ✓      |
| PR-AUC (Avg. precision) | Decision Tree       | XGBoost           |           -0.2701 | -5.071 |   99 | 0      |       0      | ✓      |
| PR-AUC (Avg. precision) | Gradient Boosting   | XGBoost           |           -0.0477 | -1.398 |   99 | 0.1651 |       0.1905 | ✗      |
| MCC                     | Logistic Regression | SVM               |            0.0435 |  1.008 |   99 | 0.316  |       0.3647 | ✗      |
| MCC                     | Logistic Regression | Random Forest     |            0.5543 |  8.708 |   99 | 0      |       0      | ✓      |
| MCC                     | Logistic Regression | Decision Tree     |            0.2306 |  3.115 |   99 | 0.0024 |       0.006  | ✓      |
| MCC                     | Logistic Regression | Gradient Boosting |            0.1252 |  2.16  |   99 | 0.0332 |       0.0553 | ✗      |
| MCC                     | Logistic Regression | XGBoost           |            0.0498 |  0.905 |   99 | 0.3678 |       0.3941 | ✗      |
| MCC                     | SVM                 | Random Forest     |            0.5108 |  7.488 |   99 | 0      |       0      | ✓      |
| MCC                     | SVM                 | Decision Tree     |            0.1871 |  2.321 |   99 | 0.0224 |       0.0479 | ✓      |
| MCC                     | SVM                 | Gradient Boosting |            0.0817 |  1.374 |   99 | 0.1726 |       0.2157 | ✗      |
| MCC                     | SVM                 | XGBoost           |            0.0063 |  0.098 |   99 | 0.9225 |       0.9225 | ✗      |
| MCC                     | Random Forest       | Decision Tree     |           -0.3237 | -4.302 |   99 | 0      |       0.0001 | ✓      |
| MCC                     | Random Forest       | Gradient Boosting |           -0.4291 | -6.676 |   99 | 0      |       0      | ✓      |
| MCC                     | Random Forest       | XGBoost           |           -0.5045 | -7.976 |   99 | 0      |       0      | ✓      |
| MCC                     | Decision Tree       | Gradient Boosting |           -0.1054 | -1.448 |   99 | 0.1509 |       0.2057 | ✗      |
| MCC                     | Decision Tree       | XGBoost           |           -0.1808 | -2.19  |   99 | 0.0308 |       0.0553 | ✗      |
| MCC                     | Gradient Boosting   | XGBoost           |           -0.0754 | -1.457 |   99 | 0.1483 |       0.2057 | ✗      |

*Note.* Corrects for the overlap of training sets across CV folds (variance inflated by 1/J + n_test/n_train). Benjamini–Hochberg adjustment within each metric (15 comparisons).