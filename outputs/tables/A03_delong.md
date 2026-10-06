**Pairwise DeLong tests of hold-out ROC-AUC.**

| Model A             | Model B           |   AUC (A) |   AUC (B) |       Z |      p |   p (BH-FDR) | Sig. (FDR α=.05)   |
|:--------------------|:------------------|----------:|----------:|--------:|-------:|-------------:|:-------------------|
| Logistic Regression | SVM               |    0.9832 |    0.9835 | -0.0723 | 0.9424 |       0.9424 | ✗                  |
| Logistic Regression | Random Forest     |    0.9832 |    0.9672 |  2.5137 | 0.0119 |       0.1133 | ✗                  |
| Logistic Regression | Decision Tree     |    0.9832 |    0.9089 |  2.2438 | 0.0248 |       0.1133 | ✗                  |
| Logistic Regression | Gradient Boosting |    0.9832 |    0.9803 |  0.6773 | 0.4982 |       0.5871 | ✗                  |
| Logistic Regression | XGBoost           |    0.9832 |    0.9777 |  0.9293 | 0.3527 |       0.481  | ✗                  |
| SVM                 | Random Forest     |    0.9835 |    0.9672 |  1.8747 | 0.0608 |       0.1141 | ✗                  |
| SVM                 | Decision Tree     |    0.9835 |    0.9089 |  2.0961 | 0.0361 |       0.1133 | ✗                  |
| SVM                 | Gradient Boosting |    0.9835 |    0.9803 |  0.657  | 0.5112 |       0.5871 | ✗                  |
| SVM                 | XGBoost           |    0.9835 |    0.9777 |  1.0294 | 0.3033 |       0.4549 | ✗                  |
| Random Forest       | Decision Tree     |    0.9672 |    0.9089 |  1.8884 | 0.059  |       0.1141 | ✗                  |
| Random Forest       | Gradient Boosting |    0.9672 |    0.9803 | -2.0123 | 0.0442 |       0.1133 | ✗                  |
| Random Forest       | XGBoost           |    0.9672 |    0.9777 | -1.2835 | 0.1993 |       0.3322 | ✗                  |
| Decision Tree       | Gradient Boosting |    0.9089 |    0.9803 | -2.0854 | 0.037  |       0.1133 | ✗                  |
| Decision Tree       | XGBoost           |    0.9089 |    0.9777 | -2.0015 | 0.0453 |       0.1133 | ✗                  |
| Gradient Boosting   | XGBoost           |    0.9803 |    0.9777 |  0.6008 | 0.548  |       0.5871 | ✗                  |

*Note.* Two-sided; Benjamini–Hochberg adjustment over 15 comparisons.