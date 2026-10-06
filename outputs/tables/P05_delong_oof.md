**DeLong tests on per-student out-of-fold probabilities (n = 1208).**

| Model A             | Model B           |   AUC (A) |   AUC (B) |      Z |      p |   p (BH-FDR) | Sig.   |
|:--------------------|:------------------|----------:|----------:|-------:|-------:|-------------:|:-------|
| Logistic Regression | SVM               |    0.9642 |    0.96   |  1.922 | 0.0546 |       0.0697 | ✗      |
| Logistic Regression | Random Forest     |    0.9642 |    0.9363 |  4.06  | 0      |       0.0007 | ✓      |
| Logistic Regression | Decision Tree     |    0.9642 |    0.889  |  3.412 | 0.0006 |       0.0028 | ✓      |
| Logistic Regression | Gradient Boosting |    0.9642 |    0.9442 |  2.726 | 0.0064 |       0.0125 | ✓      |
| Logistic Regression | XGBoost           |    0.9642 |    0.9507 |  1.913 | 0.0558 |       0.0697 | ✗      |
| SVM                 | Random Forest     |    0.96   |    0.9363 |  3.567 | 0.0004 |       0.0027 | ✓      |
| SVM                 | Decision Tree     |    0.96   |    0.889  |  3.372 | 0.0007 |       0.0028 | ✓      |
| SVM                 | Gradient Boosting |    0.96   |    0.9442 |  2.192 | 0.0284 |       0.0426 | ✓      |
| SVM                 | XGBoost           |    0.96   |    0.9507 |  1.436 | 0.1511 |       0.1585 | ✗      |
| Random Forest       | Decision Tree     |    0.9363 |    0.889  |  2.28  | 0.0226 |       0.0377 | ✓      |
| Random Forest       | Gradient Boosting |    0.9363 |    0.9442 | -1.475 | 0.1403 |       0.1585 | ✗      |
| Random Forest       | XGBoost           |    0.9363 |    0.9507 | -2.784 | 0.0054 |       0.0125 | ✓      |
| Decision Tree       | Gradient Boosting |    0.889  |    0.9442 | -2.712 | 0.0067 |       0.0125 | ✓      |
| Decision Tree       | XGBoost           |    0.889  |    0.9507 | -3.243 | 0.0012 |       0.0036 | ✓      |
| Gradient Boosting   | XGBoost           |    0.9442 |    0.9507 | -1.41  | 0.1585 |       0.1585 | ✗      |

*Note.* Benjamini–Hochberg adjustment over 15 comparisons.