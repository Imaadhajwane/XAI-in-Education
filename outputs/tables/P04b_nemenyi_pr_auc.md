**Nemenyi post-hoc p-values (PR-AUC (Avg. precision)).**

| model               |   Logistic Regression |    SVM |   Random Forest |   Decision Tree |   Gradient Boosting |   XGBoost |
|:--------------------|----------------------:|-------:|----------------:|----------------:|--------------------:|----------:|
| Logistic Regression |                1      | 0.03   |          0      |               0 |              0      |    0.0004 |
| SVM                 |                0.03   | 1      |          0      |               0 |              0.0002 |    0.8505 |
| Random Forest       |                0      | 0      |          1      |               0 |              0.8978 |    0.0004 |
| Decision Tree       |                0      | 0      |          0      |               1 |              0      |    0      |
| Gradient Boosting   |                0      | 0.0002 |          0.8978 |               0 |              1      |    0.0211 |
| XGBoost             |                0.0004 | 0.8505 |          0.0004 |               0 |              0.0211 |    1      |