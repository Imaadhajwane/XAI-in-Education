**Friedman test on fold-wise accuracy ranks (10 folds, six models).**

|   Rank | Model               |   Mean rank |   Friedman χ² |   df |       p |   Nemenyi CD (α=.05) |
|-------:|:--------------------|------------:|--------------:|-----:|--------:|---------------------:|
|      1 | XGBoost             |        2.4  |        17.314 |    5 | 0.00394 |                2.384 |
|      2 | Logistic Regression |        3    |        17.314 |    5 | 0.00394 |                2.384 |
|      3 | SVM                 |        3.2  |        17.314 |    5 | 0.00394 |                2.384 |
|      4 | Gradient Boosting   |        3.25 |        17.314 |    5 | 0.00394 |                2.384 |
|      5 | Random Forest       |        3.8  |        17.314 |    5 | 0.00394 |                2.384 |
|      6 | Decision Tree       |        5.35 |        17.314 |    5 | 0.00394 |                2.384 |

*Note.* Models whose mean ranks differ by more than the critical difference (CD) differ significantly (Nemenyi post-hoc).