**Reliability of the six classifiers: Cohen's κ, MCC and Brier score on the held-out test set.**

| Model               |   Accuracy (CV mean) |   Accuracy (test) |   Cohen's κ (test) | Agreement level   |   MCC (test) |   Brier score (test) |   Brier skill score |
|:--------------------|---------------------:|------------------:|-------------------:|:------------------|-------------:|---------------------:|--------------------:|
| Logistic Regression |               0.9539 |            0.9697 |             0.6515 | Substantial       |       0.6682 |               0.0217 |              0.583  |
| SVM                 |               0.9503 |            0.9614 |             0.5155 | Moderate          |       0.5495 |               0.0265 |              0.4912 |
| Random Forest       |               0.9456 |            0.9449 |             0      | Slight            |       0      |               0.037  |              0.2898 |
| Decision Tree       |               0.9266 |            0.9559 |             0.507  | Moderate          |       0.5159 |               0.0361 |              0.3075 |
| Gradient Boosting   |               0.9491 |            0.9587 |             0.5248 | Moderate          |       0.5382 |               0.0275 |              0.4719 |
| XGBoost             |               0.9574 |            0.9532 |             0.4902 | Moderate          |       0.4958 |               0.033  |              0.3665 |

*Note.* Agreement levels follow Landis & Koch (1977). Brier skill score > 0 means better than always predicting the base rate.