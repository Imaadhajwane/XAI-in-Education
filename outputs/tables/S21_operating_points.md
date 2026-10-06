**Operating points: thresholds needed to reach target sensitivity, and the resulting workload.**

| Model               | Target recall   |   Threshold |   Recall |   Precision |   Specificity |   Students flagged |   Flagged per true At-Risk |   Missed At-Risk |   Net benefit |
|:--------------------|:----------------|------------:|---------:|------------:|--------------:|-------------------:|---------------------------:|-----------------:|--------------:|
| Logistic Regression | ≥ 95%           |      0.0287 |    0.955 |       0.269 |         0.85  |                234 |                       3.71 |                3 |        0.048  |
| Logistic Regression | ≥ 90%           |      0.0465 |    0.909 |       0.316 |         0.886 |                190 |                       3.17 |                6 |        0.0444 |
| Logistic Regression | ≥ 80%           |      0.1589 |    0.803 |       0.453 |         0.944 |                117 |                       2.21 |               13 |        0.0339 |
| Logistic Regression | ≥ 70%           |      0.227  |    0.712 |       0.485 |         0.956 |                 97 |                       2.06 |               19 |        0.0267 |
| SVM                 | ≥ 95%           |      0.0343 |    0.955 |       0.234 |         0.82  |                269 |                       4.27 |                3 |        0.0461 |
| SVM                 | ≥ 90%           |      0.0601 |    0.909 |       0.291 |         0.872 |                206 |                       3.43 |                6 |        0.0419 |
| SVM                 | ≥ 80%           |      0.1297 |    0.803 |       0.396 |         0.929 |                134 |                       2.53 |               13 |        0.0339 |
| SVM                 | ≥ 70%           |      0.2108 |    0.712 |       0.452 |         0.95  |                104 |                       2.21 |               19 |        0.0263 |
| Random Forest       | ≥ 95%           |      0.0604 |    0.955 |       0.19  |         0.764 |                332 |                       5.27 |                3 |        0.0378 |
| Random Forest       | ≥ 90%           |      0.0745 |    0.909 |       0.211 |         0.804 |                284 |                       4.73 |                6 |        0.0347 |
| Random Forest       | ≥ 80%           |      0.103  |    0.803 |       0.273 |         0.877 |                194 |                       3.66 |               13 |        0.0305 |
| Random Forest       | ≥ 70%           |      0.1397 |    0.712 |       0.359 |         0.926 |                131 |                       2.79 |               19 |        0.0276 |
| Decision Tree       | ≥ 95%           |      0.0005 |    0.955 |       0.078 |         0.347 |                809 |                      12.84 |                3 |        0.0518 |
| Decision Tree       | ≥ 90%           |      0.01   |    0.909 |       0.124 |         0.629 |                484 |                       8.07 |                6 |        0.0461 |
| Decision Tree       | ≥ 80%           |      0.0751 |    0.803 |       0.272 |         0.876 |                195 |                       3.68 |               13 |        0.0343 |
| Decision Tree       | ≥ 70%           |      0.147  |    0.712 |       0.392 |         0.936 |                120 |                       2.55 |               19 |        0.0285 |
| Gradient Boosting   | ≥ 95%           |      0.0236 |    0.955 |       0.185 |         0.757 |                341 |                       5.41 |                3 |        0.0466 |
| Gradient Boosting   | ≥ 90%           |      0.0518 |    0.909 |       0.287 |         0.87  |                209 |                       3.48 |                6 |        0.0429 |
| Gradient Boosting   | ≥ 80%           |      0.0765 |    0.803 |       0.333 |         0.907 |                159 |                       3    |               13 |        0.0366 |
| Gradient Boosting   | ≥ 70%           |      0.126  |    0.712 |       0.409 |         0.94  |                115 |                       2.45 |               19 |        0.0308 |
| XGBoost             | ≥ 95%           |      0.0141 |    0.955 |       0.209 |         0.792 |                301 |                       4.78 |                3 |        0.0493 |
| XGBoost             | ≥ 90%           |      0.0365 |    0.909 |       0.308 |         0.882 |                195 |                       3.25 |                6 |        0.0454 |
| XGBoost             | ≥ 80%           |      0.0711 |    0.803 |       0.405 |         0.932 |                131 |                       2.47 |               13 |        0.0389 |
| XGBoost             | ≥ 70%           |      0.1117 |    0.712 |       0.427 |         0.945 |                110 |                       2.34 |               19 |        0.0323 |

*Note.* Computed on out-of-fold probabilities for all 1,208 students (66 At-Risk). 'Flagged per true At-Risk' is the number of students a school must follow up per correctly identified At-Risk student. Note these thresholds are chosen post hoc on the pooled OOF predictions — use for planning, not as unbiased estimates.