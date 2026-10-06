**Class-wise precision, recall and F1 for both classes.**

| Model               | Class       | Precision (%)   | Recall (%)    | F1 (%)        |   Test precision (%) |   Test recall (%) |   Test F1 (%) |   Test support |
|:--------------------|:------------|:----------------|:--------------|:--------------|---------------------:|------------------:|--------------:|---------------:|
| Logistic Regression | Not At-Risk | 96.81 ± 1.72    | 98.38 ± 1.77  | 97.58 ± 1.45  |                97.43 |             99.42 |         98.41 |            343 |
|                     | At-Risk     | 63.45 ± 39.64   | 44.00 ± 29.80 | 49.40 ± 31.62 |                84.62 |             55    |         66.67 |             20 |
| SVM                 | Not At-Risk | 95.45 ± 1.33    | 99.50 ± 0.87  | 97.43 ± 0.86  |                96.6  |             99.42 |         97.99 |            343 |
|                     | At-Risk     | 53.33 ± 50.18   | 18.00 ± 22.63 | 25.07 ± 27.12 |                80    |             40    |         53.33 |             20 |
| Random Forest       | Not At-Risk | 94.56 ± 0.59    | 100.00 ± 0.00 | 97.20 ± 0.31  |                94.49 |            100    |         97.17 |            343 |
|                     | At-Risk     | 0.00 ± 0.00     | 0.00 ± 0.00   | 0.00 ± 0.00   |                 0    |              0    |          0    |             20 |
| Decision Tree       | Not At-Risk | 95.24 ± 1.24    | 97.12 ± 2.21  | 96.15 ± 0.95  |                96.85 |             98.54 |         97.69 |            343 |
|                     | At-Risk     | 18.00 ± 18.41   | 16.50 ± 16.51 | 16.42 ± 15.74 |                64.29 |             45    |         52.94 |             20 |
| Gradient Boosting   | Not At-Risk | 95.89 ± 1.12    | 98.87 ± 1.38  | 97.35 ± 0.72  |                96.86 |             98.83 |         97.84 |            343 |
|                     | At-Risk     | 60.00 ± 40.22   | 26.00 ± 19.41 | 33.50 ± 21.47 |                69.23 |             45    |         54.55 |             20 |
| XGBoost             | Not At-Risk | 96.48 ± 1.41    | 99.12 ± 1.32  | 97.78 ± 0.98  |                96.84 |             98.25 |         97.54 |            343 |
|                     | At-Risk     | 70.00 ± 35.83   | 37.50 ± 23.36 | 46.36 ± 25.49 |                60    |             45    |         51.43 |             20 |

*Note.* '(%)' columns: mean ± SD over 10 CV folds of the training partition. 'Test' columns: single evaluation on the held-out set; support = number of test students per class.