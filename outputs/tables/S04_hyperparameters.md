**Hyper-parameters selected by stratified 10-fold GridSearchCV on the training partition.**

| Model               | Selected hyper-parameters                         |   Inner-CV average_precision |
|:--------------------|:--------------------------------------------------|-----------------------------:|
| Logistic Regression | C=100                                             |                       0.6527 |
| SVM                 | C=0.1, gamma=0.01                                 |                       0.5855 |
| Random Forest       | max_depth=None, min_samples_leaf=5                |                       0.5358 |
| Decision Tree       | max_depth=7, min_samples_leaf=10                  |                       0.2205 |
| Gradient Boosting   | learning_rate=0.05, max_depth=2, n_estimators=200 |                       0.5633 |
| XGBoost             | learning_rate=0.1, max_depth=5                    |                       0.6127 |

*Note.* Optimisation criterion: average_precision (PR-AUC), appropriate for a 5.5% minority class.