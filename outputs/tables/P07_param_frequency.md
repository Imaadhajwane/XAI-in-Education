**Most frequently selected hyper-parameters across outer folds (top 3 per model).**

| Model               | Selected hyper-parameters                                                   |   Outer folds |   Share of outer folds (%) |
|:--------------------|:----------------------------------------------------------------------------|--------------:|---------------------------:|
| Decision Tree       | {'clf__max_depth': 7, 'clf__min_samples_leaf': 10}                          |            32 |                         32 |
| Decision Tree       | {'clf__max_depth': 5, 'clf__min_samples_leaf': 5}                           |            22 |                         22 |
| Decision Tree       | {'clf__max_depth': 5, 'clf__min_samples_leaf': 10}                          |            16 |                         16 |
| Gradient Boosting   | {'clf__learning_rate': 0.05, 'clf__max_depth': 2, 'clf__n_estimators': 200} |            38 |                         38 |
| Gradient Boosting   | {'clf__learning_rate': 0.1, 'clf__max_depth': 2, 'clf__n_estimators': 100}  |            30 |                         30 |
| Gradient Boosting   | {'clf__learning_rate': 0.05, 'clf__max_depth': 3, 'clf__n_estimators': 200} |             9 |                          9 |
| Logistic Regression | {'clf__C': 100}                                                             |            83 |                         83 |
| Logistic Regression | {'clf__C': 10}                                                              |            17 |                         17 |
| Random Forest       | {'clf__max_depth': None, 'clf__min_samples_leaf': 5}                        |            36 |                         36 |
| Random Forest       | {'clf__max_depth': 10, 'clf__min_samples_leaf': 5}                          |            35 |                         35 |
| Random Forest       | {'clf__max_depth': 10, 'clf__min_samples_leaf': 1}                          |            15 |                         15 |
| SVM                 | {'clf__C': 10, 'clf__gamma': 0.01}                                          |            44 |                         44 |
| SVM                 | {'clf__C': 1, 'clf__gamma': 0.01}                                           |            28 |                         28 |
| SVM                 | {'clf__C': 0.1, 'clf__gamma': 0.01}                                         |            27 |                         27 |
| XGBoost             | {'clf__learning_rate': 0.05, 'clf__max_depth': 3}                           |            43 |                         43 |
| XGBoost             | {'clf__learning_rate': 0.1, 'clf__max_depth': 3}                            |            23 |                         23 |
| XGBoost             | {'clf__learning_rate': 0.05, 'clf__max_depth': 5}                           |            19 |                         19 |