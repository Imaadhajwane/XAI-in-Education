**Robustness of PREX-Edu explanations to the choice of predictive model.**

| Comparison     | SHAP output space   |   Global feature-rank ρ |   Mean per-student τ (dimension ranks) | Top-1 dimension agreement (all test)   | Top-1 agreement (flagged At-Risk)   | Global dimension order                                  |
|:---------------|:--------------------|------------------------:|---------------------------------------:|:---------------------------------------|:------------------------------------|:--------------------------------------------------------|
| LR (reference) | log-odds            |                   1     |                                  1     | 100%                                   | 100%                                | Engagement > Lifestyle > Cognitive Load > Participation |
| LR vs XGB      | log-odds            |                   0.945 |                                  0.629 | 70.8%                                  | 81.0%                               | Engagement > Lifestyle > Cognitive Load > Participation |
| LR vs RF       | probability         |                   0.808 |                                  0.713 | 79.3%                                  | 95.2%                               | Engagement > Cognitive Load > Lifestyle > Participation |
| LR vs GB       | log-odds            |                   0.919 |                                  0.757 | 75.5%                                  | 100.0%                              | Engagement > Cognitive Load > Lifestyle > Participation |

*Note.* Same 363 test students. High agreement means the pedagogical message does not hinge on the classifier.