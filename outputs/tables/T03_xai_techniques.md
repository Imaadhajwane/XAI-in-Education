**Comparative analysis of XAI techniques.**

| Method              | Strengths                                                               | Weaknesses                                                      | Best use case                                        |
|:--------------------|:------------------------------------------------------------------------|:----------------------------------------------------------------|:-----------------------------------------------------|
| SHAP                | Axiomatic (Shapley) attribution; consistent global/local view           | Computational cost; feature-level output is technical           | Auditing model behaviour                             |
| LIME                | Simple, instance-level, model-agnostic                                  | Unstable local surrogate; no global consistency                 | Case-level teacher-facing explanation                |
| PDP                 | Visual population-level trends                                          | Ignores interactions; extrapolates under correlation            | Policy-level trend analysis                          |
| PREX-Edu (proposed) | Pedagogically grounded; inherits SHAP local accuracy at dimension level | Depends on the mapping and on the underlying attribution method | Translating risk predictions into intervention areas |