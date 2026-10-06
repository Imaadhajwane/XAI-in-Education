**Class distribution of the training partition before and after SMOTE-NC.**

| Class           |   Pre-SMOTE count |   Pre-SMOTE (%) |   Post-SMOTE count |   Post-SMOTE (%) |
|:----------------|------------------:|----------------:|-------------------:|-----------------:|
| At-Risk (1)     |                46 |             5.4 |                799 |               50 |
| Not At-Risk (0) |               799 |            94.6 |                799 |               50 |
| Total           |               845 |           100   |               1598 |              100 |

*Note.* Held-out test set (n = 363, 20 At-Risk) is never resampled. Inside cross-validation SMOTE-NC is re-fitted on each training fold only. Primary analysis uses: No resampling (justified in Table S5); SMOTE-NC is a sensitivity analysis.