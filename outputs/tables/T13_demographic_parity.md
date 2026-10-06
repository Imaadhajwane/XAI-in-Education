**Demographic parity of At-Risk predictions by gender and family income (hold-out test set).**

| Attribute    | Subgroup          |   N (test) |   N At-Risk (true) |   Base rate |   P(ŷ=1 | group) | Parity diff. vs reference   | Disparate-impact ratio   | TPR (recall) in group   |
|:-------------|:------------------|-----------:|-------------------:|------------:|-----------------:|:----------------------------|:-------------------------|:------------------------|
| Gender       | Female            |        192 |                 14 |      0.0729 |           0.0729 | — (reference)               | —                        | 0.714                   |
| Gender       | Male              |        156 |                  6 |      0.0385 |           0.0385 | -0.0345                     | 0.527                    | 0.667                   |
| Gender       | Prefer not to say |         15 |                  0 |      0      |           0.0667 | -0.0063                     | 0.914                    | n/a                     |
| Income range | High              |         70 |                  4 |      0.0571 |           0.0571 | — (reference)               | —                        | 0.5                     |
| Income range | Medium            |        203 |                 10 |      0.0493 |           0.0493 | -0.0079                     | 0.862                    | 0.7                     |
| Income range | Low               |         90 |                  6 |      0.0667 |           0.0778 | 0.0206                      | 1.361                    | 0.833                   |

*Note.* P(ŷ=1 | group) is the share of the subgroup flagged At-Risk (threshold 0.339, F1-optimal on training CV). Disparate-impact ratio < 0.8 or > 1.25 would breach the four-fifths rule. With ≤ 20 true At-Risk students in the test set, subgroup TPRs are highly uncertain; see script 14 for the full-sample audit with CIs.