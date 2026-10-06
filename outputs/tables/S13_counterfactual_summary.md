**Actionable counterfactual explanations for flagged At-Risk students (greedy minimal-cost search).**

| Measure                                | Value     |
|:---------------------------------------|:----------|
| Students                               | 21        |
| Valid counterfactuals found            | 21 (100%) |
| Median cost (SD units)                 | 0.72      |
| Median number of features changed      | 1.0       |
| Main CF dimension = PREX top dimension | 95.2%     |

*Note.* Target: P(At-Risk) below the decision threshold 0.339. Only the six actionable features may change; demographics are immutable. Step sizes: 0.25 SD for continuous features, 1 point for Likert items, Yes for attendance. Cost = sum of standardised step sizes.