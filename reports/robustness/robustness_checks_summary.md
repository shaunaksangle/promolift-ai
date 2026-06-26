# Robustness Checks Summary

## Why Robustness Checks Were Added

This robustness step is not intended to search for a flattering result. It checks whether the project's conclusion is stable across alternative uplift methods, interpretable segments, and treatment definitions.

## X-Learner Robustness Result

The X-Learner is an exploratory robustness method built with scikit-learn. It uses separate treated/control outcome models, imputed treatment effects, treatment-effect regressors, and propensity-weighted combination.

- X-Learner top-decile observed uplift: 0.64%
- X-Learner top-30% estimated incremental conversions: 21.4
- X-Learner Qini coefficient: -0.730
- X-Learner interpretation: weak or noisy ranking signal

## Method Agreement

The strongest method by top-30% estimated incremental conversions is `X-Learner`. Agreement across methods should be interpreted cautiously because conversions are rare and uplift ranking can be noisy.

See:

- `reports/robustness/uplift_method_robustness_comparison.csv`
- `reports/figures/uplift_method_robustness_comparison.png`

## Segment Heterogeneity Findings

Direct segment checks look for observed treatment-effect differences in business-readable groups rather than relying only on model ranking.

- Strongest observed segment uplift: `history_segment = 6) $750 - $1,000` at 2.00%
- Weakest observed segment uplift: `history_segment = 2) $100 - $200` at 0.48%

These segment-level differences are exploratory. Small segments can be noisy, so caution flags are included in the CSV output.

## Treatment Definition Findings

- Mens E-Mail vs No E-Mail: observed ATE 0.68%, conversion p-value 0.0000, secondary visit lift 7.66%
- Womens E-Mail vs No E-Mail: observed ATE 0.31%, conversion p-value 0.0002, secondary visit lift 4.52%
- Any E-Mail vs No E-Mail: observed ATE 0.50%, conversion p-value 0.0000, secondary visit lift 6.09%

The main project keeps Mens E-Mail vs No E-Mail as the clean primary setup. Womens E-Mail and Any E-Mail comparisons are exploratory robustness checks.

## Final Honest Conclusion

If fine-grained targeting does not strongly beat random targeting, that is still a valid and useful business finding. It suggests that the campaign may create broad lift, while individual-level targeting needs stronger signal or richer features.

The robustness checks should be used to pressure-test the project narrative, not to replace the main uplift model or force a better-looking result.
