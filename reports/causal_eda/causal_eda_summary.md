# Causal EDA Summary

## Why Causal EDA Is Different From Generic EDA

Generic EDA describes distributions and correlations. Causal EDA asks whether a treatment/control comparison is credible enough to support campaign-impact reasoning. For PromoLift AI, that means inspecting treatment balance, overlap, potential selection bias, subgroup heterogeneity, and leakage risks before interpreting uplift scores.

## Treatment Balance and Covariate Overlap

The balance plots compare treated and control customers on pre-campaign covariates: recency, purchase history, mens/womens indicators, newbie status, channel, and history segment. Because the Hillstrom dataset comes from a marketing experiment, the groups should look similar before treatment. The causal EDA charts help make that assumption visible rather than implicit.

Figures:

- `reports/figures/causal_eda_numeric_overlap.png`
- `reports/figures/causal_eda_categorical_balance.png`

## Propensity Score Overlap

The propensity model predicts treatment assignment using only pre-campaign features. A propensity AUC near 0.5 suggests treatment assignment is difficult to predict from observed customer features, which is consistent with randomized assignment.

- Propensity model AUC: 0.510
- Minimum group share inside common support: 99.99%

Figures and tables:

- `reports/figures/causal_eda_propensity_overlap.png`
- `reports/causal_eda/propensity_overlap_summary.csv`

## Naive vs Stratified-Adjusted Effect Comparison

The naive ATE is the treatment conversion rate minus the control conversion rate. Because Hillstrom is a randomized experiment, this naive estimate is already meaningful. The stratified-adjusted estimates are robustness checks that weight segment-level treatment effects by segment size.

- Naive ATE: 0.68 percentage points

Stratified-adjusted effect estimates:

- channel: 0.68 percentage points (3 segments)
- history_segment: 0.68 percentage points (7 segments)
- newbie: 0.68 percentage points (2 segments)

These are not claimed to be perfect causal adjustments. They are segment-adjusted checks that help show whether the average effect is broadly stable across important business groupings.

Figure and table:

- `reports/figures/causal_eda_naive_vs_adjusted_effect.png`
- `reports/causal_eda/naive_vs_stratified_effects.csv`

## Subgroup Heterogeneity

Average treatment effects can hide important business differences. In this project, subgroup uplift by channel, history segment, and newbie status motivates uplift modeling because the best campaign decision may differ across customer types.

- Strongest observed subgroup uplift: `history_segment = 6) $750 - $1,000` at 2.00 percentage points
- Weakest observed subgroup uplift: `history_segment = 2) $100 - $200` at 0.48 percentage points

Figures:

- `reports/figures/causal_eda_subgroup_uplift.png`
- `reports/figures/causal_eda_heterogeneity_heatmap.png`

## Causal DAG Explanation

The DAG sketch documents the causal reasoning behind the project. Pre-campaign customer history, recency, channel, and segment can influence both treatment assignment checks and conversion outcomes. Treatment assignment can influence conversion. Unobserved factors may also influence conversion, which is why balance checks and careful interpretation matter.

Figure:

- `reports/figures/causal_eda_dag.png`

## Leakage Check

The dataset is a marketing experiment where pre-campaign customer features are used to predict post-campaign conversion. No detailed timestamp-level leakage check can be performed because timestamp columns are not available in the processed dataset. PromoLift AI avoids leakage by excluding post-campaign columns such as `visit`, `conversion`, and `spend` from model features. This limitation is explicitly documented.

## Why This Motivates Uplift Modeling

Causal EDA shows that treatment/control comparison matters, that overlap is strong, and that treatment effects differ across segments. This motivates moving beyond normal conversion prediction toward uplift modeling, where customers are ranked by estimated incremental campaign response.

## Business Interpretation

For marketing teams, causal EDA turns a model project into a decision framework. It clarifies whether the experiment is balanced, whether comparison groups overlap, and whether campaign impact varies across customer groups. That is exactly the evidence needed to reduce discount waste and focus campaign spend on customers whose behavior is most likely to change.
