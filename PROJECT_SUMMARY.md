# PromoLift AI Project Summary

## 30-Second Explanation

PromoLift AI is a causal uplift modeling project for smarter coupon and email targeting. Instead of predicting who is most likely to buy, it estimates who is more likely to buy because of receiving the campaign. The project uses the Hillstrom Email Marketing experiment, compares `Mens E-Mail` against `No E-Mail`, validates treatment/control balance, and builds a Streamlit dashboard for business storytelling.

## 2-Minute Explanation

Many companies target coupons using normal machine learning models that predict conversion probability. The problem is that high-probability buyers may purchase anyway, so giving them a coupon can waste margin.

PromoLift AI reframes the problem as a treatment-effect question. The treatment group received `Mens E-Mail`, the control group received `No E-Mail`, and the outcome is conversion. The project first compares conversion rates between treatment and control, then builds a baseline conversion model to show why standard classification is incomplete. It then uses T-Learner and S-Learner uplift models to rank customers by estimated incremental response.

Causal validation is included to make the interpretation more responsible. The propensity model AUC is 0.510 and the largest absolute standardized mean difference is 0.016, which suggests the experiment is well balanced. The final dashboard translates the analysis into a clear recommendation: target high-uplift customers rather than everyone or only likely buyers.

## Technical Explanation

The project pipeline starts by downloading and preprocessing the Hillstrom dataset into a binary treatment dataset. It then performs EDA, including treatment/control conversion lift, visit lift, spend analysis, and segment-level uplift tables.

Causal EDA makes the project stronger than generic EDA by checking whether treated and control customers look comparable before treatment, whether propensity scores overlap, whether naive effects are robust to simple segment adjustment, and whether treatment effects differ across subgroups. It also documents a causal DAG and leakage limitations, which makes the modeling story more credible for real decision-making.

The baseline model is a normal conversion classifier using pre-campaign features only. It demonstrates that ranking likely converters is not the same as estimating incremental campaign impact.

The uplift modeling step compares T-Learner and S-Learner approaches. The T-Learner trains separate treatment and control outcome models, while the S-Learner trains one model with treatment as a feature and scores each customer twice. Model selection prioritizes estimated incremental conversions at the top 30% targeting threshold, which is more stable for rare conversion campaigns than only using top-decile observed uplift.

Advanced uplift evaluation makes the model assessment more mature. The Qini curve matters because it shows whether uplift-ranked targeting beats random targeting. Calibration matters because predicted uplift magnitudes can be larger than the validated campaign-level ATE, especially with rare conversions and imperfect probability calibration. Even when raw predicted uplift is not perfectly calibrated, the ranking can still be useful if high-ranked groups show stronger observed incremental response and better policy value.

Robustness checks make the conclusion more credible by asking whether the same story appears under an exploratory X-Learner, direct segment-level checks, and alternative treatment definitions. Weak uplift is still useful because it may show that the campaign creates broad lift but individual-level targeting needs richer features or stronger signal. Not tuning for a better Qini curve is the responsible choice because the goal is honest decision support, not forcing a flattering result.

The causal validation step estimates the observed average treatment effect, checks covariate balance with SMD, inspects propensity scores, and optionally uses DoWhy for an additional causal estimation/refutation layer.

## Business Impact Explanation

The project supports a more efficient campaign strategy. Rather than sending coupons to all customers or only to customers with high purchase probability, the business can prioritize customers with high predicted incremental response.

This can reduce wasted discounts, improve campaign ROI, and make marketing decisions more defensible because the recommendation is grounded in treatment/control comparisons instead of only predictive accuracy.

## Possible Interview Questions and Answers

### Why not just use normal classification?

Normal classification predicts who is likely to buy. It does not tell us whether the campaign caused the purchase. A customer can have a high conversion probability and still be a bad coupon target if they would buy without the coupon.

### What is uplift modeling?

Uplift modeling estimates the incremental impact of a treatment. In this project, uplift is the difference between the predicted conversion probability if a customer receives the email and the predicted conversion probability if they do not.

### What is treatment?

Treatment is the intervention being tested. Here, treatment means the customer received the `Mens E-Mail` campaign.

### What is control?

Control is the comparison group that did not receive the intervention. Here, control means the customer received `No E-Mail`.

### What is ATE?

ATE means average treatment effect. It is the average difference in outcomes between the treatment and control groups. In this project, the observed ATE is about +0.68 percentage points for conversion.

### Why is accuracy misleading here?

Conversion is rare, so a model can achieve high accuracy by predicting that most customers will not convert. Metrics such as ROC-AUC, average precision, uplift by decile, and policy value are more useful for this business problem.

### Why did you use T-Learner and S-Learner?

They are beginner-friendly uplift modeling approaches that are easy to explain. The T-Learner models treatment and control outcomes separately, while the S-Learner uses one model and includes treatment assignment as a feature.

### Why was T-Learner selected?

T-Learner was selected because it performed better on the top 30% estimated incremental conversions metric. That metric is more business-stable for campaign targeting than relying only on top-decile observed uplift, especially when conversions are rare.

### What does propensity AUC 0.510 mean?

The propensity model tried to predict whether a customer was in the treatment group. An AUC close to 0.5 means treatment assignment is difficult to predict from observed features, which is consistent with a balanced experiment.

### What does SMD 0.016 mean?

SMD means standardized mean difference. The largest absolute SMD of 0.016 indicates very small observed covariate imbalance between treatment and control groups.

### What are the project limitations?

Individual counterfactual outcomes are not directly observed. Uplift scores should be interpreted as ranking and decision-support signals, not perfect individual causal truths. The project also focuses on one treatment comparison in the first version: `Mens E-Mail` vs `No E-Mail`.
