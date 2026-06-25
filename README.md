# PromoLift AI: Causal Uplift Modeling for Smarter Coupon Targeting

PromoLift AI is an end-to-end data science project for smarter coupon and email targeting.
It uses a real randomized marketing experiment to compare normal conversion prediction with uplift modeling and causal validation.
The project shows why companies should not only target customers who are likely to buy.
Instead, they should target customers whose behavior is likely to change because of the campaign.

## Business Problem

Traditional coupon targeting often sends discounts to customers who already have a high chance of buying. That can waste marketing budget because the company pays for incentives without creating many incremental purchases.

The better business question is not "Who will buy?" It is "Who will buy because we sent the campaign?"

## Core Idea

Normal machine learning asks:

> Who is likely to buy?

Uplift modeling asks:

> Who is likely to buy because of the campaign?

This project focuses on incremental conversion lift, treatment/control comparisons, and campaign targeting decisions instead of only predicting purchase probability.

## Dataset

This project uses the Hillstrom Email Marketing dataset, a real marketing experiment dataset with customer features, campaign assignment, and post-campaign outcomes.

- Treatment: `Mens E-Mail`
- Control: `No E-Mail`
- Outcome: `conversion`
- Processed dataset: `data/processed/hillstrom_mens_email.csv`

The `Womens E-Mail` group is excluded in this version so the first treatment/control comparison stays simple and easy to explain.

## Key Results

| Metric | Result |
| --- | ---: |
| Total customers | 42,613 |
| Treatment conversion rate | 1.25% |
| Control conversion rate | 0.57% |
| Observed ATE | +0.68 percentage points |
| Relative lift | 118.84% |
| Propensity model AUC | 0.510 |
| Largest absolute SMD | 0.016 |
| Selected uplift model | T-Learner |
| Top 30% estimated incremental conversions | About 18 |

The experiment shows a positive average treatment effect, and the balance checks support a credible treatment/control comparison. The uplift model is used to rank customers by expected incremental response.

## Project Workflow

1. Data loading and preprocessing
2. EDA and treatment/control comparison
3. Baseline conversion model
4. Uplift modeling with T-Learner and S-Learner
5. Causal validation with balance checks, propensity scores, and DoWhy
6. Streamlit dashboard

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- matplotlib and seaborn
- Streamlit
- DoWhy
- Git and GitHub

## Folder Structure

```text
promolift-ai/
├── app/                  # Streamlit dashboard
├── data/                 # Raw, processed, and synthetic data folders
├── notebooks/            # Exploratory notebooks
├── reports/              # Generated tables, charts, and written summaries
├── scripts/              # Utility scripts for running the pipeline
└── src/
    ├── analysis/         # EDA and treatment/control analysis
    ├── causal/           # Causal validation and balance diagnostics
    ├── data/             # Dataset loading and preprocessing
    ├── evaluation/       # Modeling and uplift evaluation helpers
    ├── features/         # Feature preparation utilities
    ├── models/           # Baseline and uplift models
    └── visualization/    # Plotting helpers
```

## How to Run Locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python scripts/run_all.py
```

Or run each step manually:

```powershell
python -m src.data.load_hillstrom
python -m src.analysis.eda_hillstrom
python -m src.models.baseline_model
python -m src.models.uplift_model
python -m src.causal.causal_validation
streamlit run app/streamlit_app.py
```

The data files are ignored by Git, so regenerate them locally before running downstream steps.

## Dashboard

Run the dashboard with:

```powershell
streamlit run app/streamlit_app.py
```

Dashboard pages:

- Executive Overview: main KPIs, campaign lift, and the project story.
- Dataset & Experiment: Hillstrom experiment setup, treatment/control counts, and EDA visuals.
- Baseline ML Model: standard conversion prediction and why it is not enough.
- Uplift Modeling: T-Learner and S-Learner comparison, uplift ranking, and policy value.
- Causal Validation: ATE, propensity scores, balance checks, SMD, and DoWhy validation.
- Final Recommendation: business recommendation for uplift-based targeting.

## Why This Project Stands Out

PromoLift AI goes beyond a standard classification project. It combines causal inference, uplift modeling, treatment/control experiment design, rare conversion modeling, business decision-making, and dashboard storytelling. The result is a project that connects model outputs to a concrete marketing decision: who should receive a campaign.

## Important Limitation

Uplift scores are used for ranking and decision support, not as perfect individual causal truth. For any single customer, we only observe one outcome: what happened under treatment or what happened under control. The unobserved counterfactual outcome is estimated, not directly observed.

## Final Project Pitch

PromoLift AI is a portfolio-ready causal ML project that shows how to move from "predict who buys" to "target who changes behavior." It demonstrates practical data science judgment across experiment analysis, predictive modeling, uplift modeling, causal validation, and executive dashboard storytelling.
