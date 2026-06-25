# PromoLift AI

PromoLift AI is a portfolio-level data science project for coupon targeting with causal inference and uplift modeling.

## Problem Statement

Many companies send coupons to customers who appear most likely to buy. This can waste marketing budget because some customers would have purchased even without receiving a coupon.

The goal of this project is to identify which customers are most likely to change their behavior because of a coupon. In other words, the project focuses on incremental impact, not just purchase probability.

## Why Normal Machine Learning Is Not Enough

A standard machine learning model can predict whether a customer is likely to purchase. However, that does not answer the most important business question:

Will the coupon cause this customer to buy?

Traditional prediction models often rank already-loyal or high-intent customers at the top. Giving coupons to those customers may reduce profit because the company pays for discounts without creating additional sales.

## What Uplift Modeling Solves

Uplift modeling estimates the difference between a customer's expected behavior with treatment and without treatment. In this project, the treatment is receiving a coupon.

This helps separate customers into useful business groups:

- Customers who are likely to buy only if they receive a coupon
- Customers who would buy anyway
- Customers who are unlikely to buy even with a coupon
- Customers who may react negatively to a coupon

The final goal is to recommend coupon targeting strategies that improve incremental revenue and reduce wasted discounts.

## Planned Tech Stack

- Python
- pandas and NumPy for data processing
- scikit-learn for baseline machine learning
- XGBoost and LightGBM for stronger predictive models
- DoWhy, EconML, and CausalML for causal inference and uplift modeling
- SHAP for model explainability
- Matplotlib, Seaborn, and Plotly for visualization
- Streamlit for an interactive project demo
- Jupyter notebooks for exploration and storytelling

## Dataset

This project uses the Hillstrom Email Marketing dataset, a real marketing experiment dataset with customer features, campaign assignment, and post-campaign outcomes. For the first version, we compare `Mens E-Mail` as the treatment group against `No E-Mail` as the control group. The main outcome is `conversion`, and the processed dataset is saved at `data/processed/hillstrom_mens_email.csv`.

## Exploratory Data Analysis

The EDA compares `Mens E-Mail` treatment customers against `No E-Mail` control customers. It calculates conversion lift, visit lift, and spend differences, then saves segment-level analysis under `reports/eda/` and charts under `reports/figures/`. This analysis motivates why uplift modeling is needed instead of only predicting who is likely to buy.

## Baseline Conversion Model

The baseline model predicts conversion probability using only pre-campaign customer features. Treatment assignment and post-campaign columns are excluded to avoid leakage. The model is evaluated with ROC-AUC, Average Precision, a confusion matrix, and decile lift. This baseline intentionally shows the limitation of normal ML: it can predict who is likely to convert, but it cannot tell whether the email caused the conversion.

## Uplift Modeling

The project moves from conversion prediction to treatment-effect estimation with T-Learner and S-Learner models. Uplift scores estimate `P(conversion | treated) - P(conversion | control)`, allowing customers to be ranked by expected incremental campaign effect. Uplift outputs are saved under `reports/uplift/`, and uplift charts are saved under `reports/figures/`.

## Causal Validation

The causal validation step calculates the observed average treatment effect, checks treatment/control covariate balance, and estimates propensity scores to inspect treatment assignment predictability. It also optionally runs DoWhy causal estimation and refutation when available. The purpose is to support responsible interpretation of uplift results before moving into broader business recommendations.

## Streamlit Dashboard

The Streamlit dashboard reads generated reports, CSV files, JSON files, and PNG charts to present the full PromoLift AI story. Run it with:

```bash
streamlit run app/streamlit_app.py
```

## Planned Project Stages

1. Project setup and repository structure
2. Synthetic customer, coupon, and purchase data generation
3. Exploratory data analysis
4. Baseline purchase prediction model
5. Uplift modeling and treatment effect estimation
6. Model evaluation with uplift-focused metrics
7. Business targeting recommendations
8. Streamlit dashboard for interactive exploration
9. Final report with visuals and project summary

## Current Status

This repository currently contains the initial project structure and placeholder modules. Advanced modeling code will be added in later stages.
