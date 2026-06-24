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
