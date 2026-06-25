"""Uplift modeling for the Hillstrom Email Marketing dataset.

This module estimates who is likely to convert because of the Mens E-Mail
campaign. It implements simple T-Learner and S-Learner approaches without
adding formal causal validation yet.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.evaluation.uplift_metrics import (
    calculate_average_treatment_effect,
    calculate_policy_value,
    calculate_top_decile_uplift,
    make_uplift_decile_table,
)
from src.features.build_features import build_preprocessor, get_feature_columns
from src.visualization.plots import (
    plot_cumulative_uplift_curve,
    plot_policy_comparison,
    plot_predicted_uplift_distribution,
    plot_uplift_decile_bar,
)


PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "hillstrom_mens_email.csv"
UPLIFT_REPORTS_DIR = REPORTS_DIR / "uplift"


def load_data() -> pd.DataFrame:
    """Load the processed Hillstrom dataset for uplift modeling.

    Raises:
        FileNotFoundError: If the processed dataset has not been created yet.

    Returns:
        Processed Hillstrom DataFrame.
    """
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed Hillstrom dataset was not found at "
            f"{PROCESSED_DATA_PATH}. Run this command first: "
            "python -m src.data.load_hillstrom"
        )

    return pd.read_csv(PROCESSED_DATA_PATH)


def get_uplift_features(df: pd.DataFrame):
    """Return allowed features, treatment, and target for uplift modeling.

    Only pre-campaign customer features are included in X. Post-campaign columns
    such as visit, conversion, and spend are intentionally excluded.
    """
    numeric_features, categorical_features = get_feature_columns()
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns].copy()
    treatment = df["treatment"].astype(int).copy()
    y = df["outcome"].astype(int).copy()

    return X, treatment, y


def split_uplift_data(df: pd.DataFrame):
    """Split uplift data into train and test sets.

    The split is stratified on treatment and outcome together so both treatment
    groups preserve rare conversion balance.
    """
    X, treatment, y = get_uplift_features(df)
    stratify_label = treatment.astype(str) + "_" + y.astype(str)

    return train_test_split(
        X,
        treatment,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_label,
    )


def build_base_classifier() -> LogisticRegression:
    """Return a stable classifier for rare conversion modeling."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )


def train_t_learner(
    X_train: pd.DataFrame,
    treatment_train: pd.Series,
    y_train: pd.Series,
):
    """Train separate treated and control outcome models for a T-Learner."""
    treated_mask = treatment_train == 1
    control_mask = treatment_train == 0

    treated_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_base_classifier()),
        ]
    )
    control_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_base_classifier()),
        ]
    )

    treated_model.fit(X_train.loc[treated_mask], y_train.loc[treated_mask])
    control_model.fit(X_train.loc[control_mask], y_train.loc[control_mask])

    return treated_model, control_model


def predict_t_learner_uplift(
    treated_model,
    control_model,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Predict treatment, control, and uplift scores from a T-Learner."""
    p_treatment = treated_model.predict_proba(X_test)[:, 1]
    p_control = control_model.predict_proba(X_test)[:, 1]

    return pd.DataFrame(
        {
            "p_treatment": p_treatment,
            "p_control": p_control,
            "uplift_score": p_treatment - p_control,
        }
    )


def train_s_learner(
    X_train: pd.DataFrame,
    treatment_train: pd.Series,
    y_train: pd.Series,
):
    """Train one outcome model that includes treatment as a feature."""
    X_train_with_treatment = X_train.copy()
    X_train_with_treatment["treatment"] = treatment_train.astype(int)

    s_model = Pipeline(
        steps=[
            ("preprocessor", _build_s_learner_preprocessor()),
            ("model", build_base_classifier()),
        ]
    )
    s_model.fit(X_train_with_treatment, y_train)

    return s_model


def predict_s_learner_uplift(s_model, X_test: pd.DataFrame) -> pd.DataFrame:
    """Predict counterfactual treatment and control probabilities."""
    treated_version = X_test.copy()
    control_version = X_test.copy()
    treated_version["treatment"] = 1
    control_version["treatment"] = 0

    p_treatment = s_model.predict_proba(treated_version)[:, 1]
    p_control = s_model.predict_proba(control_version)[:, 1]

    return pd.DataFrame(
        {
            "p_treatment": p_treatment,
            "p_control": p_control,
            "uplift_score": p_treatment - p_control,
        }
    )


def run_uplift_modeling() -> dict[str, str | float]:
    """Run the full uplift modeling workflow and save outputs."""
    UPLIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    ate = calculate_average_treatment_effect(df)
    X_train, X_test, treatment_train, treatment_test, y_train, y_test = (
        split_uplift_data(df)
    )

    print("\nTraining T-Learner...")
    treated_model, control_model = train_t_learner(
        X_train,
        treatment_train,
        y_train,
    )
    t_predictions = predict_t_learner_uplift(
        treated_model,
        control_model,
        X_test,
    )

    print("\nTraining S-Learner...")
    s_model = train_s_learner(X_train, treatment_train, y_train)
    s_predictions = predict_s_learner_uplift(s_model, X_test)

    t_decile_table = make_uplift_decile_table(
        y_test,
        treatment_test,
        t_predictions["uplift_score"],
    )
    s_decile_table = make_uplift_decile_table(
        y_test,
        treatment_test,
        s_predictions["uplift_score"],
    )

    model_comparison = _compare_uplift_models(
        t_predictions=t_predictions,
        s_predictions=s_predictions,
        t_decile_table=t_decile_table,
        s_decile_table=s_decile_table,
        y_test=y_test,
        treatment_test=treatment_test,
    )
    best_model_name = model_comparison.iloc[0]["model"]

    if best_model_name == "T-Learner":
        best_decile_table = t_decile_table
        best_predictions = t_predictions
    else:
        best_decile_table = s_decile_table
        best_predictions = s_predictions

    policy_values = _build_policy_value_table(
        y_test,
        treatment_test,
        best_predictions["uplift_score"],
        best_model_name,
    )

    t_decile_table.to_csv(
        UPLIFT_REPORTS_DIR / "t_learner_decile_table.csv",
        index=False,
    )
    s_decile_table.to_csv(
        UPLIFT_REPORTS_DIR / "s_learner_decile_table.csv",
        index=False,
    )
    best_decile_table.to_csv(
        UPLIFT_REPORTS_DIR / "best_uplift_decile_table.csv",
        index=False,
    )
    model_comparison.to_csv(
        UPLIFT_REPORTS_DIR / "uplift_model_comparison.csv",
        index=False,
    )
    policy_values.to_csv(
        UPLIFT_REPORTS_DIR / "uplift_policy_values.csv",
        index=False,
    )

    plot_uplift_decile_bar(
        best_decile_table,
        FIGURES_DIR / "uplift_decile_bar.png",
    )
    plot_cumulative_uplift_curve(
        best_decile_table,
        FIGURES_DIR / "cumulative_uplift_curve.png",
    )
    plot_predicted_uplift_distribution(
        best_predictions["uplift_score"],
        FIGURES_DIR / "predicted_uplift_distribution.png",
    )
    plot_policy_comparison(
        policy_values,
        FIGURES_DIR / "uplift_policy_comparison.png",
    )

    _write_uplift_summary(
        ate=ate,
        best_model_name=best_model_name,
        model_comparison=model_comparison,
        best_decile_table=best_decile_table,
        policy_values=policy_values,
    )

    print("\nUplift model comparison:")
    print(model_comparison.to_string(index=False))
    print(
        "\nSelected best uplift model by top 30% estimated incremental "
        f"conversions: {best_model_name}"
    )

    return {
        "best_model": best_model_name,
        "top_decile_observed_uplift": float(
            calculate_top_decile_uplift(best_decile_table)
        ),
        "top_30_estimated_incremental_conversions": float(
            model_comparison.iloc[0]["top_30_estimated_incremental_conversions"]
        ),
    }


def _compare_uplift_models(
    t_predictions: pd.DataFrame,
    s_predictions: pd.DataFrame,
    t_decile_table: pd.DataFrame,
    s_decile_table: pd.DataFrame,
    y_test: pd.Series,
    treatment_test: pd.Series,
) -> pd.DataFrame:
    """Compare T-Learner and S-Learner using simple uplift diagnostics."""
    rows = []

    for model_name, predictions, decile_table in [
        ("T-Learner", t_predictions, t_decile_table),
        ("S-Learner", s_predictions, s_decile_table),
    ]:
        top_30_policy = calculate_policy_value(
            y_test,
            treatment_test,
            predictions["uplift_score"],
            top_percent=0.3,
        )
        rows.append(
            {
                "model": model_name,
                "top_decile_observed_uplift": calculate_top_decile_uplift(
                    decile_table
                ),
                "average_predicted_uplift": float(
                    predictions["uplift_score"].mean()
                ),
                "top_30_policy_incremental_conversion_rate": top_30_policy[
                    "observed_incremental_conversion_rate"
                ],
                "top_30_estimated_incremental_conversions": top_30_policy[
                    "estimated_incremental_conversions"
                ],
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(
        [
            "top_30_estimated_incremental_conversions",
            "top_decile_observed_uplift",
        ],
        ascending=[False, False],
    )
    comparison_df = comparison_df.reset_index(drop=True)
    comparison_df["selected_model"] = comparison_df.index == 0

    return comparison_df[
        [
            "model",
            "top_decile_observed_uplift",
            "average_predicted_uplift",
            "top_30_policy_incremental_conversion_rate",
            "top_30_estimated_incremental_conversions",
            "selected_model",
        ]
    ]


def _build_policy_value_table(
    y_test: pd.Series,
    treatment_test: pd.Series,
    uplift_score: pd.Series,
    best_model_name: str,
) -> pd.DataFrame:
    """Create policy values for common campaign targeting rules."""
    policies = [
        ("Target everyone", 1.0),
        ("Target top 50% uplift", 0.5),
        ("Target top 30% uplift", 0.3),
        ("Target top 10% uplift", 0.1),
    ]
    rows = []

    for policy_name, top_percent in policies:
        policy_value = calculate_policy_value(
            y_test,
            treatment_test,
            uplift_score,
            top_percent=top_percent,
        )
        policy_value["policy"] = policy_name
        policy_value["model"] = best_model_name
        rows.append(policy_value)

    return pd.DataFrame(rows)[
        [
            "model",
            "policy",
            "target_percent",
            "targeted_customers",
            "observed_conversion_rate_targeted_treated",
            "observed_conversion_rate_targeted_control",
            "observed_incremental_conversion_rate",
            "estimated_incremental_conversions",
        ]
    ]


def _build_s_learner_preprocessor() -> ColumnTransformer:
    """Build preprocessing for S-Learner features including treatment."""
    numeric_features, categorical_features = get_feature_columns()
    numeric_features = numeric_features + ["treatment"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder that works across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _write_uplift_summary(
    ate: float,
    best_model_name: str,
    model_comparison: pd.DataFrame,
    best_decile_table: pd.DataFrame,
    policy_values: pd.DataFrame,
) -> None:
    """Write a portfolio-ready markdown summary of uplift modeling results."""
    selected_model_row = model_comparison[model_comparison["selected_model"]].iloc[0]
    top_decile_uplift = calculate_top_decile_uplift(best_decile_table)
    top_30_policy = policy_values[policy_values["target_percent"] == 0.3].iloc[0]

    summary_text = f"""# Uplift Modeling Summary

## What Uplift Modeling Means

Uplift modeling estimates the incremental effect of treatment. In this project, the treatment is receiving the Mens E-Mail campaign, and the uplift score is:

`P(conversion | treated) - P(conversion | control)`

The goal is to rank customers by expected incremental conversion, not just by overall conversion probability.

## Baseline Conversion Model vs Uplift Model

The baseline conversion model predicts who is likely to convert. The uplift model asks a different question: who is likely to convert because of the email? This distinction matters because high-probability customers may have purchased even without the campaign.

## Learners Implemented

The **T-Learner** trains two separate models: one on treated customers and one on control customers. Uplift is the difference between those two predicted probabilities.

The **S-Learner** trains one model with treatment included as a feature. It estimates uplift by scoring each customer twice: once as treated and once as control.

## Model Comparison

{_format_comparison_table(model_comparison)}

Top-decile observed uplift is useful because it checks whether the highest-ranked customers show stronger treatment response. However, conversion is rare, so one decile can be noisy. For campaign targeting, the selected model is **{best_model_name}** because it produced the highest estimated incremental conversions when targeting the top 30% uplift-ranked customers.

Selection metric:

- Primary: top 30% estimated incremental conversions = {selected_model_row["top_30_estimated_incremental_conversions"]:,.1f}
- Tie-breaker: top-decile observed uplift = {selected_model_row["top_decile_observed_uplift"]:.2%}

## Top Uplift Decile

The top decile contains the customers predicted to be most persuadable by the email. For the selected model, the observed uplift in decile 1 was {top_decile_uplift:.2%}. This is directionally useful, but it should not be the only selection metric because rare conversions make small slices of the test set volatile. The full dataset observed average treatment effect was {ate:.2%}.

## Targeting Top 30% Uplift Users

Targeting the top 30% uplift-ranked users reaches {int(top_30_policy["targeted_customers"]):,} customers in the test set. The observed incremental conversion rate inside that targeted group is {top_30_policy["observed_incremental_conversion_rate"]:.2%}, which corresponds to an estimated {top_30_policy["estimated_incremental_conversions"]:.1f} incremental conversions.

This top-30% policy value is more business-stable than a single top decile because it evaluates a larger campaignable audience. It better reflects the question a marketing team would ask: how many incremental conversions can we expect if we target a realistic share of customers?

## Interpreting Predicted Uplift Scores

Predicted uplift scores are mainly used for ranking customers, not as perfectly calibrated causal probabilities. Class imbalance can inflate probability estimates, especially for rare conversion outcomes, so interpretation should focus on rank ordering, decile performance, and policy value rather than taking every individual uplift score literally.

## Charts

![Observed uplift by decile](../figures/uplift_decile_bar.png)

![Cumulative uplift curve](../figures/cumulative_uplift_curve.png)

![Predicted uplift distribution](../figures/predicted_uplift_distribution.png)

![Policy comparison](../figures/uplift_policy_comparison.png)

## Important Limitation

Individual counterfactual outcomes are not directly observed. We never see the same customer both receiving and not receiving the email at the same time. These uplift estimates rely on the randomized treatment-control structure and model assumptions.

## Why Causal Validation Is Next

The next step is causal validation and more robust treatment-effect estimation with tools such as DoWhy or EconML. That step will make the assumptions explicit, test robustness, and strengthen the argument that the estimated uplift reflects campaign impact rather than model calibration artifacts.
"""

    output_path = UPLIFT_REPORTS_DIR / "uplift_model_summary.md"
    output_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved uplift model summary to: {output_path}")


def _format_comparison_table(model_comparison: pd.DataFrame) -> str:
    """Format the uplift model comparison as a Markdown table."""
    lines = [
        "| Model | Top Decile Observed Uplift | Avg Predicted Uplift | Top 30% Incremental Rate | Top 30% Incremental Conversions | Selected |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in model_comparison.itertuples(index=False):
        selected_label = "Yes" if row.selected_model else "No"
        lines.append(
            "| "
            f"{row.model} | "
            f"{row.top_decile_observed_uplift:.2%} | "
            f"{row.average_predicted_uplift:.2%} | "
            f"{row.top_30_policy_incremental_conversion_rate:.2%} | "
            f"{row.top_30_estimated_incremental_conversions:,.1f} | "
            f"{selected_label} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    run_uplift_modeling()
