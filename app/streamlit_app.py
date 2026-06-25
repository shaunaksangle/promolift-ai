"""Streamlit dashboard for PromoLift AI.

This app reads generated reports, tables, and charts. It does not train models,
download data, or create model artifacts.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PROCESSED_DATA_PATH = DATA_DIR / "processed" / "hillstrom_mens_email.csv"
MODELING_DIR = REPORTS_DIR / "modeling"
UPLIFT_DIR = REPORTS_DIR / "uplift"
CAUSAL_DIR = REPORTS_DIR / "causal"

DISPLAY_COLUMN_NAMES = {
    "customer_id": "Customer ID",
    "history_segment": "History Segment",
    "zip_code": "Zip Code",
    "group": "Group",
    "outcome": "Outcome",
    "customers": "Customers",
    "model": "Model",
    "selected_model": "Selected",
    "target_percent": "Target %",
    "targeted_customers": "Targeted Customers",
    "observed_conversion_rate_targeted_treated": "Treated CVR",
    "observed_conversion_rate_targeted_control": "Control CVR",
    "observed_incremental_conversion_rate": "Incremental CVR",
    "estimated_incremental_conversions": "Est. Incremental Conversions",
    "average_predicted_uplift": "Avg Predicted Uplift",
    "top_decile_observed_uplift": "Top Decile Uplift",
    "top_30_policy_incremental_conversion_rate": "Top 30% Incremental CVR",
    "top_30_estimated_incremental_conversions": "Top 30% Est. Incremental Conversions",
    "feature": "Feature",
    "treatment_mean": "Treatment Mean",
    "control_mean": "Control Mean",
    "smd": "SMD",
    "abs_smd": "Abs SMD",
}

PERCENT_TABLE_COLUMNS = {
    "average_predicted_uplift",
    "top_decile_observed_uplift",
    "top_30_policy_incremental_conversion_rate",
    "observed_conversion_rate_targeted_treated",
    "observed_conversion_rate_targeted_control",
    "observed_incremental_conversion_rate",
}

NUMBER_TABLE_COLUMNS = {
    "estimated_incremental_conversions",
    "top_30_estimated_incremental_conversions",
}


def load_csv_safe(path):
    """Load a CSV file if it exists, otherwise show a clear warning."""
    path = Path(path)

    if not path.exists():
        st.warning(f"Missing file: `{path}`. {generation_hint(path)}")
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Could not load `{path}`: {error}")
        return None


def load_json_safe(path):
    """Load a JSON file if it exists, otherwise show a clear warning."""
    path = Path(path)

    if not path.exists():
        st.warning(f"Missing file: `{path}`. {generation_hint(path)}")
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        st.warning(f"Could not load `{path}`: {error}")
        return None


def show_image_if_exists(path, caption=None):
    """Display an image if it exists, otherwise show a clear warning."""
    path = Path(path)

    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Missing image: `{path}`. {generation_hint(path)}")


def format_percent(value):
    """Format a decimal value as a percentage."""
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.2%}"


def format_pp(value):
    """Format a decimal lift value as percentage points."""
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.2f} pp"


def format_target_percent(value):
    """Format a target share whether it is stored as 0.3 or 30."""
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)
    if abs(value) <= 1:
        value *= 100

    return f"{value:.0f}%"


def prepare_display_table(df):
    """Return a display-friendly copy of a CSV table."""
    display_df = df.copy()

    for column in PERCENT_TABLE_COLUMNS.intersection(display_df.columns):
        display_df[column] = display_df[column].map(format_percent)

    for column in NUMBER_TABLE_COLUMNS.intersection(display_df.columns):
        display_df[column] = display_df[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{float(value):,.1f}"
        )

    if "target_percent" in display_df.columns:
        display_df["target_percent"] = display_df["target_percent"].map(format_target_percent)

    return display_df.rename(columns=DISPLAY_COLUMN_NAMES)


def generation_hint(path):
    """Return the command most likely needed to generate a missing artifact."""
    path_text = str(path).replace("\\", "/")

    if "reports/modeling" in path_text or "baseline_" in path.name:
        return "Run `python -m src.models.baseline_model` to generate baseline outputs."
    if "reports/uplift" in path_text or "uplift_" in path.name or "cumulative_uplift" in path.name:
        return "Run `python -m src.models.uplift_model` to generate uplift outputs."
    if "reports/causal" in path_text or "causal_" in path.name or "propensity_" in path.name or "covariate_balance" in path.name:
        return "Run `python -m src.causal.causal_validation` to generate causal validation outputs."
    if "reports/eda" in path_text or path.suffix.lower() == ".png":
        return "Run `python -m src.analysis.eda_hillstrom` to generate EDA outputs."
    if "data/processed" in path_text:
        return "Run `python -m src.data.load_hillstrom` to generate the processed dataset."

    return "Run the relevant project step to generate this artifact."


def metric_row(metrics):
    """Render a row of Streamlit metric cards."""
    columns = st.columns(len(metrics))

    for column, (label, value, help_text) in zip(columns, metrics):
        with column:
            st.metric(label=label, value=value, help=help_text)


def show_chart_grid(image_paths, captions=None, columns_per_row=2):
    """Display images in a clean responsive grid."""
    captions = captions or {}

    for index in range(0, len(image_paths), columns_per_row):
        row_paths = image_paths[index : index + columns_per_row]

        if columns_per_row == 1:
            for path in row_paths:
                show_image_if_exists(path, captions.get(path.name))
            continue

        columns = st.columns(columns_per_row)
        for column, path in zip(columns, row_paths):
            with column:
                show_image_if_exists(path, captions.get(path.name))


def page_executive_overview():
    """Render the executive overview page."""
    st.title("PromoLift AI: Causal Uplift Modeling for Smarter Coupon Targeting")
    st.caption(
        "A portfolio dashboard for comparing normal conversion prediction, "
        "uplift modeling, and causal validation."
    )
    st.divider()

    st.subheader("Executive Overview")
    st.write(
        "PromoLift AI is an end-to-end causal uplift modeling project for smarter "
        "coupon and email targeting."
    )
    st.info(
        "Normal ML asks: Who is likely to buy? PromoLift AI asks: Who is likely "
        "to buy because of the campaign?"
    )

    ate_results = load_json_safe(CAUSAL_DIR / "naive_ate.json") or {}
    total_customers = 42613
    treatment_rate = ate_results.get("treatment_conversion_rate", 0.0125)
    control_rate = ate_results.get("control_conversion_rate", 0.0057)
    observed_ate = ate_results.get("naive_ate", 0.0068)
    propensity_auc = get_propensity_auc()

    metric_row(
        [
            ("Total Customers", f"{total_customers:,}", "Mens E-Mail vs No E-Mail sample"),
            ("Treatment CVR", format_percent(treatment_rate), "Mens E-Mail conversion rate"),
            ("Control CVR", format_percent(control_rate), "No E-Mail conversion rate"),
            ("ATE", format_pp(observed_ate), "Treatment minus control conversion lift"),
            ("Propensity AUC", f"{propensity_auc:.3f}", "Treatment assignment predictability"),
        ]
    )

    st.divider()
    st.subheader("Business Decision")
    st.write(
        "Instead of sending coupons to everyone or only likely buyers, target "
        "customers with positive predicted uplift."
    )

    st.write("The KPI snapshot summarizes the experiment result in recruiter-friendly terms.")
    show_image_if_exists(FIGURES_DIR / "eda_executive_summary.png")
    st.write("The causal summary keeps the focus on treatment effect, not just prediction.")
    show_image_if_exists(FIGURES_DIR / "causal_ate_summary.png")


def page_dataset_experiment():
    """Render the dataset and experiment page."""
    st.header("Dataset & Experiment")
    st.write("**Dataset:** Hillstrom Email Marketing dataset")
    st.write("**Treatment:** Mens E-Mail")
    st.write("**Control:** No E-Mail")
    st.write("**Outcome:** conversion")
    st.write(
        "This is a real randomized marketing experiment with customer features, "
        "campaign assignment, and post-campaign outcomes."
    )

    df = load_csv_safe(PROCESSED_DATA_PATH)

    if df is not None:
        st.subheader("Processed Dataset Sample")
        st.dataframe(prepare_display_table(df.head(10)), width="stretch", hide_index=True)

        st.subheader("Treatment and Outcome Distributions")
        col1, col2 = st.columns(2)
        with col1:
            treatment_dist = (
                df["treatment"]
                .map({0: "Control: No E-Mail", 1: "Treatment: Mens E-Mail"})
                .value_counts()
                .rename_axis("group")
                .reset_index(name="customers")
            )
            st.dataframe(prepare_display_table(treatment_dist), width="stretch", hide_index=True)
        with col2:
            outcome_dist = (
                df["outcome"]
                .map({0: "Did Not Convert", 1: "Converted"})
                .value_counts()
                .rename_axis("outcome")
                .reset_index(name="customers")
            )
            st.dataframe(prepare_display_table(outcome_dist), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Experiment Balance and Outcomes")
    st.write("These charts show whether the experiment is balanced and whether Mens E-Mail changed customer behavior.")
    show_chart_grid(
        [
            FIGURES_DIR / "treatment_distribution.png",
            FIGURES_DIR / "conversion_rate_by_group.png",
            FIGURES_DIR / "visit_rate_by_group.png",
            FIGURES_DIR / "average_spend_by_group.png",
        ]
    )
    st.subheader("Segment Response")
    st.write("Segment charts show why the next step is to rank customers by incremental response.")
    show_chart_grid(
        [
            FIGURES_DIR / "segment_uplift_by_channel.png",
            FIGURES_DIR / "segment_uplift_by_history_segment.png",
        ],
        columns_per_row=1,
    )


def page_baseline_ml_model():
    """Render the baseline ML model page."""
    st.header("Baseline ML Model")
    st.write(
        "The baseline model predicts who is likely to convert using pre-campaign "
        "features."
    )
    st.warning(
        "Limitation: it does not estimate whether the email caused conversion. "
        "Accuracy is misleading because conversion is rare."
    )

    metrics = load_json_safe(MODELING_DIR / "baseline_metrics.json") or {}
    comparison_df = load_csv_safe(MODELING_DIR / "baseline_model_comparison.csv")

    metric_row(
        [
            ("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}" if metrics else "N/A", "Ranking quality"),
            (
                "Avg Precision",
                f"{metrics.get('average_precision', float('nan')):.3f}" if metrics else "N/A",
                "Rare-conversion ranking metric",
            ),
            ("Recall", f"{metrics.get('recall', float('nan')):.3f}" if metrics else "N/A", "Share of converters found"),
            ("Precision", f"{metrics.get('precision', float('nan')):.3f}" if metrics else "N/A", "Positive prediction quality"),
        ]
    )

    if comparison_df is not None:
        st.subheader("Model Comparison")
        st.dataframe(prepare_display_table(comparison_df), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Baseline Model Charts")
    st.write("The baseline model can rank likely buyers, but it cannot show who changed behavior because of treatment.")
    show_chart_grid(
        [
            FIGURES_DIR / "baseline_roc_curve.png",
            FIGURES_DIR / "baseline_precision_recall_curve.png",
        ]
    )
    show_chart_grid(
        [
            FIGURES_DIR / "baseline_confusion_matrix.png",
            FIGURES_DIR / "baseline_probability_distribution.png",
        ]
    )
    show_chart_grid(
        [
            FIGURES_DIR / "baseline_decile_lift.png",
        ],
        columns_per_row=1,
    )


def page_uplift_modeling():
    """Render the uplift modeling page."""
    st.header("Uplift Modeling")
    st.write("**Uplift = P(conversion | treated) - P(conversion | control)**")
    st.write(
        "T-Learner trains separate treatment and control models. S-Learner trains "
        "one model and scores each customer twice: once as treated and once as control."
    )
    st.info(
        "The uplift model ranks customers by expected incremental impact, not just "
        "purchase likelihood."
    )

    comparison_df = load_csv_safe(UPLIFT_DIR / "uplift_model_comparison.csv")
    policy_df = load_csv_safe(UPLIFT_DIR / "uplift_policy_values.csv")

    selected_row = get_selected_uplift_row(comparison_df)
    selected_model = selected_row.get("model", "T-Learner") if selected_row else "T-Learner"
    top_30_incremental_conversions = (
        selected_row.get("top_30_estimated_incremental_conversions") if selected_row else None
    )
    top_30_incremental_rate = (
        selected_row.get("top_30_policy_incremental_conversion_rate") if selected_row else None
    )

    metric_row(
        [
            ("Selected Model", str(selected_model), "Chosen by top-30% policy value"),
            (
                "Top 30% Conv.",
                f"{float(top_30_incremental_conversions):.1f}" if top_30_incremental_conversions is not None else "N/A",
                "Estimated conversions added by targeting top uplift users",
            ),
            (
                "Top 30% Lift",
                format_percent(top_30_incremental_rate),
                "Observed treatment-control lift in targeted group",
            ),
        ]
    )

    if comparison_df is not None:
        st.subheader("Uplift Model Comparison")
        st.dataframe(prepare_display_table(comparison_df), width="stretch", hide_index=True)

    if policy_df is not None:
        st.subheader("Policy Value Comparison")
        st.dataframe(prepare_display_table(policy_df), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Uplift Charts")
    st.write("These charts evaluate ranking quality and the business value of targeting high-uplift customers.")
    show_chart_grid(
        [
            FIGURES_DIR / "uplift_decile_bar.png",
            FIGURES_DIR / "cumulative_uplift_curve.png",
        ]
    )
    show_chart_grid(
        [
            FIGURES_DIR / "predicted_uplift_distribution.png",
            FIGURES_DIR / "uplift_policy_comparison.png",
        ],
        columns_per_row=1,
    )


def page_causal_validation():
    """Render the causal validation page."""
    st.header("Causal Validation")
    st.write(
        "Before interpreting uplift as causal, we check whether the treatment/control "
        "comparison is credible."
    )
    st.info(
        "Low SMD and propensity AUC near 0.5 suggest treatment assignment is "
        "balanced and close to random."
    )

    ate_results = load_json_safe(CAUSAL_DIR / "naive_ate.json") or {}
    balance_df = load_csv_safe(CAUSAL_DIR / "covariate_balance.csv")
    dowhy_results = load_json_safe(CAUSAL_DIR / "dowhy_results.json") or {}

    largest_smd = None
    if balance_df is not None and not balance_df.empty:
        largest_smd = balance_df["abs_smd"].max()

    metric_row(
        [
            (
                "Treatment CVR",
                format_percent(ate_results.get("treatment_conversion_rate")),
                "Mens E-Mail group",
            ),
            (
                "Control CVR",
                format_percent(ate_results.get("control_conversion_rate")),
                "No E-Mail group",
            ),
            ("ATE", format_pp(ate_results.get("naive_ate")), "Observed treatment-control lift"),
            ("Largest SMD", f"{largest_smd:.3f}" if largest_smd is not None else "N/A", "Covariate balance diagnostic"),
            ("Propensity AUC", f"{get_propensity_auc():.3f}", "Treatment assignment predictability"),
            ("DoWhy", str(dowhy_results.get("status", "N/A")), "Optional causal validation layer"),
        ]
    )

    if balance_df is not None:
        st.subheader("Covariate Balance")
        st.dataframe(prepare_display_table(balance_df.head(20)), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Causal Validation Charts")
    st.write("Balance and propensity diagnostics check whether the experiment comparison is credible.")
    show_chart_grid(
        [
            FIGURES_DIR / "covariate_balance_smd.png",
            FIGURES_DIR / "propensity_score_overlap.png",
        ],
        columns_per_row=1,
    )
    show_chart_grid(
        [
            FIGURES_DIR / "treatment_assignment_predictability.png",
            FIGURES_DIR / "causal_ate_summary.png",
        ]
    )


def page_final_recommendation():
    """Render the final recommendation page."""
    st.header("Final Recommendation")
    st.success("Use uplift targeting instead of blanket couponing or normal conversion targeting.")
    st.write(
        "Do not target everyone because discounts are costly. Do not only target "
        "likely buyers because many would buy anyway. Target high-uplift users "
        "because they are more likely to change behavior due to the campaign."
    )

    comparison = pd.DataFrame(
        [
            {
                "Strategy": "Target everyone",
                "What it does": "Sends coupon to all customers",
                "Problem/Benefit": "High discount waste",
            },
            {
                "Strategy": "Normal ML targeting",
                "What it does": "Targets likely buyers",
                "Problem/Benefit": "May target customers who would buy anyway",
            },
            {
                "Strategy": "Uplift targeting",
                "What it does": "Targets customers with incremental response",
                "Problem/Benefit": "Better use of campaign budget",
            },
        ]
    )
    st.dataframe(comparison, width="stretch", hide_index=True)

    st.subheader("Project Pitch")
    st.success(
        "PromoLift AI is an end-to-end causal uplift modeling dashboard that "
        "combines EDA, baseline ML, treatment-effect modeling, and causal "
        "validation to support smarter marketing decisions."
    )


def get_selected_uplift_row(comparison_df):
    """Return the selected uplift model row from the comparison table."""
    if comparison_df is None or comparison_df.empty:
        return None

    if "selected_model" in comparison_df.columns:
        selected_mask = comparison_df["selected_model"].astype(str).str.lower() == "true"
        if selected_mask.any():
            return comparison_df[selected_mask].iloc[0].to_dict()

    return comparison_df.iloc[0].to_dict()


def get_propensity_auc():
    """Return the propensity AUC used in the dashboard."""
    default_auc = 0.510
    dowhy_results = load_json_safe(CAUSAL_DIR / "dowhy_results.json")

    if isinstance(dowhy_results, dict) and "propensity_auc" in dowhy_results:
        return float(dowhy_results["propensity_auc"])

    return default_auc


def main():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="PromoLift AI",
        layout="wide",
    )

    pages = {
        "Executive Overview": page_executive_overview,
        "Dataset & Experiment": page_dataset_experiment,
        "Baseline ML Model": page_baseline_ml_model,
        "Uplift Modeling": page_uplift_modeling,
        "Causal Validation": page_causal_validation,
        "Final Recommendation": page_final_recommendation,
    }

    with st.sidebar:
        st.header("Navigation")
        selected_page = st.radio("Go to", list(pages.keys()))
        st.divider()
        st.write("Core story")
        st.write("Normal ML asks: Who is likely to buy?")
        st.write("PromoLift AI asks: Who buys because of the campaign?")

    pages[selected_page]()


if __name__ == "__main__":
    main()
