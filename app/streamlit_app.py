"""Streamlit dashboard for PromoLift AI.

The dashboard reads generated reports, tables, and charts. It does not train
models, download data, or create model artifacts.
"""

import json
from html import escape
from pathlib import Path
from textwrap import dedent

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
CAUSAL_EDA_DIR = REPORTS_DIR / "causal_eda"
ROBUSTNESS_DIR = REPORTS_DIR / "robustness"

APP_TAGLINE = "Causal uplift modeling for smarter campaign targeting"
FOOTER_TEXT = "PromoLift AI | Causal uplift modeling for smarter campaign targeting"


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
    "treated_customers": "Treated Customers",
    "control_customers": "Control Customers",
    "treatment_customers": "Treatment Customers",
    "treated_conversions": "Treated Conversions",
    "control_conversions": "Control Conversions",
    "treatment_conversion_rate": "Treatment Conversion Rate",
    "control_conversion_rate": "Control Conversion Rate",
    "observed_uplift": "Observed Uplift",
    "observed_conversion_rate_targeted_treated": "Treated CVR",
    "observed_conversion_rate_targeted_control": "Control CVR",
    "observed_incremental_conversion_rate": "Incremental CVR",
    "estimated_incremental_conversions": "Est. Incremental Conversions",
    "average_predicted_uplift": "Avg Predicted Uplift",
    "top_decile_observed_uplift": "Top Decile Uplift",
    "top_30_policy_incremental_conversion_rate": "Top 30% Incremental CVR",
    "top_30_estimated_incremental_conversions": "Top 30% Est. Incremental Conversions",
    "feature": "Feature",
    "type": "Type",
    "treatment_mean": "Treatment Mean",
    "control_mean": "Control Mean",
    "smd": "SMD",
    "abs_smd": "Abs SMD",
    "standardized_mean_difference": "SMD",
    "balance_flag": "Balance",
    "estimate_type": "Estimate Type",
    "adjustment_variable": "Adjustment",
    "effect_estimate": "Effect Estimate",
    "effect_percentage_points": "Effect (pp)",
    "segments_used": "Segments Used",
    "interpretation": "Interpretation",
    "mean_propensity": "Mean Propensity",
    "median_propensity": "Median Propensity",
    "p05_propensity": "P05 Propensity",
    "p95_propensity": "P95 Propensity",
    "min_propensity": "Min Propensity",
    "max_propensity": "Max Propensity",
    "share_in_common_support": "Share in Common Support",
    "common_support_min": "Common Support Min",
    "common_support_max": "Common Support Max",
    "propensity_auc": "Propensity AUC",
    "calibration_gap": "Calibration Gap",
    "qini_coefficient": "Qini Coefficient",
    "top_30_targeted_customers": "Top 30% Customers",
    "top_30_treated_customers": "Top 30% Treated",
    "top_30_control_customers": "Top 30% Control",
    "method": "Method",
    "policy": "Policy",
    "segment_variable": "Segment Variable",
    "segment_value": "Segment Value",
    "absolute_incremental_conversions": "Abs. Incremental Conversions",
    "caution_flag": "Caution",
    "treatment_definition": "Treatment Definition",
    "observed_ate": "Observed ATE",
    "relative_lift": "Relative Lift",
    "standard_error": "Standard Error",
    "p_value": "p-value",
    "treatment_visit_rate": "Treatment Visit Rate",
    "control_visit_rate": "Control Visit Rate",
    "visit_lift": "Visit Lift",
    "strategy": "Strategy",
    "what_it_does": "What It Does",
    "problem_benefit": "Problem / Benefit",
}

PERCENT_TABLE_COLUMNS = {
    "average_predicted_uplift",
    "top_decile_observed_uplift",
    "top_30_policy_incremental_conversion_rate",
    "observed_conversion_rate_targeted_treated",
    "observed_conversion_rate_targeted_control",
    "observed_incremental_conversion_rate",
    "observed_uplift",
    "effect_estimate",
    "treatment_conversion_rate",
    "control_conversion_rate",
    "segment_uplift",
    "share_in_common_support",
    "calibration_gap",
    "ci_lower",
    "ci_upper",
    "ate",
    "observed_ate",
    "relative_lift",
    "treatment_visit_rate",
    "control_visit_rate",
    "visit_lift",
}

NUMBER_TABLE_COLUMNS = {
    "estimated_incremental_conversions",
    "top_30_estimated_incremental_conversions",
    "absolute_incremental_conversions",
}

INTEGER_TABLE_COLUMNS = {
    "customers",
    "targeted_customers",
    "treated_customers",
    "control_customers",
    "treatment_customers",
    "treated_conversions",
    "control_conversions",
    "segments_used",
    "top_30_targeted_customers",
    "top_30_treated_customers",
    "top_30_control_customers",
}

DECIMAL_TABLE_COLUMNS = {
    "propensity_auc",
    "qini_coefficient",
    "mean_propensity",
    "median_propensity",
    "p05_propensity",
    "p95_propensity",
    "min_propensity",
    "max_propensity",
    "smd",
    "abs_smd",
    "standardized_mean_difference",
    "standard_error",
    "p_value",
}

CHART_TITLES = {
    "eda_executive_summary.png": "Experiment Snapshot",
    "causal_ate_summary.png": "Observed Campaign Lift",
    "treatment_distribution.png": "Experiment Balance",
    "conversion_rate_by_group.png": "Conversion Rate by Group",
    "visit_rate_by_group.png": "Visit Rate by Group",
    "average_spend_by_group.png": "Average Spend by Group",
    "segment_uplift_by_channel.png": "Uplift by Channel",
    "segment_uplift_by_history_segment.png": "Uplift by Customer History",
    "causal_eda_numeric_overlap.png": "Numeric Covariate Overlap",
    "causal_eda_categorical_balance.png": "Categorical Balance",
    "causal_eda_propensity_overlap.png": "Propensity Overlap",
    "causal_eda_naive_vs_adjusted_effect.png": "Observed ATE Robustness",
    "causal_eda_subgroup_uplift.png": "Subgroup Uplift",
    "causal_eda_heterogeneity_heatmap.png": "Heterogeneity Heatmap",
    "causal_eda_dag.png": "Causal DAG and Leakage View",
    "baseline_roc_curve.png": "Baseline Ranking Quality",
    "baseline_precision_recall_curve.png": "Rare-Conversion Performance",
    "baseline_confusion_matrix.png": "Threshold Classification View",
    "baseline_probability_distribution.png": "Conversion Score Distribution",
    "baseline_decile_lift.png": "Conversion Lift by Decile",
    "qini_curve.png": "Qini Curve",
    "uplift_calibration_by_decile.png": "Uplift Calibration by Decile",
    "uplift_decile_bar.png": "Observed Uplift by Decile",
    "cumulative_uplift_curve.png": "Cumulative Uplift",
    "predicted_uplift_distribution.png": "Predicted Uplift Distribution",
    "uplift_policy_comparison.png": "Targeting Policy Value",
    "x_learner_qini_curve.png": "X-Learner Qini Check",
    "x_learner_calibration_by_decile.png": "X-Learner Calibration Check",
    "uplift_method_robustness_comparison.png": "Method Robustness Comparison",
    "segment_heterogeneity_checks.png": "Segment Heterogeneity Checks",
    "treatment_definition_comparison.png": "Treatment Definition Robustness",
    "ate_confidence_interval.png": "ATE Confidence Interval",
    "covariate_balance_smd.png": "Covariate Balance",
    "propensity_score_overlap.png": "Propensity Score Overlap",
    "treatment_assignment_predictability.png": "Treatment Assignment Predictability",
}


def load_csv_safe(path):
    """Load a CSV file if it exists, otherwise show a clear warning."""
    path = Path(path)

    if not path.exists():
        render_insight_box(f"Missing file: `{path}`. {generation_hint(path)}", "warning")
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        render_insight_box(f"Could not load `{path}`: {error}", "warning")
        return None


def load_json_safe(path):
    """Load a JSON file if it exists, otherwise show a clear warning."""
    path = Path(path)

    if not path.exists():
        render_insight_box(f"Missing file: `{path}`. {generation_hint(path)}", "warning")
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        render_insight_box(f"Could not load `{path}`: {error}", "warning")
        return None


def format_percent(value):
    """Format a decimal value as a percentage."""
    if value is None or pd.isna(value):
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(numeric_value):
        return "N/A"

    return f"{numeric_value:.2%}"


def format_pp(value):
    """Format a decimal lift value as percentage points."""
    if value is None or pd.isna(value):
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(numeric_value):
        return "N/A"

    return f"{numeric_value * 100:.2f} pp"


def format_target_percent(value):
    """Format a target share whether it is stored as 0.3 or 30."""
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)
    if abs(value) <= 1:
        value *= 100

    return f"{value:.0f}%"


def clean_column_names(df):
    """Return a copy of a table with clean display columns and formatted values."""
    display_df = df.copy()

    if "selected_model" in display_df.columns:
        display_df["selected_model"] = display_df["selected_model"].map(
            lambda value: "Yes" if str(value).lower() in {"true", "1", "yes"} else "No"
        )

    for column in PERCENT_TABLE_COLUMNS.intersection(display_df.columns):
        display_df[column] = display_df[column].map(format_percent)

    for column in NUMBER_TABLE_COLUMNS.intersection(display_df.columns):
        display_df[column] = display_df[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{float(value):,.1f}"
        )

    for column in INTEGER_TABLE_COLUMNS.intersection(display_df.columns):
        display_df[column] = display_df[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{float(value):,.0f}"
        )

    for column in DECIMAL_TABLE_COLUMNS.intersection(display_df.columns):
        display_df[column] = display_df[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{float(value):.3f}"
        )

    if "target_percent" in display_df.columns:
        display_df["target_percent"] = display_df["target_percent"].map(format_target_percent)

    display_df.columns = [
        DISPLAY_COLUMN_NAMES.get(column, column.replace("_", " ").title())
        for column in display_df.columns
    ]

    return display_df.reset_index(drop=True)


def prepare_display_table(df):
    """Backward-compatible alias for table display formatting."""
    return clean_column_names(df)


def generation_hint(path):
    """Return the command most likely needed to generate a missing artifact."""
    path_text = str(path).replace("\\", "/")

    if "reports/robustness" in path_text or path.name in {
        "x_learner_qini_curve.png",
        "x_learner_calibration_by_decile.png",
        "uplift_method_robustness_comparison.png",
        "segment_heterogeneity_checks.png",
        "treatment_definition_comparison.png",
    }:
        return "Run `python -m src.analysis.robustness_checks` to generate robustness outputs."
    if "reports/causal_eda" in path_text or "causal_eda_" in path.name:
        return "Run `python -m src.analysis.causal_eda` to generate causal EDA outputs."
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


def apply_visual_system():
    """Apply a centralized CSS polish layer for the dashboard."""
    st.markdown(
        dedent(
            """
        <style>
        :root {
            --pl-text: #111827;
            --pl-muted: #4B5563;
            --pl-border: #E5E7EB;
            --pl-panel: #F8FAFC;
            --pl-navy: #1E3A8A;
            --pl-green: #0F766E;
            --pl-amber: #B45309;
            --pl-red: #B91C1C;
        }

        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, Arial, sans-serif;
            color: var(--pl-text);
            background: #F8FAFC;
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }

        [data-testid="stSidebar"] {
            background: #F8FAFC;
            border-right: 1px solid var(--pl-border);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: 0;
        }

        .pl-page-header {
            padding: 0.1rem 0 1.1rem 0;
            border-bottom: 1px solid var(--pl-border);
            margin-bottom: 1.25rem;
        }

        .pl-page-header h1 {
            margin: 0;
            font-size: 2.05rem;
            line-height: 1.18;
            letter-spacing: 0;
            color: var(--pl-text);
        }

        .pl-page-header p {
            margin: 0.55rem 0 0 0;
            max-width: 860px;
            color: var(--pl-muted);
            font-size: 1.02rem;
            line-height: 1.55;
        }

        .pl-section {
            margin-top: 1.75rem;
            margin-bottom: 0.8rem;
        }

        .pl-section h2 {
            margin: 0;
            font-size: 1.24rem;
            line-height: 1.3;
            letter-spacing: 0;
        }

        .pl-section p {
            margin: 0.35rem 0 0 0;
            color: var(--pl-muted);
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .pl-callout {
            padding: 0.9rem 1rem;
            border-radius: 10px;
            margin: 0.85rem 0 1.05rem 0;
            border: 1px solid var(--pl-border);
            border-left: 4px solid #94A3B8;
            background: #FFFFFF;
            color: var(--pl-text);
            line-height: 1.55;
            font-size: 0.98rem;
        }

        .pl-callout.neutral {
            border-left-color: #94A3B8;
        }

        .pl-callout.evidence {
            border-left-color: var(--pl-green);
        }

        .pl-callout.caution {
            border-left-color: var(--pl-amber);
        }

        .pl-callout.recommendation {
            border-left-color: var(--pl-navy);
        }

        .pl-caption {
            margin-top: -0.2rem;
            margin-bottom: 1.1rem;
            color: var(--pl-muted);
            font-size: 0.88rem;
            text-align: center;
            line-height: 1.45;
        }

        .pl-footer {
            margin-top: 2.3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--pl-border);
            color: #6B7280;
            font-size: 0.86rem;
            text-align: center;
        }

        .pl-sidebar-title {
            font-weight: 800;
            font-size: 1.25rem;
            color: var(--pl-text);
            margin-bottom: 0.2rem;
        }

        .pl-sidebar-tagline {
            color: var(--pl-muted);
            font-size: 0.88rem;
            line-height: 1.4;
            margin-bottom: 0.7rem;
        }

        .pl-sidebar-workflow-title {
            margin: 0.25rem 0 0.35rem 0;
            color: var(--pl-text);
            font-size: 0.9rem;
            font-weight: 780;
        }

        .pl-sidebar-workflow {
            margin: 0;
            padding-left: 1.1rem;
            color: var(--pl-muted);
            font-size: 0.86rem;
            line-height: 1.55;
        }

        .pl-kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.85rem;
            margin: 1.05rem 0 1.15rem 0;
        }

        .pl-kpi-card {
            background: #FFFFFF;
            border: 1px solid var(--pl-border);
            border-radius: 10px;
            padding: 0.95rem 1rem;
            min-height: 112px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow-wrap: anywhere;
        }

        .pl-kpi-card.primary {
            border-left: 4px solid var(--pl-green);
            min-height: 120px;
        }

        .pl-kpi-label {
            color: var(--pl-muted);
            font-size: 0.81rem;
            font-weight: 750;
            line-height: 1.25;
        }

        .pl-kpi-value {
            margin-top: 0.45rem;
            color: var(--pl-text);
            font-size: clamp(1.28rem, 1rem + 0.7vw, 1.85rem);
            font-weight: 800;
            line-height: 1.12;
        }

        .pl-kpi-card.primary .pl-kpi-value {
            color: var(--pl-green);
            font-size: clamp(1.55rem, 1.1rem + 1vw, 2.25rem);
        }

        .pl-kpi-help {
            margin-top: 0.45rem;
            color: #64748B;
            font-size: 0.76rem;
            line-height: 1.35;
        }

        .pl-chart-block {
            margin: 1rem 0 1.35rem 0;
        }

        .pl-chart-title {
            margin: 0 0 0.25rem 0;
            color: var(--pl-text);
            font-size: 1.02rem;
            font-weight: 780;
            line-height: 1.35;
        }

        .pl-chart-takeaway {
            margin: 0 0 0.65rem 0;
            color: var(--pl-muted);
            font-size: 0.93rem;
            line-height: 1.5;
        }

        .pl-rec-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 0.9rem;
            margin: 0.9rem 0 1.2rem 0;
        }

        .pl-rec-card {
            background: #FFFFFF;
            border: 1px solid var(--pl-border);
            border-radius: 10px;
            padding: 1rem;
            min-height: 140px;
        }

        .pl-rec-card h3 {
            margin: 0 0 0.55rem 0;
            font-size: 0.98rem;
            line-height: 1.35;
            color: var(--pl-navy);
        }

        .pl-rec-card p {
            margin: 0;
            color: var(--pl-muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--pl-border);
            border-radius: 8px;
            overflow: hidden;
        }

        .stExpander {
            border-color: var(--pl-border);
            border-radius: 8px;
        }

        img {
            border-radius: 6px;
        }
        </style>
        """
        ),
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle=None):
    """Render a consistent page title and subtitle."""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        dedent(
            f"""
        <div class="pl-page-header">
            <h1>{title}</h1>
            {subtitle_html}
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


def render_section_title(title, description=None):
    """Render a consistent section title with optional description."""
    description_html = f"<p>{description}</p>" if description else ""
    st.markdown(
        dedent(
            f"""
        <div class="pl-section">
            <h2>{title}</h2>
            {description_html}
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


def render_section_header(title, body=None):
    """Alias for section headers used by the dashboard pages."""
    render_section_title(title, body)


def render_kpi_cards(cards, primary_index=0):
    """Render responsive KPI cards without truncating labels or values."""
    if not cards:
        return

    cards_html = ""
    for index, card in enumerate(cards):
        if isinstance(card, dict):
            label = card.get("label", "")
            value = card.get("value", "")
            help_text = card.get("help_text", "")
        else:
            label, value, *rest = card
            help_text = rest[0] if rest else ""

        title_attr = f' title="{escape(str(help_text))}"' if help_text else ""
        help_html = (
            f'<div class="pl-kpi-help">{escape(str(help_text))}</div>'
            if help_text
            else ""
        )
        card_class = "pl-kpi-card primary" if index == primary_index else "pl-kpi-card"
        cards_html += (
            f'<div class="{card_class}"{title_attr}>'
            '<div>'
            f'<div class="pl-kpi-label">{escape(str(label))}</div>'
            f'<div class="pl-kpi-value">{escape(str(value))}</div>'
            '</div>'
            f'{help_html}'
            '</div>'
        )

    html_string = f'<div class="pl-kpi-grid">{cards_html}</div>'
    st.markdown(html_string, unsafe_allow_html=True)


def render_metric_row(metrics, primary_index=0):
    """Render KPI cards from the existing metric tuple format."""
    render_kpi_cards(metrics, primary_index=primary_index)


def metric_row(metrics):
    """Backward-compatible alias for metric rendering."""
    render_metric_row(metrics)


def render_callout(text, tone="neutral"):
    """Render a compact custom callout without Streamlit's default alert styling."""
    safe_tone = tone if tone in {"neutral", "evidence", "caution", "recommendation"} else "neutral"
    st.markdown(
        f'<div class="pl-callout {safe_tone}">{text}</div>',
        unsafe_allow_html=True,
    )


def render_insight_box(text, type="info"):
    """Backward-compatible alias for the custom callout component."""
    tone_map = {
        "info": "neutral",
        "warning": "caution",
        "success": "recommendation",
        "danger": "caution",
        "evidence": "evidence",
        "caution": "caution",
        "recommendation": "recommendation",
        "neutral": "neutral",
    }
    render_callout(text, tone_map.get(type, "neutral"))


def chart_title_from_path(path):
    """Create a readable chart title from a report image filename."""
    path = Path(path)
    return CHART_TITLES.get(path.name, path.stem.replace("_", " ").title())


def render_chart(path, title=None, takeaway=None):
    """Display one report chart with a short title and business takeaway."""
    path = Path(path)
    chart_title = title or chart_title_from_path(path)

    if path.exists():
        if chart_title or takeaway:
            takeaway_html = f'<p class="pl-chart-takeaway">{escape(str(takeaway))}</p>' if takeaway else ""
            st.markdown(
                (
                    '<div class="pl-chart-block">'
                    f'<div class="pl-chart-title">{escape(str(chart_title))}</div>'
                    f'{takeaway_html}'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
        st.image(str(path), width="stretch")
    else:
        render_insight_box(f"Missing image: `{path}`. {generation_hint(path)}", "warning")


def render_recommendation_cards(rows):
    """Render stakeholder-facing recommendation cards."""
    cards_html = ""
    for row in rows:
        title = row.get("title") or row.get("Recommendation Area") or ""
        body = row.get("body") or row.get("Recommendation") or ""
        cards_html += (
            '<div class="pl-rec-card">'
            f'<h3>{escape(str(title))}</h3>'
            f'<p>{escape(str(body))}</p>'
            '</div>'
        )

    st.markdown(f'<div class="pl-rec-grid">{cards_html}</div>', unsafe_allow_html=True)


def render_image(path, caption=None):
    """Display an image with a consistent missing-file warning and caption."""
    render_chart(path, takeaway=caption)


def show_image_if_exists(path, caption=None):
    """Backward-compatible alias for image rendering."""
    render_image(path, caption)


def render_table(df, title=None, max_rows=None, columns=None, expander=None):
    """Render a display-friendly table with a hidden index."""
    if df is None:
        return

    display_df = df.copy()
    if columns:
        available_columns = [column for column in columns if column in display_df.columns]
        display_df = display_df[available_columns]
    if max_rows:
        display_df = display_df.head(max_rows)
    highlight_mask = None
    if "selected_model" in display_df.columns:
        highlight_mask = (
            display_df["selected_model"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
            .reset_index(drop=True)
        )
    display_df = clean_column_names(display_df)

    def _draw_table():
        if title:
            st.markdown(f"**{escape(str(title))}**")
        if highlight_mask is not None and highlight_mask.any():
            styled_df = display_df.style.apply(
                lambda row: [
                    "background-color: #ECFDF5; font-weight: 650;"
                    if bool(highlight_mask.iloc[row.name])
                    else ""
                    for _ in row
                ],
                axis=1,
            )
            st.dataframe(styled_df, width="stretch", hide_index=True)
        else:
            st.dataframe(display_df, width="stretch", hide_index=True)

    if expander:
        with st.expander(expander):
            _draw_table()
    else:
        _draw_table()


def show_chart_grid(image_paths, captions=None, columns_per_row=2):
    """Display report images in a readable responsive grid."""
    captions = captions or {}

    for index in range(0, len(image_paths), columns_per_row):
        row_paths = image_paths[index : index + columns_per_row]
        if columns_per_row == 1:
            for path in row_paths:
                render_chart(path, takeaway=captions.get(Path(path).name))
            continue

        columns = st.columns(columns_per_row)
        for column, path in zip(columns, row_paths):
            with column:
                render_chart(path, takeaway=captions.get(Path(path).name))


def render_footer():
    """Render a subtle consistent footer."""
    st.markdown(f'<div class="pl-footer">{FOOTER_TEXT}</div>', unsafe_allow_html=True)


def page_executive_overview():
    """Render the executive overview page."""
    render_page_header(
        "PromoLift AI",
        "Normal ML predicts buyers. PromoLift AI estimates who buys because of the campaign.",
    )
    render_callout(
        "Normal ML asks: <b>Who is likely to buy?</b> PromoLift AI asks: "
        "<b>Who is likely to buy because of the campaign?</b>",
        "neutral",
    )

    ate_results = load_json_safe(CAUSAL_DIR / "naive_ate.json") or {}
    balance_df = load_csv_safe(CAUSAL_DIR / "covariate_balance.csv")
    total_customers = 42613
    treatment_rate = ate_results.get("treatment_conversion_rate", 0.0125)
    control_rate = ate_results.get("control_conversion_rate", 0.0057)
    observed_ate = ate_results.get("observed_ate", ate_results.get("naive_ate", 0.0068))
    propensity_auc = get_propensity_auc()
    largest_smd = (
        balance_df["abs_smd"].max()
        if balance_df is not None and "abs_smd" in balance_df.columns and not balance_df.empty
        else 0.016
    )

    render_metric_row(
        [
            ("Observed ATE", format_pp(observed_ate), "Treatment minus control conversion lift"),
            ("Total Customers", f"{total_customers:,}", "Mens E-Mail vs No E-Mail sample"),
            ("Treatment Conversion Rate", format_percent(treatment_rate), "Mens E-Mail conversion rate"),
            ("Control Conversion Rate", format_percent(control_rate), "No E-Mail conversion rate"),
            ("Propensity AUC", f"{propensity_auc:.3f}", "Treatment assignment predictability"),
            ("Largest SMD", f"{largest_smd:.3f}", "Largest covariate imbalance diagnostic"),
        ]
    )

    render_callout(
        "The campaign creates measurable lift, but individual-level targeting "
        "signals are modest/noisy; therefore the project evaluates both campaign-level "
        "impact and targeting value.",
        "evidence",
    )

    render_section_title(
        "Executive Evidence",
        "The first chart summarizes core KPIs. The second keeps the focus on observed treatment effect.",
    )
    show_chart_grid(
        [
            FIGURES_DIR / "eda_executive_summary.png",
            FIGURES_DIR / "causal_ate_summary.png",
        ],
        captions={
            "eda_executive_summary.png": "The campaign creates measurable average lift before any targeting model is introduced.",
            "causal_ate_summary.png": "The treatment/control comparison keeps the story anchored in campaign impact, not purchase likelihood.",
        },
        columns_per_row=1,
    )

    render_callout(
        "Business takeaway: avoid blanket couponing and avoid targeting only likely buyers. "
        "Use uplift as a ranking signal, then validate targeting policies with controlled tests.",
        "recommendation",
    )
    render_footer()


def page_dataset_experiment():
    """Render the dataset and experiment page."""
    render_page_header(
        "Dataset & Experiment",
        "A real randomized marketing experiment comparing Mens E-Mail treatment against a No E-Mail holdout.",
    )

    render_section_title("Treatment Design")
    design_df = pd.DataFrame(
        [
            {"Group": "Treatment", "Definition": "Mens E-Mail", "Role": "Receives campaign"},
            {"Group": "Control", "Definition": "No E-Mail", "Role": "Holdout comparison"},
            {"Group": "Outcome", "Definition": "conversion", "Role": "Post-campaign purchase conversion"},
        ]
    )
    render_table(design_df, title="Experiment setup")
    render_callout(
        "Post-campaign fields such as <b>visit</b>, <b>conversion</b>, and "
        "<b>spend</b> are excluded from model features to avoid leakage.",
        "caution",
    )

    df = load_csv_safe(PROCESSED_DATA_PATH)
    if df is not None:
        render_section_title("Processed Dataset", "A small sample confirms the modeling-ready columns.")
        render_table(df, max_rows=10, expander="View first 10 processed rows")

        treatment_dist = (
            df["treatment"]
            .map({0: "Control: No E-Mail", 1: "Treatment: Mens E-Mail"})
            .value_counts()
            .rename_axis("group")
            .reset_index(name="customers")
        )
        outcome_dist = (
            df["outcome"]
            .map({0: "Did Not Convert", 1: "Converted"})
            .value_counts()
            .rename_axis("outcome")
            .reset_index(name="customers")
        )
        col1, col2 = st.columns(2)
        with col1:
            render_table(treatment_dist, title="Treatment distribution")
        with col2:
            render_table(outcome_dist, title="Outcome distribution")

    render_section_title(
        "Experiment Balance and Outcomes",
        "These figures show whether the experiment is balanced and whether the campaign changed customer behavior.",
    )
    show_chart_grid(
        [
            FIGURES_DIR / "treatment_distribution.png",
            FIGURES_DIR / "conversion_rate_by_group.png",
            FIGURES_DIR / "visit_rate_by_group.png",
            FIGURES_DIR / "average_spend_by_group.png",
        ],
        captions={
            "treatment_distribution.png": "The treatment and holdout groups are large enough for a credible experiment readout.",
            "conversion_rate_by_group.png": "Mens E-Mail increased conversion compared with the no-email holdout.",
            "visit_rate_by_group.png": "The campaign also moved engagement, not only final purchases.",
            "average_spend_by_group.png": "Spend differences support the business value question behind targeting.",
        },
    )

    render_section_title(
        "Segment Response",
        "Segment-level response hints at why average conversion prediction is not the whole story.",
    )
    show_chart_grid(
        [
            FIGURES_DIR / "segment_uplift_by_channel.png",
            FIGURES_DIR / "segment_uplift_by_history_segment.png",
        ],
        captions={
            "segment_uplift_by_channel.png": "Campaign response differs by acquisition channel, which motivates segment-aware targeting.",
            "segment_uplift_by_history_segment.png": "Customer history changes response, so one average conversion model is too blunt.",
        },
        columns_per_row=1,
    )

    render_callout(
        "Dataset takeaway: this is a clean treatment/control setup where causal reasoning is natural, "
        "but sparse conversion still makes fine-grained targeting difficult.",
        "evidence",
    )
    render_footer()


def page_causal_eda():
    """Render the causal EDA page."""
    render_page_header(
        "Causal EDA",
        "Causal projects use EDA to inspect treatment assignment, overlap, selection bias, leakage, and heterogeneity.",
    )
    render_callout(
        "Post-campaign columns are excluded from model features. Timestamp-level leakage checks are limited "
        "because the processed dataset does not contain detailed event timestamps.",
        "caution",
    )

    effects_df = load_csv_safe(CAUSAL_EDA_DIR / "naive_vs_stratified_effects.csv")
    overlap_df = load_csv_safe(CAUSAL_EDA_DIR / "propensity_overlap_summary.csv")

    render_section_title(
        "Balance and Overlap",
        "Before modeling uplift, check whether treated and control customers look comparable before treatment.",
    )
    show_chart_grid(
        [
            FIGURES_DIR / "causal_eda_numeric_overlap.png",
            FIGURES_DIR / "causal_eda_categorical_balance.png",
            FIGURES_DIR / "causal_eda_propensity_overlap.png",
        ],
        captions={
            "causal_eda_numeric_overlap.png": "Pre-campaign numeric features overlap across treatment and control.",
            "causal_eda_categorical_balance.png": "Categorical segment shares are similar enough to support a clean comparison.",
            "causal_eda_propensity_overlap.png": "Propensity AUC near 0.5 supports randomized-like assignment.",
        },
        columns_per_row=1,
    )
    render_table(overlap_df, expander="View propensity overlap summary")

    render_section_title(
        "Average Effect Robustness",
        "The randomized setup makes the observed ATE meaningful; stratified checks show whether it is stable across key groups.",
    )
    render_chart(
        FIGURES_DIR / "causal_eda_naive_vs_adjusted_effect.png",
        takeaway="The observed effect remains business-relevant after simple stratified checks.",
    )
    render_table(effects_df, expander="View naive vs stratified-adjusted effects")

    render_section_title(
        "Subgroup Uplift",
        "Heterogeneous treatment effects motivate uplift modeling instead of relying only on one average effect.",
    )
    show_chart_grid(
        [
            FIGURES_DIR / "causal_eda_subgroup_uplift.png",
            FIGURES_DIR / "causal_eda_heterogeneity_heatmap.png",
        ],
        captions={
            "causal_eda_subgroup_uplift.png": "Segment-level lift shows where campaign response differs.",
            "causal_eda_heterogeneity_heatmap.png": "Heterogeneity patterns explain why one average effect is not enough.",
        },
        columns_per_row=1,
    )

    render_section_title("DAG and Leakage Check")
    render_chart(
        FIGURES_DIR / "causal_eda_dag.png",
        takeaway="The DAG keeps pre-campaign features, treatment assignment, and post-campaign outcomes conceptually separated.",
    )
    render_callout(
        "Causal EDA takeaway: the experiment appears suitable for treatment/control reasoning, "
        "and subgroup variation motivates uplift modeling.",
        "evidence",
    )
    render_footer()


def page_baseline_ml_model():
    """Render the baseline ML model page."""
    render_page_header(
        "Baseline ML Model",
        "A normal classifier predicts who is likely to convert, but not whether the email caused conversion.",
    )
    render_callout(
        "Accuracy can be misleading when conversion is rare. A high-probability buyer may still be a poor coupon target "
        "if they would have purchased without the email.",
        "caution",
    )

    metrics = load_json_safe(MODELING_DIR / "baseline_metrics.json") or {}
    comparison_df = load_csv_safe(MODELING_DIR / "baseline_model_comparison.csv")

    render_metric_row(
        [
            ("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}" if metrics else "N/A", "Ranking quality"),
            ("Avg Precision", f"{metrics.get('average_precision', float('nan')):.3f}" if metrics else "N/A", "Rare-conversion ranking metric"),
            ("Recall", f"{metrics.get('recall', float('nan')):.3f}" if metrics else "N/A", "Share of converters found"),
            ("Precision", f"{metrics.get('precision', float('nan')):.3f}" if metrics else "N/A", "Positive prediction quality"),
        ]
    )

    render_table(comparison_df, expander="View baseline model comparison")

    render_section_title(
        "Baseline Model Charts",
        "These charts show predictive performance, while also exposing why prediction is not causal targeting.",
    )
    show_chart_grid(
        [
            FIGURES_DIR / "baseline_roc_curve.png",
            FIGURES_DIR / "baseline_precision_recall_curve.png",
            FIGURES_DIR / "baseline_confusion_matrix.png",
            FIGURES_DIR / "baseline_probability_distribution.png",
            FIGURES_DIR / "baseline_decile_lift.png",
        ],
        captions={
            "baseline_roc_curve.png": "The classifier can rank likely converters, which is useful but not causal.",
            "baseline_precision_recall_curve.png": "Rare conversion makes precision-recall more informative than accuracy.",
            "baseline_confusion_matrix.png": "A fixed threshold is less useful than ranking when positives are rare.",
            "baseline_probability_distribution.png": "Purchase-likelihood scores are not treatment-effect scores.",
            "baseline_decile_lift.png": "High-probability deciles find buyers, not necessarily persuadable customers.",
        },
        columns_per_row=1,
    )

    render_callout(
        "Baseline takeaway: classification is a useful reference point, but campaign targeting needs treatment-effect logic.",
        "evidence",
    )
    render_footer()


def page_uplift_modeling():
    """Render the uplift modeling page."""
    render_page_header(
        "Uplift Modeling",
        "T-Learner and S-Learner models estimate heterogeneous response to the Mens E-Mail campaign.",
    )
    render_callout(
        "<b>Predicted uplift is mainly a ranking signal, not perfect individual causal truth.</b> "
        "Qini and calibration diagnostics help check whether ranking and magnitude are useful.",
        "caution",
    )

    comparison_df = load_csv_safe(UPLIFT_DIR / "uplift_model_comparison.csv")
    policy_df = load_csv_safe(UPLIFT_DIR / "uplift_policy_values.csv")
    calibration_df = load_csv_safe(UPLIFT_DIR / "uplift_calibration_table.csv")

    selected_row = get_selected_uplift_row(comparison_df)
    selected_model = selected_row.get("model", "T-Learner") if selected_row else "T-Learner"
    top_30_incremental_conversions = (
        selected_row.get("top_30_estimated_incremental_conversions") if selected_row else None
    )
    qini_coefficient = selected_row.get("qini_coefficient") if selected_row else None
    top_decile_uplift = selected_row.get("top_decile_observed_uplift") if selected_row else None
    average_predicted_uplift = selected_row.get("average_predicted_uplift") if selected_row else None

    render_metric_row(
        [
            ("Selected Model", str(selected_model), "Chosen by top-30% policy value"),
            ("Top 30% Est. Incremental Conversions", f"{float(top_30_incremental_conversions):.1f}" if top_30_incremental_conversions is not None and not pd.isna(top_30_incremental_conversions) else "N/A", "Estimated conversions added by targeting top uplift users"),
            ("Top Decile Observed Uplift", format_percent(top_decile_uplift), "Observed lift in the highest-ranked decile"),
            ("Qini Coef.", f"{float(qini_coefficient):.3f}" if qini_coefficient is not None and not pd.isna(qini_coefficient) else "N/A", "Area above random targeting"),
            ("Avg Predicted Uplift", format_percent(average_predicted_uplift), "Used as a ranking score, not a calibrated causal probability"),
        ]
    )

    render_section_title("Model and Policy Tables")
    render_table(comparison_df, expander="View T-Learner / S-Learner comparison")
    render_table(policy_df, expander="View policy value comparison")

    render_section_title(
        "Ranking and Calibration",
        "Qini asks whether targeting beats random. Calibration asks whether predicted uplift magnitude matches observed uplift.",
    )
    show_chart_grid(
        [
            FIGURES_DIR / "qini_curve.png",
            FIGURES_DIR / "uplift_calibration_by_decile.png",
        ],
        captions={
            "qini_curve.png": "Qini evaluates whether uplift ranking beats random targeting.",
            "uplift_calibration_by_decile.png": "Calibration shows predicted uplift is useful mainly as a ranking signal.",
        },
        columns_per_row=1,
    )
    render_table(calibration_df, expander="View uplift calibration table")

    render_section_title(
        "Uplift Diagnostics",
        "Decile, cumulative, distribution, and policy charts show how the ranking behaves.",
    )
    show_chart_grid(
        [
            FIGURES_DIR / "uplift_decile_bar.png",
            FIGURES_DIR / "cumulative_uplift_curve.png",
            FIGURES_DIR / "predicted_uplift_distribution.png",
            FIGURES_DIR / "uplift_policy_comparison.png",
        ],
        captions={
            "uplift_decile_bar.png": "Observed decile lift is noisy because conversion is rare.",
            "cumulative_uplift_curve.png": "Cumulative lift shows whether targeting concentrates incremental conversions.",
            "predicted_uplift_distribution.png": "Predicted uplift spread should be interpreted as ranking strength, not exact causality.",
            "uplift_policy_comparison.png": "Policy value translates model ranking into campaign targeting value.",
        },
        columns_per_row=1,
    )

    render_callout(
        "Uplift takeaway: use the score to rank customers and design tests, not to claim exact individual-level causal probabilities.",
        "recommendation",
    )
    render_footer()


def page_robustness_checks():
    """Render robustness checks for uplift findings."""
    render_page_header(
        "Robustness Checks",
        "Pressure-test the uplift story across an alternative method, interpretable segments, and treatment definitions.",
    )
    render_callout(
        "This step is not meant to search for a flattering result. Weak or noisy uplift is still useful because it shows "
        "where targeting signal may be limited.",
        "caution",
    )

    method_df = load_csv_safe(ROBUSTNESS_DIR / "uplift_method_robustness_comparison.csv")
    segment_df = load_csv_safe(ROBUSTNESS_DIR / "segment_heterogeneity_checks.csv")
    treatment_df = load_csv_safe(ROBUSTNESS_DIR / "treatment_definition_comparison.csv")

    render_section_title(
        "X-Learner Result",
        "The X-Learner is an exploratory robustness method; it does not replace the main selected uplift model.",
    )
    show_chart_grid(
        [
            FIGURES_DIR / "x_learner_qini_curve.png",
            FIGURES_DIR / "x_learner_calibration_by_decile.png",
        ],
        captions={
            "x_learner_qini_curve.png": "The X-Learner acts as a pressure test, not a replacement for the main model.",
            "x_learner_calibration_by_decile.png": "Alternative-method calibration checks whether uplift magnitude remains plausible.",
        },
        columns_per_row=1,
    )

    render_section_title("Uplift Method Comparison")
    render_chart(
        FIGURES_DIR / "uplift_method_robustness_comparison.png",
        takeaway="Robustness checks compare methods without forcing a stronger targeting conclusion.",
    )
    render_table(method_df, expander="View method robustness table")

    render_section_title(
        "Segment Heterogeneity",
        "Direct segment checks compare observed treatment-control lift inside business-readable groups. Small cells are flagged.",
    )
    render_chart(
        FIGURES_DIR / "segment_heterogeneity_checks.png",
        takeaway="Segment checks reveal where response differs, while small cells should be read cautiously.",
    )
    render_table(segment_df, expander="View segment heterogeneity table")

    render_section_title(
        "Treatment Definition Comparison",
        "Mens E-Mail remains the primary setup. Womens E-Mail and Any E-Mail are exploratory robustness checks.",
    )
    render_chart(
        FIGURES_DIR / "treatment_definition_comparison.png",
        takeaway="Mens E-Mail remains the main treatment definition; other definitions are robustness context.",
    )
    render_table(treatment_df, expander="View treatment definition comparison")

    render_callout(
        "Final interpretation: if fine-grained targeting does not strongly beat random targeting, the business lesson is still valuable. "
        "The campaign may create broad lift while individual-level targeting needs stronger signal or richer features.",
        "evidence",
    )
    render_footer()


def page_causal_validation():
    """Render the causal validation page."""
    render_page_header(
        "Causal Validation",
        "Balance, propensity, confidence interval, and DoWhy checks support responsible treatment-effect interpretation.",
    )
    render_callout(
        "Because balance checks show very low SMD and propensity AUC near 0.5, the observed treatment-control "
        "difference is more credible than a generic observational comparison.",
        "evidence",
    )

    ate_results = load_json_safe(CAUSAL_DIR / "naive_ate.json") or {}
    ate_ci = load_json_safe(CAUSAL_DIR / "ate_confidence_interval.json")
    if not ate_ci:
        ate_ci = load_json_safe(UPLIFT_DIR / "ate_confidence_interval.json") or {}
    balance_df = load_csv_safe(CAUSAL_DIR / "covariate_balance.csv")
    dowhy_results = load_json_safe(CAUSAL_DIR / "dowhy_results.json") or {}

    largest_smd = None
    if balance_df is not None and not balance_df.empty:
        largest_smd = balance_df["abs_smd"].max()

    ci_value = "N/A"
    p_value = "N/A"
    if ate_ci:
        ci_value = (
            f"{format_percent(ate_ci.get('ci_lower'))} to "
            f"{format_percent(ate_ci.get('ci_upper'))}"
        )
        p_value = f"{float(ate_ci.get('p_value', 0)):.4f}"

    refutations = dowhy_results.get("refutations") if isinstance(dowhy_results, dict) else None
    dowhy_refuter_value = str(dowhy_results.get("status", "N/A"))
    if isinstance(refutations, dict) and refutations:
        dowhy_refuter_value = f"{len(refutations)} checks"

    render_metric_row(
        [
            ("Observed ATE", format_pp(ate_results.get("observed_ate", ate_results.get("naive_ate"))), "Validated after balance checks"),
            ("Confidence Interval", ci_value, "ATE uncertainty interval"),
            ("p-value", p_value, "Statistical uncertainty for the observed ATE"),
            ("Propensity AUC", f"{get_propensity_auc():.3f}", "Treatment assignment predictability"),
            ("Largest SMD", f"{largest_smd:.3f}" if largest_smd is not None else "N/A", "Covariate balance diagnostic"),
            ("DoWhy Refuters", dowhy_refuter_value, "Optional causal validation checks"),
        ]
    )

    if ate_ci:
        render_callout(
            "ATE confidence interval: "
            f"<b>{format_percent(ate_ci.get('ci_lower'))}</b> to "
            f"<b>{format_percent(ate_ci.get('ci_upper'))}</b>; "
            f"p-value = <b>{float(ate_ci.get('p_value', 0)):.4f}</b>",
            "evidence",
        )

    render_section_title("ATE and Balance Charts")
    show_chart_grid(
        [
            FIGURES_DIR / "ate_confidence_interval.png",
            FIGURES_DIR / "covariate_balance_smd.png",
            FIGURES_DIR / "propensity_score_overlap.png",
            FIGURES_DIR / "treatment_assignment_predictability.png",
            FIGURES_DIR / "causal_ate_summary.png",
        ],
        captions={
            "ate_confidence_interval.png": "The observed ATE is framed with uncertainty, not just a point estimate.",
            "covariate_balance_smd.png": "Small standardized mean differences support treatment/control comparability.",
            "propensity_score_overlap.png": "Overlap supports comparing treated and control customers on common support.",
            "treatment_assignment_predictability.png": "Low treatment predictability is consistent with randomized-like assignment.",
            "causal_ate_summary.png": "Campaign impact remains the central causal quantity.",
        },
        columns_per_row=1,
    )
    render_table(balance_df, max_rows=20, expander="View top covariate balance rows")
    _show_dowhy_refutations(dowhy_results)

    render_callout(
        "Causal validation takeaway: the campaign-level effect is credible, while individual-level uplift still needs cautious interpretation.",
        "evidence",
    )
    render_footer()


def page_final_recommendation():
    """Render the final recommendation page."""
    render_page_header(
        "Final Recommendation",
        "A business-facing summary of what PromoLift AI recommends and what should be tested next.",
    )

    cards = [
        {
            "title": "What the campaign did",
            "body": "Mens E-Mail created measurable conversion lift against the holdout.",
        },
        {
            "title": "What targeting should do",
            "body": "Use uplift scores as a ranking input for controlled targeting tests.",
        },
        {
            "title": "What targeting should not overclaim",
            "body": "Do not treat individual uplift scores as perfect causal probabilities.",
        },
        {
            "title": "What to test next",
            "body": "Run a prospective campaign test using top-uplift segments and holdout cells.",
        },
        {
            "title": "Limitations",
            "body": "Individual counterfactual outcomes are estimated, not directly observed.",
        },
        {
            "title": "Data limitation",
            "body": "Hillstrom features capture broad customer history but not richer behavioral signals such as email clicks, browsing intent, device type, send timing, or long-term value. This may explain why campaign-level lift is measurable while individual-level targeting signal remains modest.",
        },
        {
            "title": "Future benchmark",
            "body": "A future extension could run the same Qini, calibration, robustness, and causal-validation pipeline on a larger uplift benchmark dataset to test whether stronger heterogeneity appears when richer incrementality data is available.",
        },
    ]
    render_recommendation_cards(cards)

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
    render_table(comparison, title="Strategy comparison")

    render_callout(
        "Portfolio pitch: PromoLift AI shows how to move from conversion prediction to causal campaign decision-making, "
        "with EDA, uplift modeling, robustness checks, causal validation, and executive storytelling in one workflow.",
        "recommendation",
    )
    render_footer()


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


def _show_dowhy_refutations(dowhy_results):
    """Display DoWhy refutation details when available."""
    refutations = dowhy_results.get("refutations") if isinstance(dowhy_results, dict) else None
    if not refutations:
        return

    render_section_title(
        "DoWhy Refutation Checks",
        "Refuters check whether the estimated effect is stable under simple stress tests.",
    )
    refuter_labels = {
        "random_common_cause": "Random common cause",
        "placebo_treatment_refuter": "Placebo treatment",
        "subset_refuter": "Subset refuter",
    }
    rows = []

    for refuter_name, label in refuter_labels.items():
        result = refutations.get(refuter_name)
        if isinstance(result, dict):
            rows.append(
                {
                    "Refuter": label,
                    "Status": result.get("status", "unknown"),
                    "Method": result.get("method_name", refuter_name),
                }
            )
        elif result is not None:
            rows.append(
                {
                    "Refuter": label,
                    "Status": "completed",
                    "Method": refuter_name,
                }
            )

    if rows:
        render_table(pd.DataFrame(rows))
        render_callout(
            "Random common cause adds noise as a fake confounder; placebo treatment checks whether a fake treatment removes the effect; "
            "subset refutation checks stability when supported by the installed DoWhy version.",
            "neutral",
        )


def render_sidebar(pages):
    """Render the polished sidebar navigation."""
    with st.sidebar:
        st.markdown('<div class="pl-sidebar-title">PromoLift AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pl-sidebar-tagline">Causal uplift modeling for smarter coupon and email targeting.</div>',
            unsafe_allow_html=True,
        )
        selected_page = st.radio("Page navigation", list(pages.keys()))
        st.divider()
        st.markdown(
            (
                '<div class="pl-sidebar-workflow-title">Workflow</div>'
                '<ol class="pl-sidebar-workflow">'
                '<li>Experiment setup</li>'
                '<li>Baseline model</li>'
                '<li>Uplift modeling</li>'
                '<li>Robustness checks</li>'
                '<li>Causal validation</li>'
                '<li>Business recommendation</li>'
                '</ol>'
            ),
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("Built for portfolio review and business storytelling.")

    return selected_page


def main():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="PromoLift AI | Causal Uplift Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_visual_system()

    pages = {
        "Executive Overview": page_executive_overview,
        "Dataset & Experiment": page_dataset_experiment,
        "Causal EDA": page_causal_eda,
        "Baseline ML Model": page_baseline_ml_model,
        "Uplift Modeling": page_uplift_modeling,
        "Robustness Checks": page_robustness_checks,
        "Causal Validation": page_causal_validation,
        "Final Recommendation": page_final_recommendation,
    }

    selected_page = render_sidebar(pages)
    pages[selected_page]()


if __name__ == "__main__":
    main()
