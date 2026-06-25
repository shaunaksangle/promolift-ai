"""Causal EDA for the Hillstrom Email Marketing experiment.

This module creates causal-inference-focused EDA artifacts for PromoLift AI.
It inspects treatment balance, propensity overlap, stratified effect estimates,
subgroup heterogeneity, a simple causal DAG, and leakage considerations.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.features.build_features import build_preprocessor, get_feature_columns


PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "hillstrom_mens_email.csv"
CAUSAL_EDA_REPORTS_DIR = REPORTS_DIR / "causal_eda"

GROUP_LABELS = {
    0: "Control: No E-Mail",
    1: "Treatment: Mens E-Mail",
}
CONTROL_LABEL = GROUP_LABELS[0]
TREATMENT_LABEL = GROUP_LABELS[1]
GROUP_ORDER = [CONTROL_LABEL, TREATMENT_LABEL]
GROUP_COLORS = {
    CONTROL_LABEL: "#64748B",
    TREATMENT_LABEL: "#0F766E",
}

NUMERIC_OVERLAP_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_BALANCE_FEATURES = ["channel", "history_segment", "newbie_label"]
STRATIFICATION_FEATURES = ["channel", "history_segment", "newbie"]

TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#475569"
GRID_COLOR = "#E5E7EB"
POSITIVE_COLOR = "#15803D"
NEGATIVE_COLOR = "#B91C1C"
ACCENT_COLOR = "#2563EB"
WARNING_COLOR = "#B45309"


def load_processed_data() -> pd.DataFrame:
    """Load the processed binary treatment Hillstrom dataset."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed Hillstrom dataset was not found at "
            f"{PROCESSED_DATA_PATH}. Run this command first: "
            "python -m src.data.load_hillstrom"
        )

    return pd.read_csv(PROCESSED_DATA_PATH)


def prepare_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable labels used in causal EDA plots."""
    plot_df = df.copy()
    plot_df["group"] = plot_df["treatment"].map(GROUP_LABELS)
    plot_df["newbie_label"] = plot_df["newbie"].map({0: "Existing Customer", 1: "New Customer"})

    return plot_df


def calculate_naive_ate(df: pd.DataFrame) -> dict[str, float]:
    """Calculate the unadjusted treatment-control conversion difference."""
    treatment_rate = df.loc[df["treatment"] == 1, "outcome"].mean()
    control_rate = df.loc[df["treatment"] == 0, "outcome"].mean()
    ate = treatment_rate - control_rate

    return {
        "treatment_conversion_rate": float(treatment_rate),
        "control_conversion_rate": float(control_rate),
        "effect_estimate": float(ate),
        "effect_percentage_points": float(ate * 100),
    }


def estimate_propensity_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Estimate treatment assignment probabilities from pre-campaign features."""
    numeric_features, categorical_features = get_feature_columns()
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns].copy()
    treatment = df["treatment"].astype(int)

    propensity_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    propensity_model.fit(X, treatment)

    propensity_score = propensity_model.predict_proba(X)[:, 1]
    propensity_auc = float(roc_auc_score(treatment, propensity_score))

    propensity_df = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "treatment": treatment,
            "group": treatment.map(GROUP_LABELS),
            "propensity_score": propensity_score,
        }
    )

    return propensity_df, propensity_auc


def summarize_propensity_overlap(
    propensity_df: pd.DataFrame,
    propensity_auc: float,
) -> pd.DataFrame:
    """Summarize propensity overlap by treatment group."""
    treatment_scores = propensity_df.loc[
        propensity_df["treatment"] == 1,
        "propensity_score",
    ]
    control_scores = propensity_df.loc[
        propensity_df["treatment"] == 0,
        "propensity_score",
    ]

    common_support_min = max(treatment_scores.min(), control_scores.min())
    common_support_max = min(treatment_scores.max(), control_scores.max())

    rows = []
    for treatment_value, group_name in GROUP_LABELS.items():
        scores = propensity_df.loc[
            propensity_df["treatment"] == treatment_value,
            "propensity_score",
        ]
        share_in_common_support = scores.between(
            common_support_min,
            common_support_max,
            inclusive="both",
        ).mean()
        rows.append(
            {
                "group": group_name,
                "customers": int(scores.shape[0]),
                "mean_propensity": float(scores.mean()),
                "median_propensity": float(scores.median()),
                "p05_propensity": float(scores.quantile(0.05)),
                "p95_propensity": float(scores.quantile(0.95)),
                "min_propensity": float(scores.min()),
                "max_propensity": float(scores.max()),
                "share_in_common_support": float(share_in_common_support),
                "common_support_min": float(common_support_min),
                "common_support_max": float(common_support_max),
                "propensity_auc": float(propensity_auc),
            }
        )

    return pd.DataFrame(rows)


def calculate_naive_vs_stratified_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate naive and segment-adjusted treatment effect estimates."""
    naive = calculate_naive_ate(df)
    rows = [
        {
            "estimate_type": "Naive ATE",
            "adjustment_variable": "None",
            "effect_estimate": naive["effect_estimate"],
            "effect_percentage_points": naive["effect_percentage_points"],
            "segments_used": 1,
            "customers": int(df.shape[0]),
            "interpretation": "Unadjusted randomized experiment estimate",
        }
    ]

    for variable in STRATIFICATION_FEATURES:
        segment_table = build_subgroup_uplift_table(df, variable)
        valid_segments = segment_table.dropna(subset=["segment_uplift"]).copy()

        if valid_segments.empty:
            adjusted_effect = float("nan")
        else:
            total_customers = valid_segments["customers"].sum()
            valid_segments["segment_weight"] = valid_segments["customers"] / total_customers
            adjusted_effect = (
                valid_segments["segment_uplift"] * valid_segments["segment_weight"]
            ).sum()

        rows.append(
            {
                "estimate_type": "Stratified-adjusted effect estimate",
                "adjustment_variable": variable,
                "effect_estimate": float(adjusted_effect),
                "effect_percentage_points": float(adjusted_effect * 100),
                "segments_used": int(valid_segments.shape[0]),
                "customers": int(valid_segments["customers"].sum()),
                "interpretation": "Weighted segment-level treatment-control difference",
            }
        )

    return pd.DataFrame(rows)


def build_subgroup_uplift_table(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    """Calculate treatment-control conversion differences by segment."""
    rows = []

    for segment_value, segment_df in df.groupby(segment_column, dropna=False):
        treatment_df = segment_df[segment_df["treatment"] == 1]
        control_df = segment_df[segment_df["treatment"] == 0]

        treatment_rate = (
            treatment_df["outcome"].mean() if not treatment_df.empty else float("nan")
        )
        control_rate = (
            control_df["outcome"].mean() if not control_df.empty else float("nan")
        )
        segment_uplift = treatment_rate - control_rate

        rows.append(
            {
                "segment_variable": segment_column,
                "segment_value": str(segment_value),
                "customers": int(segment_df.shape[0]),
                "treatment_customers": int(treatment_df.shape[0]),
                "control_customers": int(control_df.shape[0]),
                "treatment_conversion_rate": float(treatment_rate),
                "control_conversion_rate": float(control_rate),
                "segment_uplift": float(segment_uplift),
                "segment_uplift_pp": float(segment_uplift * 100),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("segment_uplift", ascending=False)
        .reset_index(drop=True)
    )


def create_causal_eda_plots(
    df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    effects_df: pd.DataFrame,
    subgroup_tables: dict[str, pd.DataFrame],
) -> None:
    """Create all causal EDA charts."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        from matplotlib.ticker import FuncFormatter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Plotting libraries are missing. Install requirements with: "
            "pip install -r requirements.txt"
        ) from error

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _set_theme(plt, sns)

    plot_df = prepare_plot_data(df)

    _plot_numeric_overlap(plt, sns, FuncFormatter, plot_df)
    _plot_categorical_balance(plt, sns, FuncFormatter, plot_df)
    _plot_propensity_overlap(plt, sns, FuncFormatter, propensity_df, overlap_summary)
    _plot_naive_vs_adjusted_effect(plt, FuncFormatter, effects_df)
    _plot_subgroup_uplift(plt, FuncFormatter, subgroup_tables)
    _plot_heterogeneity_heatmap(plt, sns, FuncFormatter, df)
    _plot_causal_dag(plt, FancyArrowPatch, FancyBboxPatch)


def write_causal_eda_summary(
    df: pd.DataFrame,
    propensity_auc: float,
    overlap_summary: pd.DataFrame,
    effects_df: pd.DataFrame,
    subgroup_tables: dict[str, pd.DataFrame],
) -> None:
    """Write a portfolio-ready causal EDA markdown report."""
    CAUSAL_EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    naive_row = effects_df.iloc[0]
    adjusted_rows = effects_df.iloc[1:]
    adjusted_bullets = "\n".join(
        "- "
        f"{row.adjustment_variable}: {_format_pp(row.effect_estimate)} "
        f"({row.segments_used} segments)"
        for row in adjusted_rows.itertuples(index=False)
    )

    combined_subgroups = pd.concat(subgroup_tables.values(), ignore_index=True)
    valid_subgroups = combined_subgroups.dropna(subset=["segment_uplift"])
    strongest_segment = valid_subgroups.sort_values(
        "segment_uplift",
        ascending=False,
    ).iloc[0]
    weakest_segment = valid_subgroups.sort_values("segment_uplift").iloc[0]

    common_support_share = overlap_summary["share_in_common_support"].min()

    report_text = f"""# Causal EDA Summary

## Why Causal EDA Is Different From Generic EDA

Generic EDA describes distributions and correlations. Causal EDA asks whether a treatment/control comparison is credible enough to support campaign-impact reasoning. For PromoLift AI, that means inspecting treatment balance, overlap, potential selection bias, subgroup heterogeneity, and leakage risks before interpreting uplift scores.

## Treatment Balance and Covariate Overlap

The balance plots compare treated and control customers on pre-campaign covariates: recency, purchase history, mens/womens indicators, newbie status, channel, and history segment. Because the Hillstrom dataset comes from a marketing experiment, the groups should look similar before treatment. The causal EDA charts help make that assumption visible rather than implicit.

Figures:

- `reports/figures/causal_eda_numeric_overlap.png`
- `reports/figures/causal_eda_categorical_balance.png`

## Propensity Score Overlap

The propensity model predicts treatment assignment using only pre-campaign features. A propensity AUC near 0.5 suggests treatment assignment is difficult to predict from observed customer features, which is consistent with randomized assignment.

- Propensity model AUC: {propensity_auc:.3f}
- Minimum group share inside common support: {common_support_share:.2%}

Figures and tables:

- `reports/figures/causal_eda_propensity_overlap.png`
- `reports/causal_eda/propensity_overlap_summary.csv`

## Naive vs Stratified-Adjusted Effect Comparison

The naive ATE is the treatment conversion rate minus the control conversion rate. Because Hillstrom is a randomized experiment, this naive estimate is already meaningful. The stratified-adjusted estimates are robustness checks that weight segment-level treatment effects by segment size.

- Naive ATE: {_format_pp(naive_row.effect_estimate)}

Stratified-adjusted effect estimates:

{adjusted_bullets}

These are not claimed to be perfect causal adjustments. They are segment-adjusted checks that help show whether the average effect is broadly stable across important business groupings.

Figure and table:

- `reports/figures/causal_eda_naive_vs_adjusted_effect.png`
- `reports/causal_eda/naive_vs_stratified_effects.csv`

## Subgroup Heterogeneity

Average treatment effects can hide important business differences. In this project, subgroup uplift by channel, history segment, and newbie status motivates uplift modeling because the best campaign decision may differ across customer types.

- Strongest observed subgroup uplift: `{strongest_segment.segment_variable} = {strongest_segment.segment_value}` at {_format_pp(strongest_segment.segment_uplift)}
- Weakest observed subgroup uplift: `{weakest_segment.segment_variable} = {weakest_segment.segment_value}` at {_format_pp(weakest_segment.segment_uplift)}

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
"""

    output_path = CAUSAL_EDA_REPORTS_DIR / "causal_eda_summary.md"
    output_path.write_text(report_text, encoding="utf-8")
    print(f"Saved causal EDA report to: {output_path}")


def _set_theme(plt, sns) -> None:
    """Apply a clean visual theme to causal EDA plots."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "figure.titlesize": 20,
            "figure.titleweight": "bold",
            "font.family": "sans-serif",
            "axes.edgecolor": GRID_COLOR,
            "grid.color": GRID_COLOR,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": MUTED_TEXT_COLOR,
            "ytick.color": MUTED_TEXT_COLOR,
        }
    )


def _plot_numeric_overlap(plt, sns, FuncFormatter, plot_df: pd.DataFrame) -> None:
    """Plot numeric covariate overlap between treatment and control."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, feature in zip(axes, NUMERIC_OVERLAP_FEATURES):
        is_binary = feature in {"mens", "womens", "newbie"}
        sns.histplot(
            data=plot_df,
            x=feature,
            hue="group",
            hue_order=GROUP_ORDER,
            stat="density",
            common_norm=False,
            element="step",
            fill=True,
            alpha=0.25,
            bins=[-0.5, 0.5, 1.5] if is_binary else 24,
            discrete=is_binary,
            palette=GROUP_COLORS,
            ax=ax,
        )
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Pre-campaign value")
        ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.35)

    axes[-1].axis("off")
    fig.suptitle("Treatment Balance: Pre-Campaign Numeric Covariates")
    fig.text(
        0.5,
        0.93,
        "Do Mens E-Mail and No E-Mail customers look similar before treatment?",
        ha="center",
        color=MUTED_TEXT_COLOR,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(FIGURES_DIR / "causal_eda_numeric_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_categorical_balance(plt, sns, FuncFormatter, plot_df: pd.DataFrame) -> None:
    """Plot categorical covariate balance between treatment and control."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 13))

    for ax, feature in zip(axes, CATEGORICAL_BALANCE_FEATURES):
        balance_df = (
            plot_df.groupby(["group", feature])
            .size()
            .reset_index(name="customers")
        )
        balance_df["share"] = balance_df.groupby("group")["customers"].transform(
            lambda values: values / values.sum()
        )

        sns.barplot(
            data=balance_df,
            x="share",
            y=feature,
            hue="group",
            hue_order=GROUP_ORDER,
            palette=GROUP_COLORS,
            ax=ax,
        )
        ax.set_title(feature.replace("_label", "").replace("_", " ").title())
        ax.set_xlabel("Share Within Treatment Group")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.legend(title="", loc="lower right")
        ax.grid(axis="x", alpha=0.35)

    fig.suptitle("Treatment Balance: Categorical Customer Segments")
    fig.text(
        0.5,
        0.94,
        "Similar category shares support a credible treatment/control comparison.",
        ha="center",
        color=MUTED_TEXT_COLOR,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(
        FIGURES_DIR / "causal_eda_categorical_balance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_propensity_overlap(
    plt,
    sns,
    FuncFormatter,
    propensity_df: pd.DataFrame,
    overlap_summary: pd.DataFrame,
) -> None:
    """Plot propensity score overlap between treatment and control."""
    common_support_min = overlap_summary["common_support_min"].iloc[0]
    common_support_max = overlap_summary["common_support_max"].iloc[0]
    propensity_auc = overlap_summary["propensity_auc"].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(
        data=propensity_df,
        x="propensity_score",
        hue="group",
        hue_order=GROUP_ORDER,
        stat="density",
        common_norm=False,
        bins=35,
        alpha=0.35,
        palette=GROUP_COLORS,
        ax=ax,
    )
    ax.axvspan(
        common_support_min,
        common_support_max,
        color=ACCENT_COLOR,
        alpha=0.08,
        label="Common support",
    )
    ax.set_title("Propensity Overlap: Can We Compare Treated and Control Customers?")
    ax.text(
        0.02,
        0.95,
        f"Propensity AUC = {propensity_auc:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EFF6FF", "edgecolor": "#BFDBFE"},
    )
    ax.set_xlabel("Estimated Probability of Receiving Mens E-Mail")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "causal_eda_propensity_overlap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_naive_vs_adjusted_effect(plt, FuncFormatter, effects_df: pd.DataFrame) -> None:
    """Plot naive ATE against stratified-adjusted estimates."""
    plot_df = effects_df.copy()
    plot_df["label"] = plot_df.apply(
        lambda row: "Naive ATE"
        if row["adjustment_variable"] == "None"
        else f"Adjusted by {row['adjustment_variable']}",
        axis=1,
    )
    plot_df = plot_df.sort_values("effect_estimate", ascending=True)
    colors = [
        ACCENT_COLOR if label == "Naive ATE" else POSITIVE_COLOR
        for label in plot_df["label"]
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(plot_df["label"], plot_df["effect_estimate"], color=colors)
    ax.axvline(0, color=TEXT_COLOR, linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.2f} pp"))
    ax.set_title("Naive ATE vs Segment-Adjusted Treatment Effects")
    ax.set_xlabel("Conversion Lift")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.35)

    for bar, value in zip(bars, plot_df["effect_estimate"]):
        label_x = value + 0.00025 if value >= 0 else value - 0.00025
        ha = "left" if value >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value * 100:.2f} pp",
            va="center",
            ha=ha,
            fontsize=11,
            color=TEXT_COLOR,
        )

    fig.text(
        0.5,
        0.01,
        "Because Hillstrom is randomized, naive ATE is meaningful; stratified estimates are robustness checks.",
        ha="center",
        color=MUTED_TEXT_COLOR,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(
        FIGURES_DIR / "causal_eda_naive_vs_adjusted_effect.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_subgroup_uplift(
    plt,
    FuncFormatter,
    subgroup_tables: dict[str, pd.DataFrame],
) -> None:
    """Plot subgroup-level treatment effects."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, (segment_variable, table) in zip(axes, subgroup_tables.items()):
        plot_table = table.dropna(subset=["segment_uplift"]).copy()
        plot_table = plot_table.sort_values("segment_uplift", ascending=True)
        colors = [
            POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR
            for value in plot_table["segment_uplift"]
        ]
        bars = ax.barh(plot_table["segment_value"], plot_table["segment_uplift"], color=colors)
        ax.axvline(0, color=TEXT_COLOR, linewidth=1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.1f} pp"))
        ax.set_title(segment_variable.replace("_", " ").title())
        ax.set_xlabel("Treatment-Control Conversion Lift")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.35)

        for bar, value in zip(bars, plot_table["segment_uplift"]):
            label_x = value + 0.00025 if value >= 0 else value - 0.00025
            ha = "left" if value >= 0 else "right"
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{value * 100:.2f} pp",
                va="center",
                ha=ha,
                fontsize=9,
            )

    fig.suptitle("Subgroup Heterogeneity: Campaign Lift Differs by Customer Segment")
    fig.text(
        0.5,
        0.91,
        "Heterogeneous treatment effects motivate ranking customers by uplift, not only modeling the average effect.",
        ha="center",
        color=MUTED_TEXT_COLOR,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(
        FIGURES_DIR / "causal_eda_subgroup_uplift.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_heterogeneity_heatmap(plt, sns, FuncFormatter, df: pd.DataFrame) -> None:
    """Plot two-way treatment-effect heterogeneity."""
    rows = []

    for (channel, history_segment), segment_df in df.groupby(["channel", "history_segment"]):
        treatment_df = segment_df[segment_df["treatment"] == 1]
        control_df = segment_df[segment_df["treatment"] == 0]

        if treatment_df.empty or control_df.empty:
            uplift = float("nan")
        else:
            uplift = treatment_df["outcome"].mean() - control_df["outcome"].mean()

        rows.append(
            {
                "channel": channel,
                "history_segment": history_segment,
                "uplift_pp": uplift * 100,
            }
        )

    heatmap_df = pd.DataFrame(rows).pivot(
        index="channel",
        columns="history_segment",
        values="uplift_pp",
    )

    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Conversion Lift (percentage points)"},
        ax=ax,
    )
    ax.set_title("Two-Way Heterogeneity: Channel x History Segment")
    ax.set_xlabel("History Segment")
    ax.set_ylabel("Channel")
    fig.text(
        0.5,
        0.01,
        "Cells show treatment-control conversion lift in percentage points.",
        ha="center",
        color=MUTED_TEXT_COLOR,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(
        FIGURES_DIR / "causal_eda_heterogeneity_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_causal_dag(plt, FancyArrowPatch, FancyBboxPatch) -> None:
    """Draw a simple causal DAG sketch with matplotlib annotations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "covariates": {
            "xy": (0.08, 0.58),
            "text": "Pre-Campaign Customer Features\nhistory, recency, channel, segment",
            "color": "#E0F2FE",
        },
        "treatment": {
            "xy": (0.44, 0.63),
            "text": "Treatment Assignment\nMens E-Mail vs No E-Mail",
            "color": "#DCFCE7",
        },
        "outcome": {
            "xy": (0.74, 0.58),
            "text": "Post-Campaign Outcome\nconversion",
            "color": "#FEF3C7",
        },
        "unobserved": {
            "xy": (0.30, 0.18),
            "text": "Unobserved Factors\npreferences, timing, intent",
            "color": "#FEE2E2",
        },
    }

    box_size = (0.22, 0.16)
    for node in nodes.values():
        _draw_node(ax, FancyBboxPatch, node["xy"], box_size, node["text"], node["color"])

    _draw_arrow(ax, FancyArrowPatch, (0.30, 0.66), (0.44, 0.70))
    _draw_arrow(ax, FancyArrowPatch, (0.30, 0.60), (0.74, 0.64))
    _draw_arrow(ax, FancyArrowPatch, (0.66, 0.70), (0.74, 0.66))
    _draw_arrow(ax, FancyArrowPatch, (0.52, 0.30), (0.74, 0.58))
    _draw_arrow(ax, FancyArrowPatch, (0.42, 0.34), (0.44, 0.63), dashed=True)

    ax.text(
        0.5,
        0.95,
        "Causal DAG Sketch for PromoLift AI",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        0.5,
        0.90,
        "Pre-campaign features are used for balance checks and modeling; post-campaign outcomes are never used as features.",
        ha="center",
        va="top",
        fontsize=12,
        color=MUTED_TEXT_COLOR,
    )
    ax.text(
        0.5,
        0.08,
        "Dashed arrow marks a limitation/check: unobserved factors could affect treatment assignment, so balance and overlap are inspected.",
        ha="center",
        fontsize=11,
        color=WARNING_COLOR,
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "causal_eda_dag.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_node(ax, FancyBboxPatch, xy, box_size, text, color) -> None:
    """Draw one rounded node in the DAG."""
    width, height = box_size
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#CBD5E1",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT_COLOR,
    )


def _draw_arrow(ax, FancyArrowPatch, start, end, dashed=False) -> None:
    """Draw one causal arrow in the DAG."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.8,
        linestyle="--" if dashed else "-",
        color=WARNING_COLOR if dashed else TEXT_COLOR,
    )
    ax.add_patch(arrow)


def _format_pp(value: float) -> str:
    """Format an effect as percentage points."""
    return f"{float(value) * 100:.2f} percentage points"


def main() -> None:
    """Run the causal EDA workflow."""
    CAUSAL_EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed_data()
    propensity_df, propensity_auc = estimate_propensity_scores(df)
    overlap_summary = summarize_propensity_overlap(propensity_df, propensity_auc)
    effects_df = calculate_naive_vs_stratified_effects(df)
    subgroup_tables = {
        segment: build_subgroup_uplift_table(df, segment)
        for segment in STRATIFICATION_FEATURES
    }

    overlap_path = CAUSAL_EDA_REPORTS_DIR / "propensity_overlap_summary.csv"
    effects_path = CAUSAL_EDA_REPORTS_DIR / "naive_vs_stratified_effects.csv"
    overlap_summary.to_csv(overlap_path, index=False)
    effects_df.to_csv(effects_path, index=False)

    create_causal_eda_plots(
        df,
        propensity_df,
        overlap_summary,
        effects_df,
        subgroup_tables,
    )
    write_causal_eda_summary(
        df,
        propensity_auc,
        overlap_summary,
        effects_df,
        subgroup_tables,
    )

    print(f"Saved propensity overlap summary to: {overlap_path}")
    print(f"Saved naive vs stratified effects to: {effects_path}")
    print(f"Saved causal EDA figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
