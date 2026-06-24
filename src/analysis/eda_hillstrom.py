"""Exploratory data analysis for the Hillstrom Email Marketing dataset.

This script summarizes the processed Mens E-Mail versus No E-Mail experiment,
creates segment-level uplift tables, saves charts, and writes a concise EDA
report for the PromoLift AI project.
"""

import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR


PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "hillstrom_mens_email.csv"
EDA_REPORTS_DIR = REPORTS_DIR / "eda"

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
POSITIVE_UPLIFT_COLOR = "#15803D"
NEGATIVE_UPLIFT_COLOR = "#B91C1C"
TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#475569"
GRID_COLOR = "#E5E7EB"


def load_processed_data() -> pd.DataFrame:
    """Load the processed Hillstrom binary treatment dataset.

    Raises:
        FileNotFoundError: If the processed dataset has not been created yet.

    Returns:
        Processed Hillstrom data.
    """
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed Hillstrom dataset was not found at "
            f"{PROCESSED_DATA_PATH}. Run this command first: "
            "python -m src.data.load_hillstrom"
        )

    return pd.read_csv(PROCESSED_DATA_PATH)


def summarize_dataset(df: pd.DataFrame) -> None:
    """Print a simple summary of the processed dataset."""
    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nTreatment distribution:")
    print(df["treatment"].map(GROUP_LABELS).value_counts())

    print("\nOutcome distribution:")
    print(df["outcome"].value_counts().sort_index())

    print(f"\nOverall conversion rate: {df['outcome'].mean():.2%}")
    print(f"Overall visit rate: {df['visit'].mean():.2%}")
    print(f"Average spend: ${df['spend'].mean():.2f}")


def calculate_campaign_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Compare campaign results between treatment and control groups.

    Returns:
        A small DataFrame with group-level campaign performance metrics.
    """
    campaign_summary = (
        df.groupby("treatment")
        .agg(
            customers=("customer_id", "count"),
            conversion_rate=("outcome", "mean"),
            visit_rate=("visit", "mean"),
            average_spend=("spend", "mean"),
            total_spend=("spend", "sum"),
        )
        .reset_index()
    )

    campaign_summary["group"] = campaign_summary["treatment"].map(GROUP_LABELS)
    campaign_summary = campaign_summary[
        [
            "group",
            "customers",
            "conversion_rate",
            "visit_rate",
            "average_spend",
            "total_spend",
        ]
    ]

    control = _get_campaign_group(campaign_summary, "Control: No E-Mail")
    treatment = _get_campaign_group(campaign_summary, "Treatment: Mens E-Mail")

    absolute_uplift = treatment["conversion_rate"] - control["conversion_rate"]
    relative_lift = (
        absolute_uplift / control["conversion_rate"]
        if control["conversion_rate"] != 0
        else float("nan")
    )

    print("\nCampaign summary:")
    print(campaign_summary.to_string(index=False))
    print(
        "\nAbsolute conversion uplift: "
        f"{absolute_uplift * 100:.2f} percentage points"
    )
    print(f"Relative conversion lift: {relative_lift:.2%}")

    return campaign_summary


def segment_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create and save segment-level uplift analysis tables.

    Returns:
        Dictionary of segment names and their analysis tables.
    """
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    segment_columns = ["channel", "zip_code", "newbie", "history_segment"]
    saved_tables = {}

    for segment_column in segment_columns:
        segment_table = _build_segment_uplift_table(df, segment_column)
        output_path = EDA_REPORTS_DIR / f"{segment_column}_uplift.csv"
        segment_table.to_csv(output_path, index=False)
        saved_tables[segment_column] = segment_table
        print(f"Saved {segment_column} uplift table to: {output_path}")

    return saved_tables


def create_eda_plots(df: pd.DataFrame, campaign_summary: pd.DataFrame) -> None:
    """Create and save EDA charts for the Hillstrom campaign analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.ticker import FuncFormatter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Plotting libraries are missing. Install the project requirements "
            "with: pip install -r requirements.txt"
        ) from error

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _set_professional_theme(plt, sns)

    plot_data = df.copy()
    plot_data["group"] = plot_data["treatment"].map(GROUP_LABELS)

    channel_uplift = _build_segment_uplift_table(df, "channel")
    history_uplift = _build_segment_uplift_table(df, "history_segment")

    _plot_treatment_distribution(plt, campaign_summary)
    _plot_rate_by_group(
        plt,
        FuncFormatter,
        campaign_summary,
        "conversion_rate",
        "Mens E-Mail Improved Conversion Against the Holdout",
        "Did the Mens E-Mail campaign improve conversion compared to no email?",
        "Conversion Rate",
        "conversion_rate_by_group.png",
    )
    _plot_rate_by_group(
        plt,
        FuncFormatter,
        campaign_summary,
        "visit_rate",
        "Mens E-Mail Increased Customer Visits",
        "Did the email increase customer visits?",
        "Visit Rate",
        "visit_rate_by_group.png",
    )
    _plot_average_spend_by_group(
        plt,
        FuncFormatter,
        campaign_summary,
    )
    _plot_segment_uplift(
        plt,
        FuncFormatter,
        channel_uplift,
        "channel",
        "Channel Uplift: Where the Email Changed Conversion Most",
        "Which acquisition channels responded best to the campaign?",
        "segment_uplift_by_channel.png",
    )
    _plot_segment_uplift(
        plt,
        FuncFormatter,
        history_uplift,
        "history_segment",
        "History Segment Uplift: Customer Value Changes Response",
        "Which customer value/history segments had the highest uplift?",
        "segment_uplift_by_history_segment.png",
    )
    _plot_segment_grouped_rates(
        plt,
        FuncFormatter,
        plot_data,
        channel_uplift,
        "channel",
        "Channel Conversion Rates: Treatment vs Control",
        "How does campaign performance differ across channels?",
        "conversion_rate_treatment_vs_control_by_channel.png",
    )
    _plot_segment_grouped_rates(
        plt,
        FuncFormatter,
        plot_data,
        history_uplift,
        "history_segment",
        "History Segment Conversion Rates: Treatment vs Control",
        "How does campaign performance differ across customer history segments?",
        "conversion_rate_treatment_vs_control_by_history_segment.png",
    )
    _plot_spend_distribution(
        plt,
        sns,
        FuncFormatter,
        plot_data,
    )
    _plot_executive_summary(
        plt,
        FancyBboxPatch,
        df,
        campaign_summary,
    )


def write_eda_summary(df: pd.DataFrame, campaign_summary: pd.DataFrame) -> None:
    """Write a portfolio-ready Markdown summary of the Hillstrom EDA."""
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    control = _get_campaign_group(campaign_summary, "Control: No E-Mail")
    treatment = _get_campaign_group(campaign_summary, "Treatment: Mens E-Mail")

    conversion_uplift = treatment["conversion_rate"] - control["conversion_rate"]
    visit_uplift = treatment["visit_rate"] - control["visit_rate"]
    spend_difference = treatment["average_spend"] - control["average_spend"]
    relative_conversion_lift = (
        conversion_uplift / control["conversion_rate"]
        if control["conversion_rate"] != 0
        else float("nan")
    )

    summary_text = f"""# Hillstrom Email Marketing EDA Summary

## Dataset Overview

The processed Hillstrom dataset contains {len(df):,} customers and {df.shape[1]:,} columns. This first analysis compares customers who received the Mens E-Mail campaign against customers who received No E-Mail.

- Treatment group: Mens E-Mail
- Control group: No E-Mail
- Main outcome: conversion
- Overall conversion rate: {df["outcome"].mean():.2%}
- Overall visit rate: {df["visit"].mean():.2%}
- Average spend: ${df["spend"].mean():.2f}

## Executive Summary

![Executive KPI summary](../figures/eda_executive_summary.png)

The KPI snapshot summarizes the causal comparison at the center of this project: customers exposed to the Mens E-Mail campaign converted at {treatment["conversion_rate"]:.2%}, compared with {control["conversion_rate"]:.2%} in the no-email holdout group. The absolute conversion uplift is {conversion_uplift * 100:.2f} percentage points.

## Treatment vs Control Comparison

{_format_campaign_summary_table(campaign_summary)}

![Experiment balance](../figures/treatment_distribution.png)

The experiment is balanced between treatment and control, which makes the comparison more credible than simply comparing customers selected by a predictive model.

![Conversion rate by group](../figures/conversion_rate_by_group.png)

The treatment group converted at a higher rate than the control group. This is the first evidence that the email created incremental behavior change, not just observed purchase intent.

![Visit rate by group](../figures/visit_rate_by_group.png)

The treatment group also visited at a higher rate. This supports the idea that the campaign increased engagement before purchase.

![Average spend by group](../figures/average_spend_by_group.png)

Average spend was ${treatment["average_spend"]:.2f} for the treatment group and ${control["average_spend"]:.2f} for the control group. The treatment group spent ${spend_difference:.2f} more per customer on average.

## Segment-Level Response

![Channel uplift](../figures/segment_uplift_by_channel.png)

The channel uplift chart ranks acquisition channels by treatment-minus-control conversion lift. This shows that campaign response is not necessarily uniform across customer acquisition paths.

![History segment uplift](../figures/segment_uplift_by_history_segment.png)

The history segment uplift chart compares customer value/history bands. Some customer groups respond more strongly than others, which is exactly the kind of variation an uplift model is designed to learn.

![Channel treatment vs control conversion](../figures/conversion_rate_treatment_vs_control_by_channel.png)

The channel comparison chart shows treatment and control conversion rates side by side. This keeps the analysis anchored to the experimental design instead of only ranking high-converting groups.

![History segment treatment vs control conversion](../figures/conversion_rate_treatment_vs_control_by_history_segment.png)

The history segment comparison chart shows where conversion differences are strongest across customer value bands.

## Spend Concentration

![Spend distribution by group](../figures/spend_distribution_by_group.png)

Positive spend is concentrated among a relatively small group of converted customers. The chart uses a log scale for spend greater than zero so typical purchase values remain readable despite skew.

## Key Business Interpretation

The campaign appears to change customer behavior at the group level when compared with the no-email control group. This matters because campaign targeting should focus on incremental impact, not only on customers who already look likely to buy.

## Why Normal ML Is Not Enough

A normal purchase prediction model would estimate who is likely to convert. That is useful, but it does not tell us whether the email caused the conversion. High-probability customers may have purchased anyway, while some lower-probability customers may be highly persuadable.

## Why Uplift Modeling Is the Next Step

The next step is uplift modeling, which estimates how each customer's outcome changes because of treatment. The segment charts show why this matters: different groups can have different treatment effects, so the best targeting strategy should prioritize incremental lift rather than raw conversion probability.
"""

    output_path = EDA_REPORTS_DIR / "hillstrom_eda_summary.md"
    output_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved EDA summary report to: {output_path}")


def _build_segment_uplift_table(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    """Build one segment-level treatment versus control comparison table."""
    total_counts = (
        df.groupby(segment_column, dropna=False)
        .agg(customer_count=("customer_id", "count"))
        .reset_index()
    )

    grouped = (
        df.groupby([segment_column, "treatment"], dropna=False)
        .agg(
            conversion_rate=("outcome", "mean"),
            average_spend=("spend", "mean"),
        )
        .reset_index()
    )

    treatment = grouped[grouped["treatment"] == 1][
        [segment_column, "conversion_rate", "average_spend"]
    ].rename(
        columns={
            "conversion_rate": "treatment_conversion_rate",
            "average_spend": "treatment_average_spend",
        }
    )
    control = grouped[grouped["treatment"] == 0][
        [segment_column, "conversion_rate", "average_spend"]
    ].rename(
        columns={
            "conversion_rate": "control_conversion_rate",
            "average_spend": "control_average_spend",
        }
    )

    segment_table = total_counts.merge(treatment, on=segment_column, how="left")
    segment_table = segment_table.merge(control, on=segment_column, how="left")
    segment_table["absolute_uplift"] = (
        segment_table["treatment_conversion_rate"]
        - segment_table["control_conversion_rate"]
    )

    output_columns = [
        segment_column,
        "customer_count",
        "treatment_conversion_rate",
        "control_conversion_rate",
        "absolute_uplift",
        "treatment_average_spend",
        "control_average_spend",
    ]

    return segment_table[output_columns].sort_values(
        "absolute_uplift", ascending=False
    )


def _conversion_rate_by_segment(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    """Calculate conversion rates by segment and campaign group for plotting."""
    rates = (
        df.groupby([segment_column, "group"], dropna=False)
        .agg(conversion_rate=("outcome", "mean"))
        .reset_index()
    )
    return rates


def _set_professional_theme(plt, sns) -> None:
    """Apply a clean visual theme for portfolio-ready charts."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "font.family": "DejaVu Sans",
            "xtick.color": MUTED_TEXT_COLOR,
            "ytick.color": MUTED_TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
        }
    )


def _plot_treatment_distribution(plt, campaign_summary: pd.DataFrame) -> None:
    """Plot customer counts to show whether the experiment is balanced."""
    summary = _get_ordered_campaign_summary(campaign_summary)
    values = summary["customers"].tolist()
    colors = [GROUP_COLORS[group] for group in summary["group"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(summary["group"], values, color=colors, width=0.55)

    balance_gap = abs(values[1] - values[0])
    ax.set_ylim(0, max(values) * 1.18)
    _label_vertical_bars(ax, bars, [f"{value:,}" for value in values])
    _add_chart_header(
        fig,
        ax,
        "Experiment Balance: Mens E-Mail vs No E-Mail",
        f"Is the experiment balanced between treatment and control? Group sizes differ by {balance_gap:,} customer(s).",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Number of Customers")
    _style_axis(ax, grid_axis="y")
    _save_figure(plt, fig, "treatment_distribution.png")


def _plot_rate_by_group(
    plt,
    FuncFormatter,
    campaign_summary: pd.DataFrame,
    metric: str,
    title: str,
    subtitle: str,
    ylabel: str,
    filename: str,
) -> None:
    """Plot a treatment-control rate comparison with uplift annotation."""
    summary = _get_ordered_campaign_summary(campaign_summary)
    values = summary[metric].tolist()
    colors = [GROUP_COLORS[group] for group in summary["group"]]
    uplift = values[1] - values[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(summary["group"], values, color=colors, width=0.55)

    ax.set_ylim(0, max(values) * 1.45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1%}"))
    _label_vertical_bars(ax, bars, [f"{value:.2%}" for value in values])
    ax.text(
        0.5,
        max(values) * 1.25,
        f"Absolute uplift: {uplift * 100:+.2f} pp",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=POSITIVE_UPLIFT_COLOR if uplift >= 0 else NEGATIVE_UPLIFT_COLOR,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#F8FAFC",
            "edgecolor": "#CBD5E1",
        },
    )

    _add_chart_header(fig, ax, title, subtitle)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    _style_axis(ax, grid_axis="y")
    _save_figure(plt, fig, filename)


def _plot_average_spend_by_group(
    plt,
    FuncFormatter,
    campaign_summary: pd.DataFrame,
) -> None:
    """Plot average customer spend for treatment and control groups."""
    summary = _get_ordered_campaign_summary(campaign_summary)
    values = summary["average_spend"].tolist()
    colors = [GROUP_COLORS[group] for group in summary["group"]]
    spend_difference = values[1] - values[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(summary["group"], values, color=colors, width=0.55)

    ax.set_ylim(0, max(values) * 1.45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.2f}"))
    _label_vertical_bars(ax, bars, [f"${value:.2f}" for value in values])
    ax.text(
        0.5,
        max(values) * 1.25,
        f"Average spend difference: ${spend_difference:+.2f}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=POSITIVE_UPLIFT_COLOR if spend_difference >= 0 else NEGATIVE_UPLIFT_COLOR,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#F8FAFC",
            "edgecolor": "#CBD5E1",
        },
    )

    _add_chart_header(
        fig,
        ax,
        "Treated Customers Spent More on Average",
        "Did treated customers spend more on average?",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Average Spend per Customer")
    _style_axis(ax, grid_axis="y")
    _save_figure(plt, fig, "average_spend_by_group.png")


def _plot_segment_uplift(
    plt,
    FuncFormatter,
    segment_table: pd.DataFrame,
    segment_column: str,
    title: str,
    subtitle: str,
    filename: str,
) -> None:
    """Plot absolute conversion uplift by segment in percentage points."""
    plot_table = segment_table.sort_values("absolute_uplift", ascending=False)
    labels = plot_table[segment_column].astype(str).tolist()
    uplift_values = (plot_table["absolute_uplift"] * 100).tolist()
    colors = [
        POSITIVE_UPLIFT_COLOR if value >= 0 else NEGATIVE_UPLIFT_COLOR
        for value in uplift_values
    ]

    fig_height = max(5.5, len(labels) * 0.75 + 2.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(labels, uplift_values, color=colors, alpha=0.9)

    x_min = min(min(uplift_values), 0)
    x_max = max(max(uplift_values), 0)
    x_padding = max(abs(x_min), abs(x_max), 0.5) * 0.25
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f} pp"))
    _label_horizontal_bars(ax, bars, [f"{value:+.2f} pp" for value in uplift_values])

    _add_chart_header(fig, ax, title, subtitle)
    ax.set_xlabel("Absolute Conversion Uplift")
    ax.set_ylabel("")
    _style_axis(ax, grid_axis="x")
    _save_figure(plt, fig, filename)


def _plot_segment_grouped_rates(
    plt,
    FuncFormatter,
    plot_data: pd.DataFrame,
    segment_uplift: pd.DataFrame,
    segment_column: str,
    title: str,
    subtitle: str,
    filename: str,
) -> None:
    """Plot treatment and control conversion rates for each segment."""
    segment_order = (
        segment_uplift.sort_values("absolute_uplift", ascending=False)[segment_column]
        .astype(str)
        .tolist()
    )
    rates = _conversion_rate_by_segment(plot_data, segment_column)
    rates[segment_column] = rates[segment_column].astype(str)
    pivot = (
        rates.pivot(index=segment_column, columns="group", values="conversion_rate")
        .reindex(segment_order)
        .fillna(0)
    )

    if segment_column == "history_segment":
        _plot_horizontal_grouped_rates(plt, FuncFormatter, pivot, title, subtitle, filename)
        return

    x_positions = list(range(len(pivot)))
    bar_width = 0.36
    control_values = pivot[CONTROL_LABEL].tolist()
    treatment_values = pivot[TREATMENT_LABEL].tolist()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    control_bars = ax.bar(
        [position - bar_width / 2 for position in x_positions],
        control_values,
        width=bar_width,
        label=CONTROL_LABEL,
        color=GROUP_COLORS[CONTROL_LABEL],
    )
    treatment_bars = ax.bar(
        [position + bar_width / 2 for position in x_positions],
        treatment_values,
        width=bar_width,
        label=TREATMENT_LABEL,
        color=GROUP_COLORS[TREATMENT_LABEL],
    )

    y_max = max(control_values + treatment_values)
    ax.set_ylim(0, y_max * 1.35)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot.index.tolist())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1%}"))
    _label_vertical_bars(ax, control_bars, [f"{value:.2%}" for value in control_values], fontsize=9)
    _label_vertical_bars(ax, treatment_bars, [f"{value:.2%}" for value in treatment_values], fontsize=9)

    _add_chart_header(fig, ax, title, subtitle)
    ax.set_xlabel("")
    ax.set_ylabel("Conversion Rate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    _style_axis(ax, grid_axis="y")
    _save_figure(plt, fig, filename)


def _plot_horizontal_grouped_rates(
    plt,
    FuncFormatter,
    pivot: pd.DataFrame,
    title: str,
    subtitle: str,
    filename: str,
) -> None:
    """Plot horizontal grouped conversion bars for long segment labels."""
    y_positions = list(range(len(pivot)))
    bar_height = 0.36
    control_values = pivot[CONTROL_LABEL].tolist()
    treatment_values = pivot[TREATMENT_LABEL].tolist()

    fig, ax = plt.subplots(figsize=(12, 7.5))
    control_bars = ax.barh(
        [position + bar_height / 2 for position in y_positions],
        control_values,
        height=bar_height,
        label=CONTROL_LABEL,
        color=GROUP_COLORS[CONTROL_LABEL],
    )
    treatment_bars = ax.barh(
        [position - bar_height / 2 for position in y_positions],
        treatment_values,
        height=bar_height,
        label=TREATMENT_LABEL,
        color=GROUP_COLORS[TREATMENT_LABEL],
    )

    x_max = max(control_values + treatment_values)
    ax.set_xlim(0, x_max * 1.35)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pivot.index.tolist())
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1%}"))
    _label_horizontal_bars(ax, control_bars, [f"{value:.2%}" for value in control_values], fontsize=9)
    _label_horizontal_bars(ax, treatment_bars, [f"{value:.2%}" for value in treatment_values], fontsize=9)

    _add_chart_header(fig, ax, title, subtitle)
    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    _style_axis(ax, grid_axis="x")
    _save_figure(plt, fig, filename)


def _plot_spend_distribution(
    plt,
    sns,
    FuncFormatter,
    plot_data: pd.DataFrame,
) -> None:
    """Plot positive spend distribution while handling right skew."""
    positive_spend = plot_data[plot_data["spend"] > 0].copy()

    fig, ax = plt.subplots(figsize=(11, 6))

    if positive_spend.empty:
        print(
            "Skipping spend distribution comparison: no customers have positive spend."
        )
        ax.text(
            0.5,
            0.5,
            "No positive spend values were found.",
            ha="center",
            va="center",
            fontsize=14,
            color=MUTED_TEXT_COLOR,
            transform=ax.transAxes,
        )
    else:
        grouped_spend = {
            group: positive_spend.loc[
                positive_spend["group"] == group, "spend"
            ].dropna()
            for group in GROUP_ORDER
        }
        missing_groups = [
            group for group, group_spend in grouped_spend.items() if group_spend.empty
        ]

        if missing_groups:
            missing_group_names = ", ".join(missing_groups)
            print(
                "Skipping spend distribution comparison: positive spend is missing "
                f"for {missing_group_names}."
            )
            ax.text(
                0.5,
                0.5,
                "Spend distribution skipped because one campaign group has no positive spend.",
                ha="center",
                va="center",
                fontsize=14,
                color=MUTED_TEXT_COLOR,
                transform=ax.transAxes,
            )
        else:
            grouped_values = [grouped_spend[group] for group in GROUP_ORDER]
            boxplot = ax.boxplot(
                grouped_values,
                orientation="horizontal",
                tick_labels=GROUP_ORDER,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "white", "linewidth": 2},
            )

            for patch, group in zip(boxplot["boxes"], GROUP_ORDER):
                patch.set_facecolor(GROUP_COLORS[group])
                patch.set_alpha(0.85)

            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"${value:,.0f}")
            )

            for row_number, group in enumerate(GROUP_ORDER, start=1):
                group_spend = grouped_spend[group]
                median_spend = group_spend.median()
                ax.text(
                    0.98,
                    row_number,
                    f"n={len(group_spend):,} | median=${median_spend:,.0f}",
                    ha="right",
                    va="center",
                    fontsize=10,
                    color=TEXT_COLOR,
                    transform=ax.get_yaxis_transform(),
                )

    _add_chart_header(
        fig,
        ax,
        "Positive Spend Is Concentrated Among Converted Customers",
        "Is spend concentrated among a few converted customers? Showing spend > $0 on a log scale.",
    )
    ax.set_xlabel("Post-Campaign Spend, Customers with Spend > $0")
    ax.set_ylabel("")
    _style_axis(ax, grid_axis="x")
    _save_figure(plt, fig, "spend_distribution_by_group.png")


def _plot_executive_summary(
    plt,
    FancyBboxPatch,
    df: pd.DataFrame,
    campaign_summary: pd.DataFrame,
) -> None:
    """Create a KPI-card summary image for executive storytelling."""
    control = _get_campaign_group(campaign_summary, CONTROL_LABEL)
    treatment = _get_campaign_group(campaign_summary, TREATMENT_LABEL)
    conversion_uplift = treatment["conversion_rate"] - control["conversion_rate"]

    cards = [
        ("Total Customers", f"{len(df):,}", "Mens E-Mail vs No E-Mail sample"),
        (
            "Treatment Conversion",
            f"{treatment['conversion_rate']:.2%}",
            "Mens E-Mail campaign group",
        ),
        (
            "Control Conversion",
            f"{control['conversion_rate']:.2%}",
            "No E-Mail holdout group",
        ),
        (
            "Absolute Uplift",
            f"{conversion_uplift * 100:+.2f} pp",
            "Treatment minus control conversion",
        ),
    ]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_axis_off()

    ax.text(
        0.04,
        0.90,
        "PromoLift AI: Hillstrom Experiment Snapshot",
        fontsize=23,
        fontweight="bold",
        color=TEXT_COLOR,
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.83,
        "A treatment-control view shows why incremental lift matters before building targeting models.",
        fontsize=13,
        color=MUTED_TEXT_COLOR,
        transform=ax.transAxes,
    )

    card_width = 0.215
    card_height = 0.34
    card_y = 0.34
    card_gap = 0.025

    for index, (label, value, note) in enumerate(cards):
        card_x = 0.04 + index * (card_width + card_gap)
        card = FancyBboxPatch(
            (card_x, card_y),
            card_width,
            card_height,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            facecolor="white",
            edgecolor="#CBD5E1",
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(card)
        value_color = POSITIVE_UPLIFT_COLOR if label == "Absolute Uplift" else TEXT_COLOR
        ax.text(
            card_x + 0.025,
            card_y + card_height - 0.09,
            label,
            fontsize=11,
            fontweight="bold",
            color=MUTED_TEXT_COLOR,
            transform=ax.transAxes,
        )
        ax.text(
            card_x + 0.025,
            card_y + 0.16,
            value,
            fontsize=25,
            fontweight="bold",
            color=value_color,
            transform=ax.transAxes,
        )
        ax.text(
            card_x + 0.025,
            card_y + 0.07,
            note,
            fontsize=10,
            color=MUTED_TEXT_COLOR,
            transform=ax.transAxes,
        )

    ax.text(
        0.04,
        0.18,
        "Interpretation: the email improved conversion at the aggregate level, but segment charts show the response is not uniform. This motivates uplift modeling.",
        fontsize=12,
        color=TEXT_COLOR,
        transform=ax.transAxes,
    )

    _save_figure(plt, fig, "eda_executive_summary.png", use_tight_layout=False)


def _get_ordered_campaign_summary(campaign_summary: pd.DataFrame) -> pd.DataFrame:
    """Return campaign summary rows in control-then-treatment order."""
    return campaign_summary.set_index("group").loc[GROUP_ORDER].reset_index()


def _add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    """Add a business-friendly title and subtitle to a chart."""
    fig.suptitle(
        title,
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.set_title(subtitle, loc="left", fontsize=11, color=MUTED_TEXT_COLOR, pad=18)


def _style_axis(ax, grid_axis: str) -> None:
    """Apply shared axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.grid(False, axis="x" if grid_axis == "y" else "y")
    ax.set_axisbelow(True)


def _label_vertical_bars(ax, bars, labels: list[str], fontsize: int = 11) -> None:
    """Add labels above vertical bars."""
    _, y_max = ax.get_ylim()
    offset = y_max * 0.025

    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color=TEXT_COLOR,
        )


def _label_horizontal_bars(ax, bars, labels: list[str], fontsize: int = 11) -> None:
    """Add labels beside horizontal bars."""
    x_min, x_max = ax.get_xlim()
    offset = (x_max - x_min) * 0.012

    for bar, label in zip(bars, labels):
        width = bar.get_width()
        label_x = width + offset if width >= 0 else width - offset
        label_align = "left" if width >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha=label_align,
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=TEXT_COLOR,
        )


def _save_figure(plt, fig, filename: str, use_tight_layout: bool = True) -> None:
    """Save a chart at high resolution and close the figure."""
    output_path = FIGURES_DIR / filename

    if use_tight_layout:
        fig.tight_layout(rect=[0, 0, 1, 0.90])

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved chart to: {output_path}")


def _get_campaign_group(campaign_summary: pd.DataFrame, group_name: str) -> pd.Series:
    """Return one row from the campaign summary by group name."""
    matching_rows = campaign_summary[campaign_summary["group"] == group_name]

    if matching_rows.empty:
        raise ValueError(f"Campaign summary is missing group: {group_name}")

    return matching_rows.iloc[0]


def _format_campaign_summary_table(campaign_summary: pd.DataFrame) -> str:
    """Format the campaign summary as a Markdown table without extra packages."""
    lines = [
        "| Group | Customers | Conversion Rate | Visit Rate | Average Spend | Total Spend |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in campaign_summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.group} | "
            f"{row.customers:,} | "
            f"{row.conversion_rate:.2%} | "
            f"{row.visit_rate:.2%} | "
            f"${row.average_spend:.2f} | "
            f"${row.total_spend:,.2f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    df = load_processed_data()
    summarize_dataset(df)
    campaign_summary = calculate_campaign_lift(df)
    segment_analysis(df)
    create_eda_plots(df, campaign_summary)
    write_eda_summary(df, campaign_summary)
