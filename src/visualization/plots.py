"""Portfolio-level plotting utilities.

This module saves charts for the normal baseline conversion model. The charts
are designed to explain model behavior and the limits of normal prediction.
"""

from pathlib import Path
from textwrap import wrap

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#475569"
GRID_COLOR = "#E5E7EB"
BASELINE_COLOR = "#64748B"
MODEL_COLOR = "#0F766E"
POSITIVE_COLOR = "#15803D"
WARNING_COLOR = "#B45309"


def save_figure(fig, output_path) -> None:
    """Save a Matplotlib figure at high resolution and close it."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def apply_readable_style(ax, title=None, xlabel=None, ylabel=None) -> None:
    """Apply readable title, axis labels, ticks, and light grid styling."""
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=16)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

    ax.tick_params(axis="both", labelsize=10, pad=8)
    ax.grid(True, alpha=0.30)
    ax.set_axisbelow(True)


def wrap_labels(labels, width=18) -> list[str]:
    """Wrap long category labels so they do not collide with chart marks."""
    wrapped_labels = []
    for label in labels:
        label_text = str(label)
        wrapped = wrap(
            label_text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_labels.append("\n".join(wrapped) if wrapped else label_text)

    return wrapped_labels


def plot_roc_curve(y_true, y_proba, output_path) -> None:
    """Save a ROC curve for the baseline conversion model."""
    plt, _ = _load_plotting_tools()
    _set_plot_theme(plt)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(fpr, tpr, color=MODEL_COLOR, linewidth=3, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=BASELINE_COLOR, linestyle="--", linewidth=2)

    _add_header(
        fig,
        ax,
        "Baseline Model Separates Converters Better Than Random",
        "ROC-AUC shows broad ranking quality, but it can look optimistic when conversions are rare.",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False)
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_precision_recall_curve(y_true, y_proba, output_path) -> None:
    """Save a precision-recall curve for the baseline conversion model."""
    plt, _ = _load_plotting_tools()
    _set_plot_theme(plt)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    average_precision = average_precision_score(y_true, y_proba)
    base_rate = pd.Series(y_true).mean()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        recall,
        precision,
        color=MODEL_COLOR,
        linewidth=3,
        label=f"Average Precision = {average_precision:.3f}",
    )
    ax.axhline(
        base_rate,
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Base conversion rate = {base_rate:.3f}",
    )

    _add_header(
        fig,
        ax,
        "Precision-Recall Highlights Rare Conversion Performance",
        "PR-AUC is more informative than accuracy when only a small share of customers convert.",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", frameon=False)
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_confusion_matrix(y_true, y_pred, output_path) -> None:
    """Save an annotated confusion matrix."""
    plt, sns = _load_plotting_tools()
    _set_plot_theme(plt)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["Did Not Convert", "Converted"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="BuGn",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        ax=ax,
    )

    _add_header(
        fig,
        ax,
        "Confusion Matrix at a 0.50 Probability Threshold",
        "The threshold view is useful operationally, but rare conversion makes ranking metrics more important.",
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    _save_plot(plt, fig, output_path)


def plot_probability_distribution(y_true, y_proba, output_path) -> None:
    """Save predicted probability distributions by actual outcome."""
    plt, sns = _load_plotting_tools()
    _set_plot_theme(plt)

    plot_df = pd.DataFrame(
        {
            "actual_outcome": pd.Series(y_true)
            .map({0: "Did Not Convert", 1: "Converted"})
            .reset_index(drop=True),
            "predicted_probability": pd.Series(y_proba).reset_index(drop=True),
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=plot_df,
        x="predicted_probability",
        hue="actual_outcome",
        hue_order=["Did Not Convert", "Converted"],
        bins=40,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2.5,
        palette=["#64748B", "#0F766E"],
        ax=ax,
    )

    _add_header(
        fig,
        ax,
        "Predicted Conversion Scores Are Rankings, Not Treatment Effects",
        "The baseline estimates who is likely to convert, not who converts because of the email.",
    )
    ax.set_xlabel("Predicted Conversion Probability")
    ax.set_ylabel("Density")
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_decile_lift(decile_df: pd.DataFrame, output_path) -> None:
    """Save a decile lift chart for ranked conversion probabilities."""
    plt, _ = _load_plotting_tools()
    _set_plot_theme(plt)

    overall_rate = (
        decile_df["conversions"].sum() / decile_df["customers"].sum()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        decile_df["decile"].astype(str),
        decile_df["conversion_rate"],
        color=MODEL_COLOR,
        alpha=0.9,
    )
    ax.axhline(
        overall_rate,
        color=WARNING_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Overall conversion rate = {overall_rate:.2%}",
    )

    for bar, value in zip(bars, decile_df["conversion_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    _add_header(
        fig,
        ax,
        "Top Model Deciles Capture Higher Conversion Rates",
        "Decile 1 contains customers with the highest predicted conversion probability.",
    )
    ax.set_xlabel("Predicted Probability Decile, 1 = Highest")
    ax.set_ylabel("Observed Conversion Rate")
    ax.legend(loc="upper right", frameon=False)
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_uplift_decile_bar(decile_df: pd.DataFrame, output_path) -> None:
    """Save observed uplift by predicted-uplift decile."""
    plt, _ = _load_plotting_tools()
    from matplotlib.ticker import FuncFormatter

    _set_plot_theme(plt)

    plot_df = decile_df.copy()
    plot_df["observed_uplift_pp"] = plot_df["observed_uplift"] * 100
    colors = [
        POSITIVE_COLOR if value >= 0 else "#B91C1C"
        for value in plot_df["observed_uplift_pp"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        plot_df["decile"].astype(str),
        plot_df["observed_uplift_pp"],
        color=colors,
        alpha=0.9,
    )
    ax.axhline(0, color=BASELINE_COLOR, linewidth=2)

    for bar, value in zip(bars, plot_df["observed_uplift_pp"]):
        va = "bottom" if value >= 0 else "top"
        y_offset = 0.05 if value >= 0 else -0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_offset,
            f"{value:+.2f} pp",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    _add_header(
        fig,
        ax,
        "Observed Uplift by Predicted-Uplift Decile",
        "Decile 1 contains the customers predicted to be most persuadable by the email.",
    )
    ax.set_xlabel("Predicted Uplift Decile, 1 = Highest")
    ax.set_ylabel("Observed Conversion Uplift")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f} pp"))
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_cumulative_uplift_curve(decile_df: pd.DataFrame, output_path) -> None:
    """Save a cumulative observed uplift curve across deciles."""
    plt, _ = _load_plotting_tools()
    from matplotlib.ticker import FuncFormatter

    _set_plot_theme(plt)

    plot_df = decile_df.copy()
    plot_df["cumulative_observed_uplift_pp"] = (
        plot_df["cumulative_observed_uplift"] * 100
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        plot_df["decile"],
        plot_df["cumulative_observed_uplift_pp"],
        color=MODEL_COLOR,
        linewidth=3,
        marker="o",
        markersize=7,
    )
    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=2)

    _add_header(
        fig,
        ax,
        "Cumulative Uplift as More Customers Are Targeted",
        "The curve shows observed treatment-control lift after adding each uplift-ranked decile.",
    )
    ax.set_xlabel("Included Deciles, Ordered by Predicted Uplift")
    ax.set_ylabel("Cumulative Observed Uplift")
    ax.set_xticks(plot_df["decile"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f} pp"))
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_predicted_uplift_distribution(uplift_scores, output_path) -> None:
    """Save the distribution of predicted individual uplift scores."""
    plt, sns = _load_plotting_tools()
    from matplotlib.ticker import FuncFormatter

    _set_plot_theme(plt)

    uplift_scores_pp = pd.Series(uplift_scores, name="uplift_score") * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        uplift_scores_pp,
        bins=40,
        color=MODEL_COLOR,
        alpha=0.75,
        edgecolor="white",
        ax=ax,
    )
    ax.axvline(
        0,
        color=WARNING_COLOR,
        linestyle="--",
        linewidth=2.5,
        label="Zero predicted uplift",
    )

    _add_header(
        fig,
        ax,
        "Predicted Uplift Scores Separate Positive and Negative Impact",
        "Scores above zero indicate customers predicted to convert more often if emailed.",
    )
    ax.set_xlabel("Predicted Uplift Score, Percentage Points")
    ax.set_ylabel("Customers")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f} pp"))
    ax.legend(loc="upper right", frameon=False)
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_policy_comparison(policy_df: pd.DataFrame, output_path) -> None:
    """Save a policy comparison chart for alternative targeting rules."""
    plt, _ = _load_plotting_tools()
    _set_plot_theme(plt)

    plot_df = policy_df.copy()
    if "policy" not in plot_df.columns:
        plot_df["policy"] = plot_df["target_percent"].map(
            lambda value: f"Top {value:.0%}"
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        plot_df["policy"],
        plot_df["estimated_incremental_conversions"],
        color=MODEL_COLOR,
        alpha=0.9,
    )
    ax.axhline(0, color=BASELINE_COLOR, linewidth=2)

    for bar, incremental_conversions, incremental_rate in zip(
        bars,
        plot_df["estimated_incremental_conversions"],
        plot_df["observed_incremental_conversion_rate"],
    ):
        va = "bottom" if incremental_conversions >= 0 else "top"
        max_value = max(abs(plot_df["estimated_incremental_conversions"]))
        y_offset = max(max_value * 0.03, 0.5)
        y_offset = y_offset if incremental_conversions >= 0 else -y_offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            incremental_conversions + y_offset,
            f"{incremental_conversions:,.1f}\n({incremental_rate:+.2%})",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    _add_header(
        fig,
        ax,
        "Targeting by Uplift Changes Expected Incremental Conversions",
        "Policy value compares emailing everyone with targeting only high-uplift customers.",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Estimated Incremental Conversions")
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_qini_curve(qini_df: pd.DataFrame, output_path) -> None:
    """Save a Qini curve comparing uplift ranking against random targeting."""
    plt, _ = _load_plotting_tools()
    _set_plot_theme(plt)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(
        qini_df["targeted_customers"],
        qini_df["cumulative_incremental_conversions"],
        color=MODEL_COLOR,
        linewidth=3,
        label="Uplift-ranked targeting",
    )
    ax.plot(
        qini_df["targeted_customers"],
        qini_df["random_incremental_conversions"],
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=2.5,
        label="Random targeting baseline",
    )
    ax.axhline(0, color=GRID_COLOR, linewidth=1.5)

    average_gain = qini_df["qini_gain"].mean()
    if average_gain >= 0:
        annotation = "Model ranking beats random targeting over the full ranked list."
        annotation_color = POSITIVE_COLOR
    else:
        annotation = "Model ranking does not beat random targeting over the full ranked list."
        annotation_color = WARNING_COLOR

    ax.text(
        0.03,
        0.94,
        annotation,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        color=annotation_color,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
        },
    )

    _add_header(
        fig,
        ax,
        "Qini Curve: Does Uplift Targeting Beat Random Targeting?",
        "The model is useful if its cumulative incremental conversions stay above the random baseline.",
    )
    ax.set_xlabel("Customers targeted")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.legend(loc="lower right", frameon=False)
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_uplift_calibration(calibration_df: pd.DataFrame, output_path) -> None:
    """Save a calibration chart comparing predicted and observed uplift."""
    plt, _ = _load_plotting_tools()
    from matplotlib.ticker import FuncFormatter

    _set_plot_theme(plt)

    plot_df = calibration_df.copy()
    plot_df["average_predicted_uplift_pp"] = (
        plot_df["average_predicted_uplift"] * 100
    )
    plot_df["observed_uplift_pp"] = plot_df["observed_uplift"] * 100

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(
        plot_df["decile"],
        plot_df["average_predicted_uplift_pp"],
        color=MODEL_COLOR,
        linewidth=3,
        marker="o",
        label="Average predicted uplift",
    )
    ax.plot(
        plot_df["decile"],
        plot_df["observed_uplift_pp"],
        color=WARNING_COLOR,
        linewidth=3,
        marker="o",
        label="Observed uplift",
    )
    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.8)

    for row in plot_df.itertuples(index=False):
        ax.text(
            row.decile,
            min(row.average_predicted_uplift_pp, row.observed_uplift_pp) - 0.25,
            f"n={int(row.customers):,}",
            ha="center",
            va="top",
            fontsize=8,
            color=MUTED_TEXT_COLOR,
        )

    _add_header(
        fig,
        ax,
        "Uplift Calibration by Decile",
        "Predicted uplift is evaluated as a ranking signal, not perfect individual causal truth.",
    )
    ax.set_xlabel("Predicted Uplift Decile, 1 = Highest")
    ax.set_ylabel("Uplift")
    ax.set_xticks(plot_df["decile"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f} pp"))
    ax.legend(loc="upper right", frameon=False)
    _style_axis(ax)
    _save_plot(plt, fig, output_path)


def plot_ate_confidence_interval(ate_ci: dict, output_path) -> None:
    """Save observed ATE with a confidence interval."""
    plt, _ = _load_plotting_tools()
    from matplotlib.ticker import FuncFormatter

    _set_plot_theme(plt)

    ate = ate_ci["ate"] * 100
    ci_lower = ate_ci["ci_lower"] * 100
    ci_upper = ate_ci["ci_upper"] * 100
    lower_error = ate - ci_lower
    upper_error = ci_upper - ate

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(
        x=[ate],
        y=["Observed ATE"],
        xerr=[[lower_error], [upper_error]],
        fmt="o",
        color=MODEL_COLOR,
        ecolor=MODEL_COLOR,
        elinewidth=3,
        capsize=8,
        markersize=10,
    )
    ax.axvline(0, color=BASELINE_COLOR, linestyle="--", linewidth=2, label="Zero effect")
    ax.text(
        ate,
        0.12,
        f"{ate:+.2f} pp\np={ate_ci['p_value']:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    _add_header(
        fig,
        ax,
        "Observed Campaign Lift with Confidence Interval",
        f"{ate_ci['confidence_level']:.0%} confidence interval for treatment minus control conversion rate.",
    )
    ax.set_xlabel("Conversion lift")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f} pp"))
    ax.legend(loc="lower right", frameon=False)
    _style_axis(ax, grid_axis="x")
    _save_plot(plt, fig, output_path)


def _load_plotting_tools():
    """Import plotting libraries only when a chart is created."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Plotting libraries are missing. Install requirements with: "
            "pip install -r requirements.txt"
        ) from error

    return plt, sns


def _set_plot_theme(plt) -> None:
    """Apply a consistent professional chart style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": MUTED_TEXT_COLOR,
            "ytick.color": MUTED_TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        }
    )


def _add_header(fig, ax, title: str, subtitle: str) -> None:
    """Add a business-readable title and subtitle."""
    fig.suptitle(
        title,
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.set_title(subtitle, loc="left", fontsize=10, color=MUTED_TEXT_COLOR, pad=16)


def _style_axis(ax, grid_axis: str = "y") -> None:
    """Apply light grid and remove unnecessary borders."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _save_plot(plt, fig, output_path) -> None:
    """Save the active figure at portfolio-ready resolution."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved chart to: {output_path}")
