"""Portfolio-level plotting utilities.

This module saves charts for the normal baseline conversion model. The charts
are designed to explain model behavior and the limits of normal prediction.
"""

from pathlib import Path

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


def _style_axis(ax) -> None:
    """Apply light grid and remove unnecessary borders."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _save_plot(plt, fig, output_path) -> None:
    """Save the active figure at portfolio-ready resolution."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved chart to: {output_path}")
