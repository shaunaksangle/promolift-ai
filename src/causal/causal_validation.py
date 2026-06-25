"""Causal validation and experiment sanity checks.

This module checks whether the Hillstrom treatment-control comparison looks
credible before interpreting uplift scores as campaign impact. The data comes
from a randomized marketing experiment, so the observed treatment-control
difference is meaningful, and these checks make that reasoning explicit.
"""

import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.features.build_features import build_preprocessor, get_feature_columns


PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "hillstrom_mens_email.csv"
CAUSAL_REPORTS_DIR = REPORTS_DIR / "causal"

NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["history_segment", "zip_code", "channel"]

TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#475569"
GRID_COLOR = "#E5E7EB"
CONTROL_COLOR = "#64748B"
TREATMENT_COLOR = "#0F766E"
WARNING_COLOR = "#B45309"
POSITIVE_COLOR = "#15803D"
NEGATIVE_COLOR = "#B91C1C"


def load_data() -> pd.DataFrame:
    """Load the processed Hillstrom dataset.

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


def calculate_naive_ate(df: pd.DataFrame) -> dict[str, float]:
    """Calculate the observed treatment-control conversion lift."""
    treatment_conversion_rate = df.loc[df["treatment"] == 1, "outcome"].mean()
    control_conversion_rate = df.loc[df["treatment"] == 0, "outcome"].mean()
    naive_ate = treatment_conversion_rate - control_conversion_rate
    relative_lift = (
        naive_ate / control_conversion_rate
        if control_conversion_rate != 0
        else 0.0
    )

    return {
        "treatment_conversion_rate": float(treatment_conversion_rate),
        "control_conversion_rate": float(control_conversion_rate),
        "naive_ate": float(naive_ate),
        "naive_ate_percentage_points": float(naive_ate * 100),
        "relative_lift": float(relative_lift),
    }


def calculate_covariate_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate standardized mean differences for pre-campaign features.

    Numeric features are compared directly. Categorical features are one-hot
    encoded and compared as category proportions.
    """
    balance_rows = []

    for feature in NUMERIC_FEATURES:
        treatment_values = df.loc[df["treatment"] == 1, feature]
        control_values = df.loc[df["treatment"] == 0, feature]
        balance_rows.append(
            _make_balance_row(
                feature=feature,
                feature_type="numeric",
                treatment_values=treatment_values,
                control_values=control_values,
            )
        )

    categorical_dummies = pd.get_dummies(
        df[CATEGORICAL_FEATURES],
        prefix=CATEGORICAL_FEATURES,
        dummy_na=False,
    )
    categorical_dummies["treatment"] = df["treatment"].values

    for feature in categorical_dummies.columns:
        if feature == "treatment":
            continue

        treatment_values = categorical_dummies.loc[
            categorical_dummies["treatment"] == 1,
            feature,
        ].astype(float)
        control_values = categorical_dummies.loc[
            categorical_dummies["treatment"] == 0,
            feature,
        ].astype(float)
        balance_rows.append(
            _make_balance_row(
                feature=feature,
                feature_type="categorical",
                treatment_values=treatment_values,
                control_values=control_values,
            )
        )

    balance_df = pd.DataFrame(balance_rows)
    balance_df["abs_smd"] = balance_df["standardized_mean_difference"].abs()
    balance_df["balance_flag"] = balance_df["abs_smd"].map(_balance_flag)

    return balance_df.sort_values("abs_smd", ascending=False).reset_index(drop=True)


def estimate_propensity_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Estimate treatment assignment probability from pre-campaign features."""
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
            "propensity_score": propensity_score,
        }
    )
    propensity_df.attrs["propensity_auc"] = propensity_auc

    return propensity_df, propensity_auc


def run_dowhy_ate_if_available(df: pd.DataFrame) -> dict[str, object]:
    """Try a simple DoWhy causal estimate and refutation if available.

    The project does not depend on DoWhy for the main result. This function is a
    learning-oriented validation layer and is intentionally robust to optional
    dependency or version issues.
    """
    try:
        from dowhy import CausalModel
    except Exception as error:
        return {
            "status": "failed",
            "error": f"DoWhy import failed: {error}",
        }

    try:
        _, categorical_features = get_feature_columns()
        common_cause_df = pd.get_dummies(
            df[NUMERIC_FEATURES + categorical_features],
            columns=categorical_features,
            drop_first=False,
        )
        dowhy_data = pd.concat(
            [
                df[["treatment", "outcome"]].reset_index(drop=True),
                common_cause_df.reset_index(drop=True),
            ],
            axis=1,
        )
        common_causes = common_cause_df.columns.tolist()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)

            causal_model = CausalModel(
                data=dowhy_data,
                treatment="treatment",
                outcome="outcome",
                common_causes=common_causes,
            )
            estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
            estimate = causal_model.estimate_effect(
                estimand,
                method_name="backdoor.propensity_score_matching",
            )

            refutation_results = {}

            for refuter_name in ["random_common_cause", "placebo_treatment_refuter"]:
                try:
                    refutation = causal_model.refute_estimate(
                        estimand,
                        estimate,
                        method_name=refuter_name,
                    )
                    refutation_results[refuter_name] = str(refutation)
                except Exception as refuter_error:
                    refutation_results[refuter_name] = f"failed: {refuter_error}"

        return {
            "status": "success",
            "estimate": float(estimate.value),
            "method": "backdoor.propensity_score_matching",
            "refutations": refutation_results,
        }
    except Exception as error:
        return {
            "status": "failed",
            "error": f"DoWhy estimation failed: {error}",
        }


def create_causal_validation_plots(
    balance_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    ate_results: dict[str, float],
    output_dir,
) -> None:
    """Create portfolio-level causal validation charts."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.ticker import FuncFormatter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Plotting libraries are missing. Install requirements with: "
            "pip install -r requirements.txt"
        ) from error

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_plot_theme(plt, sns)

    _plot_covariate_balance(
        plt,
        balance_df,
        output_dir / "covariate_balance_smd.png",
    )
    _plot_propensity_overlap(
        plt,
        sns,
        propensity_df,
        output_dir / "propensity_score_overlap.png",
    )
    _plot_ate_summary(
        plt,
        FancyBboxPatch,
        ate_results,
        output_dir / "causal_ate_summary.png",
    )
    _plot_treatment_assignment_predictability(
        plt,
        FuncFormatter,
        propensity_df.attrs.get("propensity_auc", 0.0),
        output_dir / "treatment_assignment_predictability.png",
    )


def write_causal_validation_report(
    ate_results: dict[str, float],
    balance_df: pd.DataFrame,
    propensity_auc: float,
    dowhy_results: dict[str, object],
) -> None:
    """Write a Markdown causal validation report."""
    CAUSAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    notable_imbalances = balance_df[balance_df["abs_smd"] >= 0.20]
    moderate_imbalances = balance_df[
        (balance_df["abs_smd"] >= 0.10) & (balance_df["abs_smd"] < 0.20)
    ]
    top_balance_rows = balance_df.head(8)

    if dowhy_results["status"] == "success":
        dowhy_text = (
            "DoWhy ran successfully using a simple backdoor propensity score "
            f"matching estimator. The estimated effect was {dowhy_results['estimate']:.2%}. "
            "Refutation details are saved in `reports/causal/dowhy_results.json`."
        )
    else:
        dowhy_text = (
            "DoWhy was attempted but did not complete in this environment. "
            f"Status detail: `{dowhy_results.get('error', 'unknown error')}`. "
            "This does not block the project because the Hillstrom data is already "
            "a randomized marketing experiment."
        )

    report_text = f"""# Causal Validation Summary

## What Causal Validation Means

Causal validation checks whether the treatment-control comparison is credible before interpreting uplift as campaign impact. For PromoLift AI, this means checking whether customers who received the Mens E-Mail campaign look comparable to customers who received no email before the campaign.

## Why Treatment-Control Balance Matters

If treatment and control customers are balanced on pre-campaign features, then outcome differences are more plausibly caused by the campaign rather than by pre-existing customer differences. Because Hillstrom is a randomized marketing experiment, we expect strong balance.

## Naive Average Treatment Effect

- Treatment conversion rate: {ate_results["treatment_conversion_rate"]:.2%}
- Control conversion rate: {ate_results["control_conversion_rate"]:.2%}
- Naive ATE: {ate_results["naive_ate_percentage_points"]:.2f} percentage points
- Relative lift: {ate_results["relative_lift"]:.2%}

The naive ATE is the observed treatment-control conversion difference. In a randomized experiment, this estimate is already meaningful as a campaign lift measure.

## Covariate Balance Findings

- Features checked: {len(balance_df):,}
- Good balance rows: {(balance_df["balance_flag"] == "good balance").sum():,}
- Moderate imbalance rows: {len(moderate_imbalances):,}
- Notable imbalance rows: {len(notable_imbalances):,}

Top absolute standardized mean differences:

{_format_balance_table(top_balance_rows)}

Standardized mean differences below 0.10 are usually considered good balance. Values above 0.20 would be a stronger warning sign.

## Propensity Score Overlap

The propensity model AUC for predicting treatment assignment from pre-campaign features was {propensity_auc:.3f}. A low or moderate AUC suggests treatment assignment is not easily predictable from customer features. A very high AUC would warn that the treatment group may have been selected differently from control.

## DoWhy Result

{dowhy_text}

## Limitations

These checks support the credibility of treatment-control comparisons, but they do not prove individual counterfactual outcomes. We still never observe the same customer both receiving and not receiving the campaign at the same time.

## Why This Strengthens Uplift Modeling

Uplift scores are used for customer ranking; causal validation supports whether treatment-control comparisons are credible. When balance, overlap, and observed ATE are reasonable, the uplift modeling story becomes more responsible and easier to defend.

## Charts

![Covariate balance](../figures/covariate_balance_smd.png)

![Propensity overlap](../figures/propensity_score_overlap.png)

![ATE summary](../figures/causal_ate_summary.png)

![Treatment assignment predictability](../figures/treatment_assignment_predictability.png)
"""

    output_path = CAUSAL_REPORTS_DIR / "causal_validation_summary.md"
    output_path.write_text(report_text, encoding="utf-8")
    print(f"Saved causal validation summary to: {output_path}")


def run_causal_validation() -> dict[str, object]:
    """Run all causal validation checks and save outputs."""
    CAUSAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    ate_results = calculate_naive_ate(df)
    balance_df = calculate_covariate_balance(df)
    propensity_df, propensity_auc = estimate_propensity_scores(df)
    dowhy_results = run_dowhy_ate_if_available(df)

    (CAUSAL_REPORTS_DIR / "naive_ate.json").write_text(
        json.dumps(ate_results, indent=2),
        encoding="utf-8",
    )
    balance_df.to_csv(CAUSAL_REPORTS_DIR / "covariate_balance.csv", index=False)
    propensity_df.sample(
        n=min(1000, len(propensity_df)),
        random_state=42,
    ).to_csv(CAUSAL_REPORTS_DIR / "propensity_scores_sample.csv", index=False)
    (CAUSAL_REPORTS_DIR / "dowhy_results.json").write_text(
        json.dumps(dowhy_results, indent=2, default=str),
        encoding="utf-8",
    )

    create_causal_validation_plots(
        balance_df,
        propensity_df,
        ate_results,
        FIGURES_DIR,
    )
    write_causal_validation_report(
        ate_results,
        balance_df,
        propensity_auc,
        dowhy_results,
    )

    print("\nCausal validation summary:")
    print(f"Treatment conversion rate: {ate_results['treatment_conversion_rate']:.2%}")
    print(f"Control conversion rate: {ate_results['control_conversion_rate']:.2%}")
    print(
        "Naive ATE: "
        f"{ate_results['naive_ate_percentage_points']:.2f} percentage points"
    )
    print(f"Propensity model AUC: {propensity_auc:.3f}")
    print(
        "Largest absolute SMD: "
        f"{balance_df['abs_smd'].max():.3f} "
        f"({balance_df.iloc[0]['balance_flag']})"
    )
    print(f"DoWhy status: {dowhy_results['status']}")

    return {
        "ate_results": ate_results,
        "propensity_auc": propensity_auc,
        "dowhy_status": dowhy_results["status"],
    }


def _make_balance_row(
    feature: str,
    feature_type: str,
    treatment_values: pd.Series,
    control_values: pd.Series,
) -> dict[str, float | str]:
    """Create one covariate balance row."""
    treatment_mean = treatment_values.mean()
    control_mean = control_values.mean()
    smd = _standardized_mean_difference(treatment_values, control_values)

    return {
        "feature": feature,
        "type": feature_type,
        "treatment_mean": float(treatment_mean),
        "control_mean": float(control_mean),
        "standardized_mean_difference": float(smd),
    }


def _standardized_mean_difference(
    treatment_values: pd.Series,
    control_values: pd.Series,
) -> float:
    """Calculate standardized mean difference between two groups."""
    treatment_variance = treatment_values.var(ddof=1)
    control_variance = control_values.var(ddof=1)
    pooled_std = ((treatment_variance + control_variance) / 2) ** 0.5

    if pooled_std == 0 or pd.isna(pooled_std):
        return 0.0

    return float((treatment_values.mean() - control_values.mean()) / pooled_std)


def _balance_flag(abs_smd: float) -> str:
    """Return a plain-English balance interpretation."""
    if abs_smd < 0.10:
        return "good balance"
    if abs_smd < 0.20:
        return "moderate imbalance"
    return "notable imbalance"


def _set_plot_theme(plt, sns) -> None:
    """Apply a clean professional chart theme."""
    sns.set_theme(style="whitegrid", context="talk")
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


def _plot_covariate_balance(plt, balance_df: pd.DataFrame, output_path: Path) -> None:
    """Plot the largest absolute standardized mean differences."""
    plot_df = balance_df.sort_values("abs_smd", ascending=False).head(15)
    plot_df = plot_df.sort_values("abs_smd", ascending=True)

    colors = [
        POSITIVE_COLOR if value < 0.10 else WARNING_COLOR if value < 0.20 else NEGATIVE_COLOR
        for value in plot_df["abs_smd"]
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(plot_df["feature"], plot_df["abs_smd"], color=colors, alpha=0.9)
    ax.axvline(0.10, color=WARNING_COLOR, linestyle="--", linewidth=2, label="0.10")
    ax.axvline(0.20, color=NEGATIVE_COLOR, linestyle="--", linewidth=2, label="0.20")

    for bar, value in zip(bars, plot_df["abs_smd"]):
        ax.text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    _add_header(
        fig,
        ax,
        "Treatment and Control Customers Are Checked for Pre-Campaign Balance",
        "Are treatment and control customers balanced before the campaign?",
    )
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_ylabel("")
    ax.legend(loc="lower right", frameon=False, title="Balance thresholds")
    _style_axis(ax, grid_axis="x")
    _save_plot(plt, fig, output_path)


def _plot_propensity_overlap(
    plt,
    sns,
    propensity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot propensity score overlap by treatment group."""
    plot_df = propensity_df.copy()
    plot_df["group"] = plot_df["treatment"].map(
        {0: "Control: No E-Mail", 1: "Treatment: Mens E-Mail"}
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=plot_df,
        x="propensity_score",
        hue="group",
        hue_order=["Control: No E-Mail", "Treatment: Mens E-Mail"],
        bins=40,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2.5,
        palette=[CONTROL_COLOR, TREATMENT_COLOR],
        ax=ax,
    )

    _add_header(
        fig,
        ax,
        "Propensity Scores Overlap Across Treatment and Control",
        "Do treatment and control users have comparable probability of treatment assignment?",
    )
    ax.set_xlabel("Estimated Probability of Receiving Treatment")
    ax.set_ylabel("Density")
    _style_axis(ax, grid_axis="y")
    _save_plot(plt, fig, output_path)


def _plot_ate_summary(
    plt,
    FancyBboxPatch,
    ate_results: dict[str, float],
    output_path: Path,
) -> None:
    """Create an executive ATE summary card."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_axis_off()

    ax.text(
        0.04,
        0.86,
        "Observed Causal Lift from Mens E-Mail",
        fontsize=22,
        fontweight="bold",
        color=TEXT_COLOR,
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.78,
        "What is the observed causal lift from the email campaign?",
        fontsize=12,
        color=MUTED_TEXT_COLOR,
        transform=ax.transAxes,
    )

    cards = [
        (
            "Treatment Conversion",
            f"{ate_results['treatment_conversion_rate']:.2%}",
            "Mens E-Mail group",
        ),
        (
            "Control Conversion",
            f"{ate_results['control_conversion_rate']:.2%}",
            "No E-Mail group",
        ),
        (
            "Naive ATE",
            f"{ate_results['naive_ate_percentage_points']:+.2f} pp",
            "Treatment minus control",
        ),
        (
            "Relative Lift",
            f"{ate_results['relative_lift']:.2%}",
            "Lift over control",
        ),
    ]

    card_width = 0.22
    card_height = 0.36
    y = 0.28

    for index, (label, value, note) in enumerate(cards):
        x = 0.04 + index * 0.24
        card = FancyBboxPatch(
            (x, y),
            card_width,
            card_height,
            boxstyle="round,pad=0.016,rounding_size=0.02",
            facecolor="white",
            edgecolor="#CBD5E1",
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(card)
        ax.text(
            x + 0.02,
            y + card_height - 0.09,
            label,
            fontsize=10,
            fontweight="bold",
            color=MUTED_TEXT_COLOR,
            transform=ax.transAxes,
        )
        ax.text(
            x + 0.02,
            y + 0.15,
            value,
            fontsize=22,
            fontweight="bold",
            color=POSITIVE_COLOR if label in {"Naive ATE", "Relative Lift"} else TEXT_COLOR,
            transform=ax.transAxes,
        )
        ax.text(
            x + 0.02,
            y + 0.06,
            note,
            fontsize=9,
            color=MUTED_TEXT_COLOR,
            transform=ax.transAxes,
        )

    _save_plot(plt, fig, output_path, use_tight_layout=False)


def _plot_treatment_assignment_predictability(
    plt,
    FuncFormatter,
    propensity_auc: float,
    output_path: Path,
) -> None:
    """Plot propensity model AUC as a treatment predictability diagnostic."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    color = POSITIVE_COLOR if propensity_auc < 0.65 else WARNING_COLOR
    color = NEGATIVE_COLOR if propensity_auc >= 0.80 else color
    bar = ax.barh(["Treatment predictability"], [propensity_auc], color=color)
    ax.axvline(0.50, color=CONTROL_COLOR, linestyle="--", linewidth=2, label="Random")
    ax.axvline(0.80, color=NEGATIVE_COLOR, linestyle="--", linewidth=2, label="Warning")

    ax.text(
        propensity_auc + 0.01,
        0,
        f"AUC = {propensity_auc:.3f}",
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    _add_header(
        fig,
        ax,
        "Treatment Assignment Is Not Strongly Predicted by Customer Features",
        "Low/moderate AUC supports random-like assignment; very high AUC would suggest selection bias.",
    )
    ax.set_xlim(0.45, 1.00)
    ax.set_xlabel("AUC for Predicting Treatment Assignment")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax.legend(loc="lower right", frameon=False)
    _style_axis(ax, grid_axis="x")
    _save_plot(plt, fig, output_path)


def _add_header(fig, ax, title: str, subtitle: str) -> None:
    """Add a business-readable title and subtitle to a chart."""
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


def _style_axis(ax, grid_axis: str) -> None:
    """Apply shared axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _save_plot(plt, fig, output_path: Path, use_tight_layout: bool = True) -> None:
    """Save a chart at portfolio-ready resolution."""
    if use_tight_layout:
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved chart to: {output_path}")


def _format_balance_table(balance_df: pd.DataFrame) -> str:
    """Format a compact balance table for Markdown."""
    lines = [
        "| Feature | Type | Treatment Mean | Control Mean | Abs SMD | Balance |",
        "|---|---|---:|---:|---:|---|",
    ]

    for row in balance_df.itertuples(index=False):
        lines.append(
            "| "
            f"{row.feature} | "
            f"{row.type} | "
            f"{row.treatment_mean:.3f} | "
            f"{row.control_mean:.3f} | "
            f"{row.abs_smd:.3f} | "
            f"{row.balance_flag} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    run_causal_validation()
