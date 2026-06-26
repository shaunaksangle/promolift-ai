"""Robustness checks for PromoLift AI uplift findings.

This module checks whether the project's uplift conclusions are stable across
an alternative uplift method, interpretable customer segments, and alternative
treatment definitions. The goal is honest validation, not tuning for a more
flattering result.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.data.load_hillstrom import load_raw_hillstrom
from src.evaluation.uplift_metrics import (
    calculate_ate_confidence_interval,
    calculate_policy_value,
    calculate_qini_coefficient,
    calculate_qini_curve,
    calculate_top_decile_uplift,
    make_uplift_calibration_table,
    make_uplift_decile_table,
)
from src.features.build_features import build_preprocessor, get_feature_columns
from src.visualization.plots import plot_qini_curve, plot_uplift_calibration


PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "hillstrom_mens_email.csv"
UPLIFT_REPORTS_DIR = REPORTS_DIR / "uplift"
ROBUSTNESS_REPORTS_DIR = REPORTS_DIR / "robustness"

MIN_CELL_SIZE_FOR_STABLE_SEGMENT = 300
SMALL_CELL_SIZE_WARNING = 100

TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#475569"
GRID_COLOR = "#E5E7EB"
MODEL_COLOR = "#0F766E"
BASELINE_COLOR = "#64748B"
POSITIVE_COLOR = "#15803D"
NEGATIVE_COLOR = "#B91C1C"
WARNING_COLOR = "#B45309"


def load_processed_data() -> pd.DataFrame:
    """Load the main processed Mens E-Mail versus No E-Mail dataset."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed Hillstrom dataset was not found at "
            f"{PROCESSED_DATA_PATH}. Run this first: "
            "python -m src.data.load_hillstrom"
        )

    return pd.read_csv(PROCESSED_DATA_PATH)


def get_uplift_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return pre-campaign features, treatment, and outcome."""
    numeric_features, categorical_features = get_feature_columns()
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns].copy()
    treatment = df["treatment"].astype(int).copy()
    y = df["outcome"].astype(int).copy()

    return X, treatment, y


def split_data_for_robustness(df: pd.DataFrame):
    """Create the same style of stratified train/test split used in uplift modeling."""
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


def train_x_learner(
    X_train: pd.DataFrame,
    treatment_train: pd.Series,
    y_train: pd.Series,
):
    """Train an X-Learner using only scikit-learn components."""
    treated_mask = treatment_train == 1
    control_mask = treatment_train == 0

    treated_model = _make_outcome_model()
    control_model = _make_outcome_model()
    treated_model.fit(X_train.loc[treated_mask], y_train.loc[treated_mask])
    control_model.fit(X_train.loc[control_mask], y_train.loc[control_mask])

    mu0_for_treated = control_model.predict_proba(X_train.loc[treated_mask])[:, 1]
    mu1_for_control = treated_model.predict_proba(X_train.loc[control_mask])[:, 1]

    treatment_effect_for_treated = (
        y_train.loc[treated_mask].reset_index(drop=True)
        - pd.Series(mu0_for_treated)
    )
    treatment_effect_for_control = (
        pd.Series(mu1_for_control)
        - y_train.loc[control_mask].reset_index(drop=True)
    )

    tau_treated_model = _make_effect_regressor()
    tau_control_model = _make_effect_regressor()
    tau_treated_model.fit(X_train.loc[treated_mask], treatment_effect_for_treated)
    tau_control_model.fit(X_train.loc[control_mask], treatment_effect_for_control)

    propensity_model = _make_propensity_model()
    propensity_model.fit(X_train, treatment_train)

    return {
        "tau_treated_model": tau_treated_model,
        "tau_control_model": tau_control_model,
        "propensity_model": propensity_model,
    }


def predict_x_learner_uplift(x_learner: dict, X_test: pd.DataFrame) -> pd.Series:
    """Predict X-Learner uplift scores for the test set."""
    propensity = pd.Series(
        x_learner["propensity_model"].predict_proba(X_test)[:, 1],
        index=X_test.index,
    )
    tau_treated = pd.Series(
        x_learner["tau_treated_model"].predict(X_test),
        index=X_test.index,
    )
    tau_control = pd.Series(
        x_learner["tau_control_model"].predict(X_test),
        index=X_test.index,
    )

    uplift_score = propensity * tau_control + (1 - propensity) * tau_treated

    return uplift_score.clip(lower=-1, upper=1).reset_index(drop=True)


def evaluate_x_learner(
    y_test: pd.Series,
    treatment_test: pd.Series,
    uplift_score: pd.Series,
) -> dict[str, object]:
    """Create X-Learner robustness tables and summary metrics."""
    decile_table = make_uplift_decile_table(y_test, treatment_test, uplift_score)
    qini_curve = calculate_qini_curve(y_test, treatment_test, uplift_score)
    calibration_table = make_uplift_calibration_table(
        y_test,
        treatment_test,
        uplift_score,
    )
    top_30_policy = calculate_policy_value(
        y_test,
        treatment_test,
        uplift_score,
        top_percent=0.3,
    )

    return {
        "decile_table": decile_table,
        "qini_curve": qini_curve,
        "calibration_table": calibration_table,
        "top_decile_observed_uplift": calculate_top_decile_uplift(decile_table),
        "top_30_policy": top_30_policy,
        "qini_coefficient": calculate_qini_coefficient(qini_curve),
        "average_predicted_uplift": float(pd.Series(uplift_score).mean()),
    }


def build_method_comparison(x_results: dict[str, object]) -> pd.DataFrame:
    """Compare saved T/S-Learner outputs with the X-Learner robustness check."""
    rows = []
    existing_comparison = _load_existing_uplift_comparison()

    if existing_comparison is not None:
        for row in existing_comparison.itertuples(index=False):
            method_row = {
                "method": getattr(row, "model"),
                "top_decile_observed_uplift": getattr(
                    row,
                    "top_decile_observed_uplift",
                    0.0,
                ),
                "top_30_policy_incremental_conversion_rate": getattr(
                    row,
                    "top_30_policy_incremental_conversion_rate",
                    0.0,
                ),
                "top_30_estimated_incremental_conversions": getattr(
                    row,
                    "top_30_estimated_incremental_conversions",
                    0.0,
                ),
                "qini_coefficient": getattr(row, "qini_coefficient", pd.NA),
                "average_predicted_uplift": getattr(
                    row,
                    "average_predicted_uplift",
                    0.0,
                ),
            }
            method_row["interpretation"] = _interpret_method_row(method_row)
            rows.append(method_row)

    top_30_policy = x_results["top_30_policy"]
    x_row = {
        "method": "X-Learner",
        "top_decile_observed_uplift": x_results["top_decile_observed_uplift"],
        "top_30_policy_incremental_conversion_rate": top_30_policy[
            "observed_incremental_conversion_rate"
        ],
        "top_30_estimated_incremental_conversions": top_30_policy[
            "estimated_incremental_conversions"
        ],
        "qini_coefficient": x_results["qini_coefficient"],
        "average_predicted_uplift": x_results["average_predicted_uplift"],
    }
    x_row["interpretation"] = _interpret_method_row(x_row)
    rows.append(x_row)

    return pd.DataFrame(rows)


def calculate_segment_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate observed uplift across interpretable customer segments."""
    analysis_df = df.copy()
    analysis_df["recency_bucket"] = _make_quantile_bucket(
        analysis_df["recency"],
        ["Low recency", "Medium recency", "High recency"],
    )
    analysis_df["history_bucket"] = _make_quantile_bucket(
        analysis_df["history"],
        ["Low history", "Medium history", "High history"],
    )

    segment_variables = [
        "channel",
        "history_segment",
        "newbie",
        "recency_bucket",
        "history_bucket",
    ]
    rows = []

    for segment_variable in segment_variables:
        for segment_value, segment_df in analysis_df.groupby(segment_variable, dropna=False):
            treated = segment_df[segment_df["treatment"] == 1]
            control = segment_df[segment_df["treatment"] == 0]
            treatment_rate = _safe_rate(treated["outcome"].sum(), len(treated))
            control_rate = _safe_rate(control["outcome"].sum(), len(control))
            observed_uplift = treatment_rate - control_rate

            rows.append(
                {
                    "segment_variable": segment_variable,
                    "segment_value": str(segment_value),
                    "customers": int(len(segment_df)),
                    "treated_customers": int(len(treated)),
                    "control_customers": int(len(control)),
                    "treatment_conversion_rate": treatment_rate,
                    "control_conversion_rate": control_rate,
                    "observed_uplift": observed_uplift,
                    "absolute_incremental_conversions": float(
                        observed_uplift * len(segment_df)
                    ),
                    "caution_flag": _segment_caution_flag(len(treated), len(control)),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("observed_uplift", ascending=False)
        .reset_index(drop=True)
    )


def compare_treatment_definitions() -> pd.DataFrame:
    """Compare Mens, Womens, and Any E-Mail treatment definitions."""
    try:
        raw_df = load_raw_hillstrom()
    except Exception as error:
        print(f"Could not load raw Hillstrom data for treatment robustness: {error}")
        return pd.DataFrame()

    raw_df = raw_df.copy()
    raw_df.columns = raw_df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    treatment_definitions = [
        ("Mens E-Mail vs No E-Mail", ["Mens E-Mail"]),
        ("Womens E-Mail vs No E-Mail", ["Womens E-Mail"]),
        ("Any E-Mail vs No E-Mail", ["Mens E-Mail", "Womens E-Mail"]),
    ]
    rows = []

    for treatment_definition, treatment_segments in treatment_definitions:
        comparison_df = raw_df[
            raw_df["segment"].isin(treatment_segments + ["No E-Mail"])
        ].copy()
        comparison_df["treatment"] = comparison_df["segment"].isin(
            treatment_segments
        ).astype(int)
        comparison_df["outcome"] = comparison_df["conversion"].astype(int)

        ate_ci = calculate_ate_confidence_interval(comparison_df)
        relative_lift = (
            ate_ci["ate"] / ate_ci["control_conversion_rate"]
            if ate_ci["control_conversion_rate"] != 0
            else 0.0
        )

        treatment_visit_rate = _safe_rate(
            comparison_df.loc[comparison_df["treatment"] == 1, "visit"].sum(),
            ate_ci["treatment_customers"],
        )
        control_visit_rate = _safe_rate(
            comparison_df.loc[comparison_df["treatment"] == 0, "visit"].sum(),
            ate_ci["control_customers"],
        )

        rows.append(
            {
                "treatment_definition": treatment_definition,
                "customers": int(comparison_df.shape[0]),
                "treated_customers": ate_ci["treatment_customers"],
                "control_customers": ate_ci["control_customers"],
                "treatment_conversion_rate": ate_ci["treatment_conversion_rate"],
                "control_conversion_rate": ate_ci["control_conversion_rate"],
                "observed_ate": ate_ci["ate"],
                "relative_lift": float(relative_lift),
                "standard_error": ate_ci["standard_error"],
                "ci_lower": ate_ci["ci_lower"],
                "ci_upper": ate_ci["ci_upper"],
                "p_value": ate_ci["p_value"],
                "treatment_visit_rate": treatment_visit_rate,
                "control_visit_rate": control_visit_rate,
                "visit_lift": treatment_visit_rate - control_visit_rate,
            }
        )

    return pd.DataFrame(rows)


def create_robustness_plots(
    method_comparison: pd.DataFrame,
    segment_heterogeneity: pd.DataFrame,
    treatment_comparison: pd.DataFrame,
) -> None:
    """Create portfolio-ready robustness plots."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Plotting libraries are missing. Install requirements with: "
            "pip install -r requirements.txt"
        ) from error

    _set_plot_theme(plt)
    _plot_method_comparison(plt, method_comparison)
    _plot_segment_heterogeneity(plt, FuncFormatter, segment_heterogeneity)
    _plot_treatment_definition_comparison(plt, FuncFormatter, treatment_comparison)


def write_robustness_summary(
    method_comparison: pd.DataFrame,
    segment_heterogeneity: pd.DataFrame,
    treatment_comparison: pd.DataFrame,
) -> None:
    """Write a concise robustness report."""
    best_method = method_comparison.sort_values(
        "top_30_estimated_incremental_conversions",
        ascending=False,
    ).iloc[0]
    x_row = method_comparison[method_comparison["method"] == "X-Learner"].iloc[0]

    strongest_segment = segment_heterogeneity.sort_values(
        "observed_uplift",
        ascending=False,
    ).iloc[0]
    weakest_segment = segment_heterogeneity.sort_values("observed_uplift").iloc[0]

    treatment_text = _format_treatment_definition_findings(treatment_comparison)

    report = f"""# Robustness Checks Summary

## Why Robustness Checks Were Added

This robustness step is not intended to search for a flattering result. It checks whether the project's conclusion is stable across alternative uplift methods, interpretable segments, and treatment definitions.

## X-Learner Robustness Result

The X-Learner is an exploratory robustness method built with scikit-learn. It uses separate treated/control outcome models, imputed treatment effects, treatment-effect regressors, and propensity-weighted combination.

- X-Learner top-decile observed uplift: {x_row["top_decile_observed_uplift"]:.2%}
- X-Learner top-30% estimated incremental conversions: {x_row["top_30_estimated_incremental_conversions"]:.1f}
- X-Learner Qini coefficient: {_format_optional_number(x_row["qini_coefficient"])}
- X-Learner interpretation: {x_row["interpretation"]}

## Method Agreement

The strongest method by top-30% estimated incremental conversions is `{best_method["method"]}`. Agreement across methods should be interpreted cautiously because conversions are rare and uplift ranking can be noisy.

See:

- `reports/robustness/uplift_method_robustness_comparison.csv`
- `reports/figures/uplift_method_robustness_comparison.png`

## Segment Heterogeneity Findings

Direct segment checks look for observed treatment-effect differences in business-readable groups rather than relying only on model ranking.

- Strongest observed segment uplift: `{strongest_segment["segment_variable"]} = {strongest_segment["segment_value"]}` at {strongest_segment["observed_uplift"]:.2%}
- Weakest observed segment uplift: `{weakest_segment["segment_variable"]} = {weakest_segment["segment_value"]}` at {weakest_segment["observed_uplift"]:.2%}

These segment-level differences are exploratory. Small segments can be noisy, so caution flags are included in the CSV output.

## Treatment Definition Findings

{treatment_text}

The main project keeps Mens E-Mail vs No E-Mail as the clean primary setup. Womens E-Mail and Any E-Mail comparisons are exploratory robustness checks.

## Final Honest Conclusion

If fine-grained targeting does not strongly beat random targeting, that is still a valid and useful business finding. It suggests that the campaign may create broad lift, while individual-level targeting needs stronger signal or richer features.

The robustness checks should be used to pressure-test the project narrative, not to replace the main uplift model or force a better-looking result.
"""

    output_path = ROBUSTNESS_REPORTS_DIR / "robustness_checks_summary.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved robustness summary to: {output_path}")


def run_robustness_checks() -> None:
    """Run all robustness checks and save outputs."""
    ROBUSTNESS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed_data()
    X_train, X_test, treatment_train, treatment_test, y_train, y_test = (
        split_data_for_robustness(df)
    )

    print("\nTraining exploratory X-Learner robustness model...")
    x_learner = train_x_learner(X_train, treatment_train, y_train)
    x_uplift_score = predict_x_learner_uplift(x_learner, X_test)
    x_results = evaluate_x_learner(y_test, treatment_test, x_uplift_score)

    x_results["decile_table"].to_csv(
        ROBUSTNESS_REPORTS_DIR / "x_learner_decile_table.csv",
        index=False,
    )
    x_results["qini_curve"].to_csv(
        ROBUSTNESS_REPORTS_DIR / "x_learner_qini_curve.csv",
        index=False,
    )
    x_results["calibration_table"].to_csv(
        ROBUSTNESS_REPORTS_DIR / "x_learner_calibration_table.csv",
        index=False,
    )
    plot_qini_curve(
        x_results["qini_curve"],
        FIGURES_DIR / "x_learner_qini_curve.png",
    )
    plot_uplift_calibration(
        x_results["calibration_table"],
        FIGURES_DIR / "x_learner_calibration_by_decile.png",
    )

    method_comparison = build_method_comparison(x_results)
    method_comparison.to_csv(
        ROBUSTNESS_REPORTS_DIR / "uplift_method_robustness_comparison.csv",
        index=False,
    )

    segment_heterogeneity = calculate_segment_heterogeneity(df)
    segment_heterogeneity.to_csv(
        ROBUSTNESS_REPORTS_DIR / "segment_heterogeneity_checks.csv",
        index=False,
    )

    treatment_comparison = compare_treatment_definitions()
    treatment_comparison.to_csv(
        ROBUSTNESS_REPORTS_DIR / "treatment_definition_comparison.csv",
        index=False,
    )

    create_robustness_plots(
        method_comparison,
        segment_heterogeneity,
        treatment_comparison,
    )
    write_robustness_summary(
        method_comparison,
        segment_heterogeneity,
        treatment_comparison,
    )

    print("\nRobustness checks complete.")
    print(f"Saved robustness reports to: {ROBUSTNESS_REPORTS_DIR}")
    print(f"Saved robustness figures to: {FIGURES_DIR}")


def _make_outcome_model() -> Pipeline:
    """Create a simple outcome classifier for X-Learner components."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def _make_propensity_model() -> Pipeline:
    """Create a simple propensity model for X-Learner weighting."""
    return Pipeline(
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


def _make_effect_regressor() -> Pipeline:
    """Create a stable treatment-effect regressor."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=50,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _load_existing_uplift_comparison() -> pd.DataFrame | None:
    """Load the existing T/S-Learner comparison if available."""
    comparison_path = UPLIFT_REPORTS_DIR / "uplift_model_comparison.csv"

    if not comparison_path.exists():
        print("Existing uplift comparison not found; comparing X-Learner only.")
        return None

    return pd.read_csv(comparison_path)


def _interpret_method_row(row: dict) -> str:
    """Return an honest plain-English interpretation for one method row."""
    qini = row.get("qini_coefficient", pd.NA)
    top_30_conversions = row.get("top_30_estimated_incremental_conversions", 0.0)
    top_decile_uplift = row.get("top_decile_observed_uplift", 0.0)
    average_predicted_uplift = row.get("average_predicted_uplift", 0.0)

    if pd.notna(qini) and qini > 0 and top_30_conversions > 10:
        return "stronger ranking signal"
    if abs(average_predicted_uplift) > 0.10 and abs(top_decile_uplift) < 0.02:
        return "predicted uplift magnitude appears poorly calibrated"
    if top_30_conversions <= 0 or (pd.notna(qini) and qini <= 0):
        return "weak or noisy ranking signal"

    return "supports muted heterogeneity conclusion"


def _make_quantile_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    """Create robust quantile buckets using ranks to avoid duplicate-bin issues."""
    return pd.qcut(
        series.rank(method="first"),
        q=len(labels),
        labels=labels,
    )


def _segment_caution_flag(treated_customers: int, control_customers: int) -> str:
    """Return a sample-size caution flag for a segment."""
    smallest_cell = min(treated_customers, control_customers)

    if smallest_cell < SMALL_CELL_SIZE_WARNING:
        return "small treated/control cell"
    if smallest_cell < MIN_CELL_SIZE_FOR_STABLE_SEGMENT:
        return "moderate sample size"
    return "ok"


def _safe_rate(numerator: float, denominator: float) -> float:
    """Safely calculate a rate."""
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def _set_plot_theme(plt) -> None:
    """Apply consistent plot styling for robustness charts."""
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


def _plot_method_comparison(plt, method_comparison: pd.DataFrame) -> None:
    """Plot top-30% incremental conversions by uplift method."""
    plot_df = method_comparison.sort_values(
        "top_30_estimated_incremental_conversions",
        ascending=True,
    )
    colors = [
        POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR
        for value in plot_df["top_30_estimated_incremental_conversions"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        plot_df["method"],
        plot_df["top_30_estimated_incremental_conversions"],
        color=colors,
    )
    ax.axvline(0, color=BASELINE_COLOR, linewidth=1.8)

    for bar, value in zip(bars, plot_df["top_30_estimated_incremental_conversions"]):
        label_x = value + 0.4 if value >= 0 else value - 0.4
        ha = "left" if value >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    _add_header(
        fig,
        ax,
        "Uplift Method Robustness: Top-30% Policy Value",
        "Alternative methods should support the story, not be tuned to force a better result.",
    )
    ax.set_xlabel("Estimated incremental conversions")
    ax.set_ylabel("")
    _style_axis(ax, grid_axis="x")
    _save_plot(plt, fig, FIGURES_DIR / "uplift_method_robustness_comparison.png")


def _plot_segment_heterogeneity(
    plt,
    FuncFormatter,
    segment_heterogeneity: pd.DataFrame,
) -> None:
    """Plot observed uplift across interpretable customer segments."""
    plot_df = segment_heterogeneity.copy()
    plot_df["label"] = (
        plot_df["segment_variable"].astype(str)
        + " = "
        + plot_df["segment_value"].astype(str)
    )
    plot_df = plot_df.sort_values("observed_uplift", ascending=True)
    colors = [
        POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR
        for value in plot_df["observed_uplift"]
    ]

    fig_height = max(7, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(plot_df["label"], plot_df["observed_uplift"], color=colors)
    ax.axvline(0, color=BASELINE_COLOR, linewidth=1.8)

    for bar, value, caution in zip(
        bars,
        plot_df["observed_uplift"],
        plot_df["caution_flag"],
    ):
        label_x = value + 0.0004 if value >= 0 else value - 0.0004
        ha = "left" if value >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value * 100:+.2f} pp ({caution})",
            va="center",
            ha=ha,
            fontsize=8,
            color=TEXT_COLOR,
        )

    _add_header(
        fig,
        ax,
        "Observed Segment Heterogeneity Checks",
        "Direct segment comparisons are exploratory and can be noisy in small cells.",
    )
    ax.set_xlabel("Observed conversion uplift")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.1f} pp"))
    _style_axis(ax, grid_axis="x")
    _save_plot(plt, fig, FIGURES_DIR / "segment_heterogeneity_checks.png")


def _plot_treatment_definition_comparison(
    plt,
    FuncFormatter,
    treatment_comparison: pd.DataFrame,
) -> None:
    """Plot ATE with confidence intervals across treatment definitions."""
    if treatment_comparison.empty:
        _plot_missing_treatment_definition_note(plt)
        return

    plot_df = treatment_comparison.sort_values("observed_ate", ascending=True)
    ate_pp = plot_df["observed_ate"] * 100
    lower_error = ate_pp - plot_df["ci_lower"] * 100
    upper_error = plot_df["ci_upper"] * 100 - ate_pp

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.errorbar(
        x=ate_pp,
        y=plot_df["treatment_definition"],
        xerr=[lower_error, upper_error],
        fmt="o",
        color=MODEL_COLOR,
        ecolor=MODEL_COLOR,
        elinewidth=2.5,
        capsize=6,
        markersize=8,
    )
    ax.axvline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.8)

    _add_header(
        fig,
        ax,
        "Treatment Definition Robustness",
        "Alternative email arms are exploratory checks; Mens E-Mail remains the primary setup.",
    )
    ax.set_xlabel("Observed conversion lift")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f} pp"))
    _style_axis(ax, grid_axis="x")
    _save_plot(plt, fig, FIGURES_DIR / "treatment_definition_comparison.png")


def _plot_missing_treatment_definition_note(plt) -> None:
    """Create a simple figure when raw data cannot be loaded."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "Treatment definition comparison unavailable",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        0.5,
        0.38,
        "Raw Hillstrom data was not available in this environment.",
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED_TEXT_COLOR,
    )
    _save_plot(plt, fig, FIGURES_DIR / "treatment_definition_comparison.png")


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
    """Apply light chart styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _save_plot(plt, fig, output_path: Path) -> None:
    """Save a figure at portfolio-ready resolution."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved chart to: {output_path}")


def _format_treatment_definition_findings(treatment_comparison: pd.DataFrame) -> str:
    """Format treatment definition findings for the markdown report."""
    if treatment_comparison.empty:
        return (
            "Treatment definition checks could not be completed because the raw "
            "Hillstrom data was not available."
        )

    lines = []
    for row in treatment_comparison.itertuples(index=False):
        lines.append(
            "- "
            f"{row.treatment_definition}: observed ATE {row.observed_ate:.2%}, "
            f"conversion p-value {row.p_value:.4f}, secondary visit lift {row.visit_lift:.2%}"
        )

    return "\n".join(lines)


def _format_optional_number(value) -> str:
    """Format a number that may be missing."""
    if pd.isna(value):
        return "not available"

    return f"{float(value):.3f}"


if __name__ == "__main__":
    run_robustness_checks()
