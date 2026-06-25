"""Baseline conversion prediction model.

This module trains normal machine learning models that predict who is likely to
convert. It intentionally does not estimate whether the email caused conversion.
"""

import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.evaluation.metrics import (
    calculate_classification_metrics,
    make_decile_table,
)
from src.features.build_features import (
    build_preprocessor,
    get_feature_columns,
    split_features_target,
)
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_decile_lift,
    plot_precision_recall_curve,
    plot_probability_distribution,
    plot_roc_curve,
)


PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "hillstrom_mens_email.csv"
MODELING_REPORTS_DIR = REPORTS_DIR / "modeling"


def load_data() -> pd.DataFrame:
    """Load the processed Hillstrom dataset for baseline modeling.

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


def train_baseline_model():
    """Train, compare, and evaluate baseline conversion models.

    The selected model is chosen by Average Precision because conversions are
    rare and ranking likely converters is more useful than optimizing accuracy.

    Returns:
        The best fitted Pipeline, metrics dictionary, and model comparison table.
    """
    df = load_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidate_models = _get_candidate_models()
    trained_models = {}
    comparison_rows = []

    for model_name, estimator in candidate_models.items():
        print(f"\nTraining {model_name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = calculate_classification_metrics(y_test, y_proba)

        trained_models[model_name] = pipeline
        comparison_rows.append(
            {
                "model": model_name,
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "average_precision",
        ascending=False,
    )
    best_model_name = comparison_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print("\nBaseline model comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nSelected best model by Average Precision: {best_model_name}")

    metrics = evaluate_and_save_outputs(
        best_model=best_model,
        best_model_name=best_model_name,
        comparison_df=comparison_df,
        X_test=X_test,
        y_test=y_test,
    )

    return best_model, metrics, comparison_df


def evaluate_and_save_outputs(
    best_model,
    best_model_name: str,
    comparison_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | int | str]:
    """Evaluate the selected baseline model and save report outputs."""
    MODELING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (pd.Series(y_proba) >= 0.5).astype(int)

    metrics = calculate_classification_metrics(y_test, y_proba)
    metrics["selected_model"] = best_model_name

    decile_df = make_decile_table(y_test, y_proba)

    metrics_path = MODELING_REPORTS_DIR / "baseline_metrics.json"
    comparison_path = MODELING_REPORTS_DIR / "baseline_model_comparison.csv"
    decile_path = MODELING_REPORTS_DIR / "baseline_decile_table.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    comparison_df.to_csv(comparison_path, index=False)
    decile_df.to_csv(decile_path, index=False)

    print(f"Saved baseline metrics to: {metrics_path}")
    print(f"Saved model comparison to: {comparison_path}")
    print(f"Saved decile table to: {decile_path}")

    plot_roc_curve(y_test, y_proba, FIGURES_DIR / "baseline_roc_curve.png")
    plot_precision_recall_curve(
        y_test,
        y_proba,
        FIGURES_DIR / "baseline_precision_recall_curve.png",
    )
    plot_confusion_matrix(
        y_test,
        y_pred,
        FIGURES_DIR / "baseline_confusion_matrix.png",
    )
    plot_probability_distribution(
        y_test,
        y_proba,
        FIGURES_DIR / "baseline_probability_distribution.png",
    )
    plot_decile_lift(
        decile_df,
        FIGURES_DIR / "baseline_decile_lift.png",
    )

    _write_model_summary(best_model_name, metrics, comparison_df, decile_df)

    return metrics


def _get_candidate_models() -> dict[str, object]:
    """Return simple, robust baseline classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }


def _write_model_summary(
    best_model_name: str,
    metrics: dict[str, float | int | str],
    comparison_df: pd.DataFrame,
    decile_df: pd.DataFrame,
) -> None:
    """Write a portfolio-ready markdown report for the baseline model."""
    numeric_features, categorical_features = get_feature_columns()
    top_decile = decile_df.iloc[0]
    overall_conversion_rate = (
        decile_df["conversions"].sum() / decile_df["customers"].sum()
    )

    summary_text = f"""# Baseline Conversion Model Summary

## What This Model Predicts

The baseline model predicts each customer's probability of conversion using only pre-campaign customer features. It answers: who is likely to convert?

It does not answer whether the Mens E-Mail campaign caused the customer to convert.

## Features Used

Numeric features:

- {", ".join(numeric_features)}

Categorical features:

- {", ".join(categorical_features)}

Treatment assignment was excluded because this baseline is a normal conversion prediction model, not an uplift model. Post-campaign columns such as `visit`, `conversion`, and `spend` were also excluded to avoid leakage.

## Model Comparison

{_format_model_comparison_table(comparison_df)}

The selected model is **{best_model_name}**, chosen by Average Precision.

## Key Metrics

- ROC-AUC: {metrics["roc_auc"]:.3f}
- Average Precision / PR-AUC: {metrics["average_precision"]:.3f}
- Accuracy at threshold 0.50: {metrics["accuracy"]:.3f}
- Precision at threshold 0.50: {metrics["precision"]:.3f}
- Recall at threshold 0.50: {metrics["recall"]:.3f}
- F1 at threshold 0.50: {metrics["f1"]:.3f}

Accuracy is misleading here because conversions are rare. A model can look accurate by mostly predicting non-conversion. Average Precision is more useful because it focuses on how well the model ranks likely converters.

## Decile Lift

The decile table ranks customers by predicted conversion probability. Decile 1 contains the highest-scored customers. In the test set, decile 1 had {int(top_decile["conversions"]):,} conversions among {int(top_decile["customers"]):,} customers, a conversion rate of {top_decile["conversion_rate"]:.2%}. The overall test-set conversion rate was {overall_conversion_rate:.2%}.

## Charts

![ROC curve](../figures/baseline_roc_curve.png)

![Precision-recall curve](../figures/baseline_precision_recall_curve.png)

![Confusion matrix](../figures/baseline_confusion_matrix.png)

![Probability distribution](../figures/baseline_probability_distribution.png)

![Decile lift](../figures/baseline_decile_lift.png)

## Limitation

This model predicts likelihood to convert. It does not estimate incremental uplift caused by the email. A high-scoring customer may have converted without receiving the campaign, while a lower-scoring customer may be highly persuadable.

## Why Uplift Modeling Is Next

The next step is to model the difference between expected outcomes with treatment and without treatment. That uplift-focused view is what turns normal conversion prediction into smarter campaign targeting.
"""

    output_path = MODELING_REPORTS_DIR / "baseline_model_summary.md"
    output_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved baseline model summary to: {output_path}")


def _format_model_comparison_table(comparison_df: pd.DataFrame) -> str:
    """Format the model comparison table as Markdown without extra packages."""
    lines = [
        "| Model | ROC-AUC | Average Precision | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in comparison_df.itertuples(index=False):
        lines.append(
            "| "
            f"{row.model} | "
            f"{row.roc_auc:.3f} | "
            f"{row.average_precision:.3f} | "
            f"{row.accuracy:.3f} | "
            f"{row.precision:.3f} | "
            f"{row.recall:.3f} | "
            f"{row.f1:.3f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    train_baseline_model()
