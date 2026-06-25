"""Reusable model evaluation metrics.

This module contains classification metrics for the normal conversion
prediction baseline. Uplift-specific metrics will be added in a later step.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate standard classification metrics for conversion prediction.

    Average Precision is especially important because conversion is rare in the
    Hillstrom dataset.

    Args:
        y_true: True binary outcomes.
        y_proba: Predicted conversion probabilities.
        threshold: Probability cutoff used to convert probabilities to classes.

    Returns:
        Dictionary of classification metrics and confusion matrix values.
    """
    y_pred = (pd.Series(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_decile_table(y_true, y_proba) -> pd.DataFrame:
    """Create a decile table ranked by predicted conversion probability.

    Decile 1 is the highest predicted probability group. This table helps show
    whether the baseline model concentrates actual conversions near the top of
    its ranking.

    Args:
        y_true: True binary outcomes.
        y_proba: Predicted conversion probabilities.

    Returns:
        DataFrame with decile-level and cumulative conversion metrics.
    """
    decile_df = pd.DataFrame(
        {
            "outcome": pd.Series(y_true).astype(int).reset_index(drop=True),
            "predicted_probability": pd.Series(y_proba).reset_index(drop=True),
        }
    )
    decile_df["rank"] = decile_df["predicted_probability"].rank(
        method="first",
        ascending=False,
    )
    decile_df["decile"] = pd.qcut(
        decile_df["rank"],
        q=10,
        labels=False,
    ).astype(int) + 1

    output = (
        decile_df.groupby("decile")
        .agg(
            customers=("outcome", "count"),
            conversions=("outcome", "sum"),
        )
        .reset_index()
        .sort_values("decile")
    )
    output["conversion_rate"] = output["conversions"] / output["customers"]
    output["cumulative_customers"] = output["customers"].cumsum()
    output["cumulative_conversions"] = output["conversions"].cumsum()
    output["cumulative_conversion_rate"] = (
        output["cumulative_conversions"] / output["cumulative_customers"]
    )

    return output
