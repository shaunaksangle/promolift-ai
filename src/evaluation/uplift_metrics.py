"""Evaluation helpers for uplift modeling.

These utilities summarize treatment-control differences after customers are
ranked by predicted uplift. They are portfolio-friendly diagnostics, not a
replacement for formal causal validation.
"""

import math

import pandas as pd


def calculate_average_treatment_effect(df: pd.DataFrame) -> float:
    """Calculate the observed average treatment effect in the full dataset.

    ATE is measured as the conversion rate among treated customers minus the
    conversion rate among control customers.
    """
    treatment_rate = df.loc[df["treatment"] == 1, "outcome"].mean()
    control_rate = df.loc[df["treatment"] == 0, "outcome"].mean()

    return float(treatment_rate - control_rate)


def make_uplift_decile_table(y_true, treatment, uplift_score) -> pd.DataFrame:
    """Create deciles after ranking customers by predicted uplift.

    Decile 1 contains the customers with the highest predicted uplift scores.
    Observed uplift is calculated from actual treatment-control outcomes inside
    each decile, so empty treatment or control cells are handled safely.
    """
    ranked_df = pd.DataFrame(
        {
            "outcome": pd.Series(y_true).astype(int).reset_index(drop=True),
            "treatment": pd.Series(treatment).astype(int).reset_index(drop=True),
            "uplift_score": pd.Series(uplift_score).reset_index(drop=True),
        }
    )
    ranked_df["rank"] = ranked_df["uplift_score"].rank(
        method="first",
        ascending=False,
    )
    ranked_df["decile"] = pd.qcut(
        ranked_df["rank"],
        q=10,
        labels=False,
    ).astype(int) + 1

    decile_rows = []

    for decile, decile_data in ranked_df.groupby("decile", sort=True):
        treated = decile_data[decile_data["treatment"] == 1]
        control = decile_data[decile_data["treatment"] == 0]
        treatment_conversion_rate = _safe_rate(
            treated["outcome"].sum(),
            len(treated),
        )
        control_conversion_rate = _safe_rate(
            control["outcome"].sum(),
            len(control),
        )

        decile_rows.append(
            {
                "decile": int(decile),
                "customers": int(len(decile_data)),
                "treated_customers": int(len(treated)),
                "control_customers": int(len(control)),
                "treated_conversions": int(treated["outcome"].sum()),
                "control_conversions": int(control["outcome"].sum()),
                "treatment_conversion_rate": treatment_conversion_rate,
                "control_conversion_rate": control_conversion_rate,
                "observed_uplift": treatment_conversion_rate
                - control_conversion_rate,
                "average_predicted_uplift": float(
                    decile_data["uplift_score"].mean()
                ),
            }
        )

    decile_df = pd.DataFrame(decile_rows).sort_values("decile")
    decile_df["cumulative_customers"] = decile_df["customers"].cumsum()
    decile_df["cumulative_treated_customers"] = (
        decile_df["treated_customers"].cumsum()
    )
    decile_df["cumulative_control_customers"] = (
        decile_df["control_customers"].cumsum()
    )
    decile_df["cumulative_treated_conversions"] = (
        decile_df["treated_conversions"].cumsum()
    )
    decile_df["cumulative_control_conversions"] = (
        decile_df["control_conversions"].cumsum()
    )
    decile_df["cumulative_observed_uplift"] = decile_df.apply(
        _calculate_cumulative_observed_uplift,
        axis=1,
    )

    return decile_df[
        [
            "decile",
            "customers",
            "treated_customers",
            "control_customers",
            "treated_conversions",
            "control_conversions",
            "treatment_conversion_rate",
            "control_conversion_rate",
            "observed_uplift",
            "average_predicted_uplift",
            "cumulative_customers",
            "cumulative_treated_customers",
            "cumulative_control_customers",
            "cumulative_observed_uplift",
        ]
    ]


def calculate_top_decile_uplift(decile_df: pd.DataFrame) -> float:
    """Return observed uplift for decile 1."""
    top_decile = decile_df.loc[decile_df["decile"] == 1, "observed_uplift"]

    if top_decile.empty:
        return 0.0

    return float(top_decile.iloc[0])


def calculate_policy_value(
    y_true,
    treatment,
    uplift_score,
    top_percent: float = 0.3,
) -> dict[str, float | int]:
    """Estimate performance from targeting only the top uplift-ranked users.

    This is a simple portfolio estimate based on observed randomized treatment
    and control outcomes inside the targeted group. Later causal validation
    should discuss assumptions and robustness more formally.
    """
    if top_percent <= 0 or top_percent > 1:
        raise ValueError("top_percent must be greater than 0 and at most 1.")

    policy_df = pd.DataFrame(
        {
            "outcome": pd.Series(y_true).astype(int).reset_index(drop=True),
            "treatment": pd.Series(treatment).astype(int).reset_index(drop=True),
            "uplift_score": pd.Series(uplift_score).reset_index(drop=True),
        }
    ).sort_values("uplift_score", ascending=False)
    targeted_customers = max(1, math.ceil(len(policy_df) * top_percent))
    targeted = policy_df.head(targeted_customers)
    targeted_treated = targeted[targeted["treatment"] == 1]
    targeted_control = targeted[targeted["treatment"] == 0]

    targeted_treatment_rate = _safe_rate(
        targeted_treated["outcome"].sum(),
        len(targeted_treated),
    )
    targeted_control_rate = _safe_rate(
        targeted_control["outcome"].sum(),
        len(targeted_control),
    )
    incremental_conversion_rate = targeted_treatment_rate - targeted_control_rate

    return {
        "target_percent": float(top_percent),
        "targeted_customers": int(targeted_customers),
        "observed_conversion_rate_targeted_treated": targeted_treatment_rate,
        "observed_conversion_rate_targeted_control": targeted_control_rate,
        "observed_incremental_conversion_rate": incremental_conversion_rate,
        "estimated_incremental_conversions": float(
            incremental_conversion_rate * targeted_customers
        ),
    }


def _calculate_cumulative_observed_uplift(row: pd.Series) -> float:
    """Calculate cumulative observed uplift for one decile row."""
    cumulative_treatment_rate = _safe_rate(
        row["cumulative_treated_conversions"],
        row["cumulative_treated_customers"],
    )
    cumulative_control_rate = _safe_rate(
        row["cumulative_control_conversions"],
        row["cumulative_control_customers"],
    )

    return cumulative_treatment_rate - cumulative_control_rate


def _safe_rate(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, using 0 when denominator is empty."""
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)
