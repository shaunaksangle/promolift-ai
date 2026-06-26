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
        "treated_customers": int(len(targeted_treated)),
        "control_customers": int(len(targeted_control)),
        "treated_conversions": int(targeted_treated["outcome"].sum()),
        "control_conversions": int(targeted_control["outcome"].sum()),
        "observed_conversion_rate_targeted_treated": targeted_treatment_rate,
        "observed_conversion_rate_targeted_control": targeted_control_rate,
        "observed_incremental_conversion_rate": incremental_conversion_rate,
        "estimated_incremental_conversions": float(
            incremental_conversion_rate * targeted_customers
        ),
    }


def calculate_qini_curve(y_true, treatment, uplift_score) -> pd.DataFrame:
    """Calculate a Qini-style cumulative incremental conversion curve.

    Customers are sorted by predicted uplift from highest to lowest. The model
    curve estimates cumulative incremental conversions as more users are
    targeted, using the adjusted control formula:

    treated conversions - control conversions * treated customers / control customers
    """
    ranked_df = pd.DataFrame(
        {
            "outcome": pd.Series(y_true).astype(int).reset_index(drop=True),
            "treatment": pd.Series(treatment).astype(int).reset_index(drop=True),
            "uplift_score": pd.Series(uplift_score).reset_index(drop=True),
        }
    ).sort_values("uplift_score", ascending=False)

    ranked_df["targeted_customers"] = range(1, len(ranked_df) + 1)
    ranked_df["targeted_fraction"] = ranked_df["targeted_customers"] / len(ranked_df)
    ranked_df["treated_customer"] = (ranked_df["treatment"] == 1).astype(int)
    ranked_df["control_customer"] = (ranked_df["treatment"] == 0).astype(int)
    ranked_df["treated_conversion"] = (
        (ranked_df["treatment"] == 1) & (ranked_df["outcome"] == 1)
    ).astype(int)
    ranked_df["control_conversion"] = (
        (ranked_df["treatment"] == 0) & (ranked_df["outcome"] == 1)
    ).astype(int)

    qini_df = pd.DataFrame(
        {
            "targeted_fraction": ranked_df["targeted_fraction"],
            "targeted_customers": ranked_df["targeted_customers"],
            "treated_customers": ranked_df["treated_customer"].cumsum(),
            "control_customers": ranked_df["control_customer"].cumsum(),
            "treated_conversions": ranked_df["treated_conversion"].cumsum(),
            "control_conversions": ranked_df["control_conversion"].cumsum(),
        }
    )
    qini_df["cumulative_incremental_conversions"] = qini_df.apply(
        _calculate_adjusted_incremental_conversions,
        axis=1,
    )

    total_incremental_conversions = qini_df[
        "cumulative_incremental_conversions"
    ].iloc[-1]
    qini_df["random_incremental_conversions"] = (
        total_incremental_conversions * qini_df["targeted_fraction"]
    )
    qini_df["qini_gain"] = (
        qini_df["cumulative_incremental_conversions"]
        - qini_df["random_incremental_conversions"]
    )

    return qini_df.reset_index(drop=True)


def calculate_qini_coefficient(qini_df: pd.DataFrame) -> float:
    """Approximate the Qini coefficient as area above random targeting."""
    if qini_df.empty:
        return 0.0

    x_values = [0.0] + qini_df["targeted_fraction"].astype(float).tolist()
    y_values = [0.0] + qini_df["qini_gain"].astype(float).tolist()

    area = 0.0
    for index in range(1, len(x_values)):
        width = x_values[index] - x_values[index - 1]
        average_height = (y_values[index] + y_values[index - 1]) / 2
        area += width * average_height

    return float(area)


def make_uplift_calibration_table(
    y_true,
    treatment,
    uplift_score,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compare average predicted uplift with observed uplift by decile.

    Decile 1 contains the highest predicted uplift scores. The table is meant
    to show whether uplift magnitude is calibrated or mainly useful for ranking.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

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
        q=n_bins,
        labels=False,
    ).astype(int) + 1

    rows = []
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
        observed_uplift = treatment_conversion_rate - control_conversion_rate
        average_predicted_uplift = float(decile_data["uplift_score"].mean())

        rows.append(
            {
                "decile": int(decile),
                "customers": int(len(decile_data)),
                "treated_customers": int(len(treated)),
                "control_customers": int(len(control)),
                "treatment_conversion_rate": treatment_conversion_rate,
                "control_conversion_rate": control_conversion_rate,
                "observed_uplift": observed_uplift,
                "average_predicted_uplift": average_predicted_uplift,
                "calibration_gap": average_predicted_uplift - observed_uplift,
            }
        )

    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)


def calculate_ate_confidence_interval(
    df: pd.DataFrame,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Calculate a two-proportion confidence interval for observed ATE."""
    treatment_outcomes = df.loc[df["treatment"] == 1, "outcome"].astype(int)
    control_outcomes = df.loc[df["treatment"] == 0, "outcome"].astype(int)

    treatment_customers = int(treatment_outcomes.shape[0])
    control_customers = int(control_outcomes.shape[0])
    treatment_conversions = int(treatment_outcomes.sum())
    control_conversions = int(control_outcomes.sum())

    treatment_rate = _safe_rate(treatment_conversions, treatment_customers)
    control_rate = _safe_rate(control_conversions, control_customers)
    ate = treatment_rate - control_rate
    standard_error = math.sqrt(
        _safe_variance(treatment_rate, treatment_customers)
        + _safe_variance(control_rate, control_customers)
    )
    z_critical = _z_critical_for_confidence(confidence)
    ci_lower = ate - z_critical * standard_error
    ci_upper = ate + z_critical * standard_error
    z_statistic = ate / standard_error if standard_error > 0 else 0.0
    p_value = 2 * (1 - _normal_cdf(abs(z_statistic)))

    return {
        "treatment_conversion_rate": float(treatment_rate),
        "control_conversion_rate": float(control_rate),
        "ate": float(ate),
        "standard_error": float(standard_error),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "z_statistic": float(z_statistic),
        "p_value": float(p_value),
        "confidence_level": float(confidence),
        "treatment_conversions": treatment_conversions,
        "treatment_customers": treatment_customers,
        "control_conversions": control_conversions,
        "control_customers": control_customers,
    }


def _calculate_adjusted_incremental_conversions(row: pd.Series) -> float:
    """Calculate adjusted incremental conversions for one Qini row."""
    if row["control_customers"] == 0:
        return 0.0

    expected_control_conversions = row["control_conversions"] * (
        row["treated_customers"] / row["control_customers"]
    )

    return float(row["treated_conversions"] - expected_control_conversions)


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


def _safe_variance(proportion: float, sample_size: int) -> float:
    """Return binomial proportion variance with empty-sample protection."""
    if sample_size == 0:
        return 0.0

    return proportion * (1 - proportion) / sample_size


def _normal_cdf(value: float) -> float:
    """Approximate the standard normal CDF using math.erf."""
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _z_critical_for_confidence(confidence: float) -> float:
    """Return a common normal critical value for a confidence level."""
    common_values = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }

    return common_values.get(round(confidence, 2), 1.959963984540054)
