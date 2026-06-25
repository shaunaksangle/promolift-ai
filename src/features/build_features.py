"""Feature engineering utilities for baseline conversion modeling.

This module defines the pre-campaign customer features used by the normal
machine learning baseline model.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_columns() -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature columns for the baseline model.

    These are all pre-campaign customer features. Treatment assignment and
    post-campaign outcome columns are intentionally excluded to avoid leakage.

    Returns:
        A tuple containing numeric feature names and categorical feature names.
    """
    numeric_features = ["recency", "history", "mens", "womens", "newbie"]
    categorical_features = ["history_segment", "zip_code", "channel"]

    return numeric_features, categorical_features


def split_features_target(df):
    """Split the processed Hillstrom data into model features and target.

    Args:
        df: Processed Hillstrom DataFrame.

    Returns:
        X feature DataFrame and y target Series.
    """
    numeric_features, categorical_features = get_feature_columns()
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns].copy()
    y = df["outcome"].astype(int).copy()

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing steps for numeric and categorical features.

    Numeric features are imputed with the median and scaled. Categorical
    features are imputed with the most frequent value and one-hot encoded.

    Returns:
        A scikit-learn ColumnTransformer ready to use in a Pipeline.
    """
    numeric_features, categorical_features = get_feature_columns()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder that works across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
