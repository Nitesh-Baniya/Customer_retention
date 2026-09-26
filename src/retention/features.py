"""Feature engineering and preprocessing pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def identify_categorical_columns(feature_df: pd.DataFrame) -> list[str]:
    """Extract categorical column names from feature dataframe."""
    return [col for col in feature_df.columns if col not in NUMERIC_FIELDS]


def create_feature_transformer(feature_df: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
    categorical_cols = identify_categorical_columns(feature_df)
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FIELDS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def assemble_model_pipeline(
    estimator: Any, feature_df: pd.DataFrame
) -> Pipeline:
    """Construct complete sklearn pipeline with preprocessing and model."""
    return Pipeline(
        steps=[
            ("feature_processor", create_feature_transformer(feature_df)),
            ("predictor", estimator),
        ]
    )
