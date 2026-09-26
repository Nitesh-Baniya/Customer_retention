"""Data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from retention.settings import DATA_FILE, RANDOM_SEED, VALIDATION_SPLIT_RATIO

PREDICTION_TARGET = "Churn"
CUSTOMER_ID_FIELD = "customerID"


def load_source_data(file_path: Path | None = None) -> pd.DataFrame:
    """Load raw customer data from CSV file."""
    source_path = file_path or DATA_FILE
    return pd.read_csv(source_path)


def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform raw customer data."""
    cleaned = raw_df.copy()

    # Remove customer identifier
    if CUSTOMER_ID_FIELD in cleaned.columns:
        cleaned = cleaned.drop(columns=[CUSTOMER_ID_FIELD])

    # Convert total charges to numeric
    cleaned["TotalCharges"] = pd.to_numeric(
        cleaned["TotalCharges"], errors="coerce"
    )
    cleaned = cleaned.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Encode target variable
    if cleaned[PREDICTION_TARGET].dtype == object:
        cleaned[PREDICTION_TARGET] = cleaned[PREDICTION_TARGET].map(
            {"Yes": 1, "No": 0}
        )

    cleaned[PREDICTION_TARGET] = cleaned[PREDICTION_TARGET].astype(int)
    return cleaned


def load_processed_data(file_path: Path | None = None) -> pd.DataFrame:
    """Load and preprocess the complete dataset."""
    return preprocess_data(load_source_data(file_path))


def separate_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataset into feature matrix and target vector."""
    features = df.drop(columns=[PREDICTION_TARGET])
    target = df[PREDICTION_TARGET]
    return features, target


def create_train_test_split(
    file_path: Path | None = None,
    test_ratio: float = VALIDATION_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Generate stratified train/test split for model development."""
    dataset = load_processed_data(file_path)
    X, y = separate_features_and_target(dataset)
    return train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
