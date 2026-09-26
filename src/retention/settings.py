"""Application settings and configuration management."""

from __future__ import annotations

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports"
MLFLOW_STORAGE = PROJECT_ROOT / "mlruns"

# MLflow configuration
EXPERIMENT_ID = "customer-retention-analytics"
MONITORING_EXPERIMENT_ID = "retention-monitoring"
MODEL_REGISTRY_NAME = "customer-retention-predictor"
TRACKING_SERVER_URI = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_STORAGE.as_uri())

# Data processing settings
RANDOM_SEED = 42
VALIDATION_SPLIT_RATIO = 0.2
REFERENCE_DATA_RATIO = 0.7

# Drift detection thresholds
BILLING_DRIFT_THRESHOLD = 5.0
MONITORED_FEATURES = ("MonthlyCharges", "tenure", "Contract")
