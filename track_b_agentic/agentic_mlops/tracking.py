"""MLflow setup shared by the runner, the comparer and the Airflow DAG."""
from __future__ import annotations

import os
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_LIVE = "agentic-assistant-prompts"
EXPERIMENT_OFFLINE = "agentic-assistant-prompts-OFFLINE-STUB"


def setup(offline: bool) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    name = EXPERIMENT_OFFLINE if offline else EXPERIMENT_LIVE
    mlflow.set_experiment(name)
    return name
