"""Airflow DAG: automated drift monitoring with conditional retraining."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports"
MONITORING_SUMMARY = OUTPUT_DIRECTORY / "monitoring_summary.json"
PACKAGE_MANAGER = os.environ.get("UV_BIN", "uv")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri()
)

default_dag_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


def _evaluate_retraining_necessity(**_context) -> str:
    """Determine if model retraining is required based on drift detection."""
    if not MONITORING_SUMMARY.exists():
        print("No monitoring summary found — skipping retraining cycle.")
        return "skip_retraining"
    summary_data = json.loads(MONITORING_SUMMARY.read_text())
    if summary_data.get("significant_drift"):
        print("RETRAINING TRIGGERED — significant drift detected.")
        print(json.dumps(summary_data, indent=2))
        return "execute_retraining"
    print("NO RETRAINING — drift within acceptable thresholds.")
    return "skip_retraining"


def _log_retraining_skip(**_context) -> None:
    """Log when retraining is skipped due to stable model performance."""
    print("Monitoring cycle complete; no retraining required.")


with DAG(
    dag_id="customer_retention_drift_monitor",
    description="Weekly drift detection with automated retraining trigger",
    default_args=default_dag_args,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["retention", "drift-detection", "mlops", "monitoring"],
) as dag:
    run_drift_analysis = BashOperator(
        task_id="run_drift_analysis",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'MLFLOW_TRACKING_URI="{MLFLOW_TRACKING_URI}" '
            f'{PACKAGE_MANAGER} run python -m retention.monitoring'
        ),
        env={
            **os.environ,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "PATH": os.environ.get("PATH", ""),
        },
    )

    assess_drift_impact = BranchPythonOperator(
        task_id="assess_drift_impact",
        python_callable=_evaluate_retraining_necessity,
    )

    execute_retraining = BashOperator(
        task_id="execute_retraining",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'MLFLOW_TRACKING_URI="{MLFLOW_TRACKING_URI}" '
            f'{PACKAGE_MANAGER} run python -m retention.training'
        ),
        env={
            **os.environ,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "PATH": os.environ.get("PATH", ""),
        },
    )

    skip_retraining = PythonOperator(
        task_id="skip_retraining",
        python_callable=_log_retraining_skip,
    )

    (
        run_drift_analysis
        >> assess_drift_impact
        >> [execute_retraining, skip_retraining]
    )
