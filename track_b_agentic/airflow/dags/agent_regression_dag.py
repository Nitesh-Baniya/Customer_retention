"""Nightly regression-eval DAG for the agentic assistant (Track B, optional bonus).

    run_eval  ->  check_degradation --(degraded)--> alert_degradation
                                    \\-(healthy)---> mark_healthy

* run_eval            re-runs the fixed query set through the *currently promoted* prompt
                      version (reports/promotion_decision.json) and logs an MLflow run
                      (harness metrics + Evidently LLM-judge regression suite).
* check_degradation   compares that run with the baseline thresholds below.
* alert_degradation   fails the task loudly (Airflow alert / e-mail on failure) and writes
                      reports/degradation_alert.json. Hook Slack/e-mail in ``on_failure_callback``.

Setup: ``uv sync --extra airflow``; export AIRFLOW_HOME=$PWD/airflow; set HF_TOKEN (+ JUDGE_*).
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT = Path(os.environ.get("TRACK_B_DIR", Path(__file__).resolve().parents[2]))
REPORTS = PROJECT / "reports"
UV = os.environ.get("UV_BIN", "uv")
OFFLINE = os.environ.get("AGENT_EVAL_OFFLINE", "0") == "1"   # CI mode: scripted LLM, no network

# Degradation thresholds (absolute floors / allowed drops vs. the promoted baseline).
MIN_PCT_TESTS_PASSED = 80.0      # Evidently LLM-judge regression pass rate (%)
MIN_TASK_COMPLETION = 0.75       # W16 harness
MAX_ERROR_RATE = 0.10
MAX_TOKEN_INCREASE = 0.30        # >30% more tokens/query than baseline = cost regression


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT, check=True)


def run_eval(**_):
    version = "v3"
    decision = REPORTS / f"promotion_decision{'_OFFLINE' if OFFLINE else ''}.json"
    if decision.exists() and json.loads(decision.read_text()).get("promoted"):
        version = json.loads(decision.read_text())["promoted"]
    cmd = [UV, "run", "python", "-m", "agentic_mlops.run_experiment", "--versions", version]
    _run(cmd + (["--offline"] if OFFLINE else []))
    (REPORTS / "last_nightly_version.txt").write_text(version)


def _latest_metrics(version: str) -> tuple[dict, dict | None]:
    import mlflow
    from agentic_mlops import tracking

    exp = tracking.setup(OFFLINE)
    runs = mlflow.search_runs(experiment_names=[exp], order_by=["start_time DESC"])
    runs = runs[runs["tags.prompt_version"] == version]
    cols = {c: c.removeprefix("metrics.") for c in runs.columns if c.startswith("metrics.")}
    latest = runs.iloc[0].rename(cols).to_dict()
    prev = runs.iloc[1].rename(cols).to_dict() if len(runs) > 1 else None
    return latest, prev


def check_degradation(**_) -> str:
    version = (REPORTS / "last_nightly_version.txt").read_text().strip()
    latest, prev = _latest_metrics(version)
    problems = []
    if latest["pct_tests_passed"] < MIN_PCT_TESTS_PASSED:
        problems.append(f"pct_tests_passed {latest['pct_tests_passed']} < {MIN_PCT_TESTS_PASSED}")
    if latest["task_completion_rate"] < MIN_TASK_COMPLETION:
        problems.append(f"task_completion_rate {latest['task_completion_rate']} < {MIN_TASK_COMPLETION}")
    if latest["error_rate"] > MAX_ERROR_RATE:
        problems.append(f"error_rate {latest['error_rate']} > {MAX_ERROR_RATE}")
    if prev and prev["avg_tokens_per_query"] and \
            latest["avg_tokens_per_query"] > prev["avg_tokens_per_query"] * (1 + MAX_TOKEN_INCREASE):
        problems.append("token cost per query up more than 30% vs previous nightly run")
    (REPORTS / "degradation_alert.json").write_text(
        json.dumps({"version": version, "problems": problems, "checked_at": datetime.utcnow().isoformat()}, indent=2))
    return "alert_degradation" if problems else "mark_healthy"


def alert_degradation(**_):
    alert = json.loads((REPORTS / "degradation_alert.json").read_text())
    # Failing the task triggers Airflow's configured failure alerting (e-mail/Slack callback).
    raise RuntimeError(f"Agent regression detected for prompt {alert['version']}: {alert['problems']}")


with DAG(
    dag_id="agent_regression_eval",
    description="Nightly W16-harness + Evidently LLM-judge regression eval with degradation alert",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "agentic", "evidently", "mlflow"],
) as dag:
    t_eval = PythonOperator(task_id="run_eval", python_callable=run_eval)
    t_check = BranchPythonOperator(task_id="check_degradation", python_callable=check_degradation)
    t_alert = PythonOperator(task_id="alert_degradation", python_callable=alert_degradation)
    t_ok = EmptyOperator(task_id="mark_healthy")
    t_eval >> t_check >> [t_alert, t_ok]
