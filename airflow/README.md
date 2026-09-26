# Local Airflow Setup (Customer Retention Monitoring)

The DAG [`dags/retention_monitoring_dag.py`](dags/retention_monitoring_dag.py) runs weekly:

1. `run_drift_analysis` — `uv run python -m retention.monitoring`
2. `assess_drift_impact` — reads `reports/monitoring_summary.json`
3. On significant drift → `execute_retraining` (`uv run python -m retention.training`); otherwise `skip_retraining`

## Install Airflow (optional extra)

From the project root:

```bash
uv sync --extra airflow
```

## Configure AIRFLOW_HOME

```bash
export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export MLFLOW_TRACKING_URI="$(pwd)/mlruns"
# Ensure `uv` is on PATH (or set UV_BIN to the absolute uv binary)
```

## Initialize and start

```bash
cd /path/to/retention-analytics
uv run airflow db migrate
uv run airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com

# All-in-one (webserver + scheduler)
uv run airflow standalone
```

Open http://localhost:8080 and trigger `customer_retention_drift_monitor` manually for a demo.

## Notes

- The DAG resolves the project root as the parent of `airflow/`, so it works when `AIRFLOW_HOME` is this `airflow/` directory.
- Core ML work (`uv sync` without extras) does **not** require Airflow; install the extra only when running the DAG.
