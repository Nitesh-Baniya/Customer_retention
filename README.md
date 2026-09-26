# Customer Retention Analytics Platform

A production-grade MLOps pipeline for predicting customer attrition using the IBM Telco Customer Churn dataset (~7,000 customers).

**Complete workflow: data processing → model training → MLflow tracking → model registry → FastAPI serving → Evidently monitoring → Airflow-orchestrated retraining**

## Technology Stack

| Component | Technology |
|-----------|------------|
| Environment Management | [uv](https://github.com/astral-sh/uv) (`pyproject.toml` + `uv.lock`) |
| Experiment Tracking & Registry | MLflow 2.x (local `./mlruns`) |
| Machine Learning Models | Logistic Regression, Random Forest, Gradient Boosting (scikit-learn) |
| Model Serving | FastAPI (serves from `models:/customer-retention-predictor/Production`) |
| Drift Detection | Evidently AI (HTML reports + custom metrics) |
| Workflow Orchestration | Apache Airflow (optional extra) |

---

## Architecture & Design Decisions

### Environment Reproducibility (uv)

This pipeline addresses dependency management challenges through strict version control:

- **Python version pinning** — MLflow 2.x + Evidently + Airflow require consistent interpreter versions; we pin **3.11** via `.python-version` and `requires-python = ">=3.11,<3.13"`.
- **MLflow major version stability** — `mlflow>=2.14,<3` prevents breaking API changes that would invalidate our `sklearn.log_model` workflow and registry transitions.
- **Airflow as optional dependency** — `apache-airflow>=2.9,<2.11` is heavyweight and conflict-prone; core train/serve/monitor functionality works without it. Install with `uv sync --extra airflow` when needed.
- **Deterministic lockfile** — `uv.lock` freezes all transitive dependencies (scikit-learn, Evidently, FastAPI, etc.) ensuring identical environments across different machines.

**Quick start from fresh clone:**
```bash
git clone <repo-url> retention-analytics && cd retention-analytics
uv sync
./scripts/train_model.sh      # trains 3+ models, logs to MLflow, promotes best to Production
./scripts/monitor_drift.sh    # Evidently drift report → reports/monitoring_summary.json
# optional serving: ./scripts/serve_api.sh
# optional orchestration: uv sync --extra airflow  (see airflow/README.md)
```

### Experiment Tracking Strategy (MLflow)

**Model variants and hyperparameters**

Three distinct model families are trained and compared in the `customer-retention-analytics` experiment:

| Run Name | Model Family | Key Hyperparameters |
|----------|--------------|---------------------|
| `logistic_baseline_regularized` | Logistic Regression | `C=0.1`, `max_iter=1000`, `solver=lbfgs` |
| `random_forest_conservative` | Random Forest | `n_estimators=100`, `max_depth=5`, `min_samples_leaf=5` |
| `gradient_boosting_optimized` | Gradient Boosting | `n_estimators=200`, `max_depth=3`, `learning_rate=0.05` |

**Rationale:** Compare linear baseline vs. bagging vs. boosting on identical train/test splits and feature pipelines. This ensures performance differences reflect inductive bias and model capacity—not data leakage or preprocessing variations.

**Evaluation metrics**

Per run: **accuracy, precision, recall, F1, ROC-AUC**, plus confusion matrix and ROC curve artifacts.

Given the class imbalance (~26.5% positive cases), accuracy alone is misleading. We rank models by **ROC-AUC** (with F1 as tie-breaker) for registry promotion.

**Performance comparison (from MLflow / `reports/model_comparison.csv`)**

| Run | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-----|----------|-----------|--------|-----|---------|
| `gradient_boosting_optimized` | 0.7925 | 0.6306 | 0.5294 | 0.5756 | **0.8392** |
| `logistic_baseline_regularized` | **0.8053** | **0.6543** | **0.5668** | **0.6074** | 0.8352 |
| `random_forest_conservative` | 0.7854 | 0.6488 | 0.4198 | 0.5097 | 0.8341 |

Full results: [`reports/model_comparison.csv`](reports/model_comparison.csv). UI: `uv run mlflow ui --backend-store-uri ./mlruns --port 5000` → experiment **customer-retention-analytics**.

**Model selection and promotion**

Only **`gradient_boosting_optimized`** was registered as `customer-retention-predictor` and promoted Staging → Production (`reports/training_summary.json`: ROC-AUC **0.8392**, F1 **0.5756**).

Decision rationale:
- Gradient Boosting achieved the **highest ROC-AUC (0.8392)** vs. Logistic Regression (0.8352) and Random Forest (0.8341).
- Logistic Regression had higher accuracy (0.8053 vs 0.7925) and F1 (0.6074 vs 0.5756), but we did not select it because accuracy overstates performance on imbalanced data.
- Random Forest's recall collapsed to 0.4198 (vs 0.5294 for GB and 0.5668 for LR), making it unsuitable for retention use cases where identifying at-risk customers is critical.

**Trade-off accepted:** ~1.3 pp lower accuracy and ~3.2 pp lower F1 versus Logistic Regression, in exchange for superior discrimination (ROC-AUC) for scoring and ranking at-risk customers.

### Monitoring & Drift Detection (Evidently AI)

**Reference vs. current data split**

- **Reference** — random **70%** of the cleaned Telco dataset: baseline "training-time" feature/label distribution.
- **Current** — remaining **30%**, with deliberate perturbations to simulate production data shift (noise on `MonthlyCharges` / `tenure`, Contract skewed toward `Month-to-month`, ~20% of labels flipped).

**Monitored metrics**

- Evidently **data drift** and **target drift** (HTML reports).
- Custom Evidently metric **`BillingAmountMeanShift`** vs threshold **5.0**.
- Segment analysis: absolute retention-rate shift within `Contract == Month-to-month`.
- Aggregated flag `significant_drift` written to `reports/monitoring_summary.json` for Airflow automation.

**Typical drift detection results** (`reports/monitoring_summary.json` / `reports/drift_analysis.md`)

| Signal | Value |
|--------|-------|
| Dataset drift flagged | `true` |
| Drifted columns | `retention_status`, `Contract`, `MonthlyCharges`, `tenure` |
| Injected features detected | `MonthlyCharges`, `tenure`, `Contract` |
| `MonthlyCharges` mean abs diff | **24.41** (threshold 5.0) |
| Month-to-month retention-rate shift | **0.0275** |
| Recommendation | **RETRAIN RECOMMENDED** |

Artifacts: `reports/feature_drift_report.html`, `reports/target_drift_report.html`, `reports/drift_analysis.md`.

**Response to detected drift**

When `significant_drift` is true (dataset drift and/or custom threshold exceeded):

1. Treat Production model predictions as unreliable until investigated (potential pipeline issues or population shift).
2. Automated response: Airflow branches to **retraining** (`python -m retention.training`), which creates new MLflow runs, re-selects by ROC-AUC, and re-promotes to Production.
3. Manual follow-up: review HTML reports, confirm the shift is genuine (not a broken data feed), then validate the new Production model before trusting live predictions.

### Workflow Orchestration

DAG id: **`customer_retention_drift_monitor`** ([`airflow/dags/retention_monitoring_dag.py`](airflow/dags/retention_monitoring_dag.py)).

| Component | Details |
|-----------|---------|
| **Schedule** | `@weekly` (`catchup=False`) |
| **Trigger condition** | After `run_drift_analysis`, `assess_drift_impact` reads `reports/monitoring_summary.json` and branches on `significant_drift == true` |
| **On drift detection** | `execute_retraining` → `uv run python -m retention.training` (new MLflow runs + registry promotion) |
| **Otherwise** | `skip_retraining` (log only; Production model unchanged) |

```
run_drift_analysis  →  assess_drift_impact  →  execute_retraining
                                          ↘  skip_retraining
```

Setup: [`airflow/README.md`](airflow/README.md).

---

## Installation

Requires Python 3.11–3.12 (pinned via `.python-version`).

```bash
uv sync
```

Airflow orchestration (optional):

```bash
uv sync --extra airflow
```

## Usage

### 1. Train models and register best performer

```bash
./scripts/train_model.sh
# or: uv run python -m retention.training
```

Compare experiments in the MLflow UI:

```bash
uv run mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open http://localhost:5000 → experiment **customer-retention-analytics**.

### 2. Serve the Production model via REST API

```bash
./scripts/serve_api.sh
# or: uv run uvicorn retention.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Example prediction request:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "customers": [{
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "No",
      "MultipleLines": "No phone service",
      "InternetService": "DSL",
      "OnlineSecurity": "No",
      "OnlineBackup": "Yes",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 29.85,
      "TotalCharges": 29.85
    }]
  }' | python -m json.tool
```

### 3. Run drift monitoring

```bash
./scripts/monitor_drift.sh
# or: uv run python -m retention.monitoring
```

See **Monitoring & Drift Detection** section above for reference/current semantics, metrics, and response policy.

### 4. Airflow orchestration (optional)

See **Workflow Orchestration** section and [`airflow/README.md`](airflow/README.md).

## Project Structure

```
src/retention/       # Core modules: data, features, training, API, monitoring
airflow/dags/        # Weekly drift monitoring DAG
scripts/             # Convenience shell scripts
reports/             # Model comparisons, drift HTML reports, summaries
mlruns/              # Local MLflow tracking store (gitignored)
dataset.csv          # Telco Customer Churn dataset
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `./mlruns` (file URI) | MLflow tracking / registry location |
| `UV_BIN` | `uv` | Package manager used by Airflow DAG |
| `AIRFLOW_HOME` | — | Set to `./airflow` for local DAG development |

## Key Features

- **Multi-model experimentation** with automatic best-model selection
- **Production-ready API** with health checks and batch prediction support
- **Automated drift detection** with custom metrics and segment analysis
- **Orchestrated retraining** pipeline triggered by drift detection
- **Reproducible environment** with strict dependency management
- **Comprehensive monitoring** via MLflow and Evidently dashboards
