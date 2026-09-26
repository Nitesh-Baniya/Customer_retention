"""Model performance monitoring and drift detection pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.metrics import DriftedColumnsCount, MeanValue, ValueDrift
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from retention.drift_metrics import BillingAmountMeanShift
from retention.features import NUMERIC_FIELDS
from retention.settings import (
    BILLING_DRIFT_THRESHOLD,
    MONITORING_EXPERIMENT_ID,
    MONITORED_FEATURES,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    REFERENCE_DATA_RATIO,
    TRACKING_SERVER_URI,
)
from retention.dataset import load_processed_data

PREDICTION_TARGET = "Churn"


def _simulate_production_drift(
    current_data: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Inject synthetic drift to simulate production data shifts."""
    drifted = current_data.copy()

    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] + rng.normal(
        loc=25.0, scale=8.0, size=len(drifted)
    )
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"].clip(lower=0)
    drifted["tenure"] = drifted["tenure"] + rng.integers(
        low=6, high=18, size=len(drifted)
    )
    drifted["tenure"] = drifted["tenure"].clip(lower=0, upper=72)

    force_count = max(1, int(0.45 * len(drifted)))
    force_indices = rng.choice(
        drifted.index.to_numpy(), size=force_count, replace=False
    )
    drifted.loc[force_indices, "Contract"] = "Month-to-month"

    flip_count = max(1, int(0.20 * len(drifted)))
    flip_indices = rng.choice(
        drifted.index.to_numpy(), size=flip_count, replace=False
    )
    drifted.loc[flip_indices, PREDICTION_TARGET] = (
        1 - drifted.loc[flip_indices, PREDICTION_TARGET]
    )

    return drifted


def _build_data_definition(include_target: bool = False) -> DataDefinition:
    """Construct Evidently data definition for drift analysis."""
    categorical_columns = [
        col
        for col in [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
    ]
    numeric_columns = list(NUMERIC_FIELDS)
    if include_target:
        numeric_columns = numeric_columns + [PREDICTION_TARGET]
    return DataDefinition(
        numerical_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def _convert_to_evidently_dataset(
    df: pd.DataFrame, include_target: bool = False
) -> Dataset:
    """Transform pandas DataFrame to Evidently Dataset."""
    frame = df if include_target else df.drop(columns=[PREDICTION_TARGET])
    return Dataset.from_pandas(
        frame, data_definition=_build_data_definition(include_target)
    )


def _extract_drifted_columns(
    metrics_json: list[dict[str, Any]]
) -> tuple[list[str], bool]:
    """Parse Evidently metrics to identify drifted columns."""
    drifted_features: list[str] = []
    dataset_drift_detected = False
    for metric in metrics_json:
        metric_name = metric.get("metric_name", "")
        metric_value = metric.get("value")
        metric_config = metric.get("config", {})
        if metric_name.startswith("DriftedColumnsCount"):
            if isinstance(metric_value, dict) and metric_value.get("count", 0) > 0:
                dataset_drift_detected = True
            continue
        if metric_name.startswith("ValueDrift") and "column" in metric_config:
            threshold = float(metric_config.get("threshold", 0.05))
            method = str(metric_config.get("method", "")).lower()
            try:
                score = float(metric_value)
            except (TypeError, ValueError):
                continue
            is_pvalue_method = (
                "p_value" in method
                or "p-value" in method
                or "pvalue" in method
            )
            is_drifted = (
                score < threshold if is_pvalue_method else score > threshold
            )
            if is_drifted:
                drifted_features.append(metric_config["column"])
    return sorted(set(drifted_features)), dataset_drift_detected or bool(
        drifted_features
    )


def _calculate_segment_drift(
    reference: pd.DataFrame, current: pd.DataFrame
) -> float:
    """Compute churn rate shift for month-to-month contract segment."""
    reference_segment = reference[
        reference["Contract"] == "Month-to-month"
    ]
    current_segment = current[current["Contract"] == "Month-to-month"]
    reference_rate = (
        float(reference_segment[PREDICTION_TARGET].mean())
        if len(reference_segment)
        else 0.0
    )
    current_rate = (
        float(current_segment[PREDICTION_TARGET].mean())
        if len(current_segment)
        else 0.0
    )
    return float(abs(current_rate - reference_rate))


def _generate_drift_reports(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[Path, Path, dict[str, Any], float]:
    """Create Evidently drift reports and extract metrics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_drift_report_path = OUTPUT_DIR / "feature_drift_report.html"
    target_drift_report_path = OUTPUT_DIR / "target_drift_report.html"

    reference_features = _convert_to_evidently_dataset(
        reference, include_target=False
    )
    current_features = _convert_to_evidently_dataset(
        current, include_target=False
    )

    feature_report = Report(
        [
            DataDriftPreset(),
            DriftedColumnsCount(),
            MeanValue(column="MonthlyCharges"),
            BillingAmountMeanShift(
                column="MonthlyCharges",
                threshold=BILLING_DRIFT_THRESHOLD,
            ),
        ]
    )
    feature_snapshot = feature_report.run(current_features, reference_features)
    feature_snapshot.save_html(str(feature_drift_report_path))
    feature_metrics = json.loads(feature_snapshot.json())["metrics"]

    drifted_columns, dataset_drift = _extract_drifted_columns(
        feature_metrics
    )

    custom_metric_value = None
    for metric in feature_metrics:
        if "BillingAmountMeanShift" in metric.get("metric_name", "") or (
            metric.get("config", {}).get("type", "").endswith(
                "BillingAmountMeanShift"
            )
        ):
            try:
                custom_metric_value = float(metric["value"])
            except (TypeError, ValueError, KeyError):
                pass
            break
    if custom_metric_value is None:
        custom_metric_value = float(
            abs(
                current["MonthlyCharges"].mean()
                - reference["MonthlyCharges"].mean()
            )
        )

    reference_full = _convert_to_evidently_dataset(
        reference, include_target=True
    )
    current_full = _convert_to_evidently_dataset(
        current, include_target=True
    )
    target_report = Report(
        [ValueDrift(column=PREDICTION_TARGET, threshold=0.05)]
    )
    target_snapshot = target_report.run(current_full, reference_full)
    target_snapshot.save_html(str(target_drift_report_path))
    target_metrics = json.loads(target_snapshot.json())["metrics"]
    target_drifted, _ = _extract_drifted_columns(target_metrics)
    drifted_columns = sorted(set(drifted_columns) | set(target_drifted))

    return (
        feature_drift_report_path,
        target_drift_report_path,
        {
            "drifted_columns": drifted_columns,
            "dataset_drift": bool(dataset_drift or drifted_columns),
        },
        custom_metric_value,
    )


def execute_monitoring_pipeline() -> dict[str, Any]:
    """Run drift detection, generate reports, and log to MLflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_dataset = load_processed_data()
    reference_data, current_data = train_test_split(
        full_dataset,
        test_size=1.0 - REFERENCE_DATA_RATIO,
        random_state=RANDOM_SEED,
        stratify=full_dataset[PREDICTION_TARGET],
    )
    reference_data = reference_data.reset_index(drop=True)
    current_data = current_data.reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_SEED)
    drifted_current = _simulate_production_drift(current_data, rng)

    (
        feature_report_path,
        target_report_path,
        drift_information,
        billing_shift,
    ) = _generate_drift_reports(reference_data, drifted_current)
    segment_drift = _calculate_segment_drift(
        reference_data, drifted_current
    )

    detected_monitored_features = [
        feature
        for feature in MONITORED_FEATURES
        if feature in drift_information["drifted_columns"]
    ]
    custom_threshold_exceeded = billing_shift >= BILLING_DRIFT_THRESHOLD
    significant_drift_detected = bool(
        drift_information["dataset_drift"]
        or detected_monitored_features
        or custom_threshold_exceeded
        or PREDICTION_TARGET in drift_information["drifted_columns"]
    )

    analysis_document = f"""# Drift Detection Analysis

## Configuration
- Reference dataset: random {REFERENCE_DATA_RATIO:.0%} of customer data (training distribution)
- Current dataset: remaining {(1 - REFERENCE_DATA_RATIO):.0%} with synthetic drift injected

## Simulated Production Shifts
- `MonthlyCharges`: shifted by ~N(25, 8) per customer record
- `tenure`: increased by random integer offset in [6, 18]
- `Contract`: ~45% of current records forced to `Month-to-month`
- `retention_status`: ~20% of labels flipped to simulate concept drift

## Custom Metrics
- Absolute mean difference in `MonthlyCharges`: **{billing_shift:.3f}**
  (threshold={BILLING_DRIFT_THRESHOLD})
- Absolute retention-rate shift within `Contract == Month-to-month`: **{segment_drift:.4f}**

## Detection Results
- Dataset drift flagged: **{drift_information['dataset_drift']}**
- Drifted columns: {drift_information['drifted_columns'] or 'none detected'}
- Monitored features flagged: {detected_monitored_features or 'none'}
- Significant drift (for automation): **{significant_drift_detected}**

## Production Impact
If this pattern occurred in production, feature and target distributions would
no longer match training data. Predictions would be unreliable, especially for
billing-sensitive and month-to-month customer segments. Recommended action:
investigate data pipelines, then retrain and validate before deploying new model.
"""
    analysis_path = OUTPUT_DIR / "drift_analysis.md"
    analysis_path.write_text(analysis_document)

    monitoring_summary = {
        "significant_drift": significant_drift_detected,
        "dataset_drift": drift_information["dataset_drift"],
        "drifted_columns": drift_information["drifted_columns"],
        "monitored_features_flagged": detected_monitored_features,
        "custom_metrics": {
            "billing_amount_mean_shift": billing_shift,
            "billing_drift_threshold": BILLING_DRIFT_THRESHOLD,
            "month_to_month_retention_rate_shift": segment_drift,
        },
        "artifacts": {
            "feature_drift_report": str(
                feature_report_path.relative_to(PROJECT_ROOT)
            ),
            "target_drift_report": str(
                target_report_path.relative_to(PROJECT_ROOT)
            ),
            "analysis_document": str(analysis_path.relative_to(PROJECT_ROOT)),
        },
        "recommendation": (
            "RETRAIN RECOMMENDED"
            if significant_drift_detected
            else "NO RETRAIN NEEDED"
        ),
    }
    summary_path = OUTPUT_DIR / "monitoring_summary.json"
    summary_path.write_text(json.dumps(monitoring_summary, indent=2))

    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    mlflow.set_experiment(MONITORING_EXPERIMENT_ID)
    with mlflow.start_run(run_name="drift_detection_check"):
        mlflow.log_metric("billing_amount_mean_shift", billing_shift)
        mlflow.log_metric(
            "month_to_month_retention_rate_shift", segment_drift
        )
        mlflow.log_metric(
            "significant_drift",
            1.0 if significant_drift_detected else 0.0,
        )
        mlflow.log_param("reference_data_ratio", REFERENCE_DATA_RATIO)
        mlflow.log_param(
            "monitored_features", ",".join(MONITORED_FEATURES)
        )
        for artifact_path in (
            feature_report_path,
            target_report_path,
            analysis_path,
            summary_path,
        ):
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

    print(json.dumps(monitoring_summary, indent=2))
    return monitoring_summary


def main() -> None:
    """Entry point for monitoring pipeline."""
    execute_monitoring_pipeline()


if __name__ == "__main__":
    main()
