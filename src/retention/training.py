"""Model training, evaluation, and registry management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from retention.settings import (
    EXPERIMENT_ID,
    MODEL_REGISTRY_NAME,
    OUTPUT_DIR,
    TRACKING_SERVER_URI,
)
from retention.dataset import create_train_test_split
from retention.features import assemble_model_pipeline


def _initialize_output_directory() -> None:
    """Ensure output directory exists for reports and artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _generate_confusion_matrix_plot(
    y_true, y_pred, output_path: Path
) -> None:
    """Create and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _generate_roc_curve_plot(
    y_true, y_proba, output_path: Path
) -> None:
    """Create and save ROC curve visualization."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _calculate_performance_metrics(
    y_true, y_pred, y_proba
) -> dict[str, float]:
    """Compute comprehensive model performance metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _get_model_configurations() -> list[dict[str, Any]]:
    """Define model variants with distinct hyperparameters."""
    return [
        {
            "run_name": "logistic_baseline_regularized",
            "model_type": "logistic_regression",
            "hyperparameters": {
                "C": 0.1,
                "max_iter": 1000,
                "solver": "lbfgs",
            },
            "estimator": LogisticRegression(
                C=0.1, max_iter=1000, solver="lbfgs", random_state=42
            ),
        },
        {
            "run_name": "random_forest_conservative",
            "model_type": "random_forest",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_leaf": 5,
                "random_state": 42,
            },
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
            ),
        },
        {
            "run_name": "gradient_boosting_optimized",
            "model_type": "gradient_boosting",
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.05,
                "random_state": 42,
            },
            "estimator": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
            ),
        },
    ]


def execute_training_pipeline() -> pd.DataFrame:
    """Train all model variants, log results, and promote best performer."""
    _initialize_output_directory()
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    mlflow.set_experiment(EXPERIMENT_ID)

    X_train, X_test, y_train, y_test = create_train_test_split()
    experiment_results: list[dict[str, Any]] = []

    for config in _get_model_configurations():
        model_pipeline = assemble_model_pipeline(
            config["estimator"], X_train
        )
        with mlflow.start_run(run_name=config["run_name"]) as run:
            mlflow.set_tag("model_family", config["model_type"])
            mlflow.log_params(config["hyperparameters"])
            mlflow.log_param("model_architecture", config["model_type"])

            model_pipeline.fit(X_train, y_train)
            test_predictions = model_pipeline.predict(X_test)
            test_probabilities = model_pipeline.predict_proba(X_test)[:, 1]
            performance = _calculate_performance_metrics(
                y_test, test_predictions, test_probabilities
            )
            mlflow.log_metrics(performance)

            cm_plot_path = OUTPUT_DIR / f"confusion_matrix_{config['run_name']}.png"
            roc_plot_path = OUTPUT_DIR / f"roc_curve_{config['run_name']}.png"
            _generate_confusion_matrix_plot(
                y_test, test_predictions, cm_plot_path
            )
            _generate_roc_curve_plot(
                y_test, test_probabilities, roc_plot_path
            )
            mlflow.log_artifact(str(cm_plot_path))
            mlflow.log_artifact(str(roc_plot_path))

            mlflow.sklearn.log_model(
                model_pipeline,
                artifact_path="model",
                registered_model_name=None,
                input_example=X_test.head(3),
            )

            result_row = {
                "run_id": run.info.run_id,
                "run_name": config["run_name"],
                "model_family": config["model_type"],
                **config["hyperparameters"],
                **performance,
            }
            experiment_results.append(result_row)
            print(
                f"[{config['run_name']}] "
                f"roc_auc={performance['roc_auc']:.4f} "
                f"f1={performance['f1_score']:.4f}"
            )

    comparison_df = pd.DataFrame(experiment_results)
    comparison_df = comparison_df.sort_values(
        by=["roc_auc", "f1_score"], ascending=False
    ).reset_index(drop=True)
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("\nModel performance comparison (sorted by ROC-AUC, then F1):")
    print(
        comparison_df[
            [
                "run_name",
                "model_family",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    best_model = comparison_df.iloc[0]
    best_run_id = best_model["run_id"]
    best_model_uri = f"runs:/{best_run_id}/model"

    registry_client = MlflowClient()
    registration_result = mlflow.register_model(
        best_model_uri, MODEL_REGISTRY_NAME
    )
    model_version = registration_result.version

    registry_client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=model_version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Registered {MODEL_REGISTRY_NAME} v{model_version} -> Staging")

    registry_client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=model_version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted {MODEL_REGISTRY_NAME} v{model_version} -> Production")

    training_summary = {
        "best_run_id": best_run_id,
        "best_run_name": best_model["run_name"],
        "best_roc_auc": float(best_model["roc_auc"]),
        "best_f1_score": float(best_model["f1_score"]),
        "registered_model": MODEL_REGISTRY_NAME,
        "model_version": model_version,
        "registry_stages": ["Staging", "Production"],
    }
    summary_path = OUTPUT_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(training_summary, indent=2))
    print(
        f"\nBest model: {best_model['run_name']} "
        f"(roc_auc={best_model['roc_auc']:.4f})"
    )
    return comparison_df


def main() -> None:
    """Entry point for training pipeline."""
    execute_training_pipeline()


if __name__ == "__main__":
    main()
