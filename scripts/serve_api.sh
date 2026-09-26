#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$PROJECT_ROOT/mlruns}"
uv run uvicorn retention.api:app --host 0.0.0.0 --port 8000
