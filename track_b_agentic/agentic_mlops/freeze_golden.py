"""Freeze reviewed responses of a chosen version as the new golden references.

    uv run python -m agentic_mlops.freeze_golden --version v3

Do this only after a human has read the responses. Overwrites ``data/golden.jsonl``.
"""
from __future__ import annotations

import argparse
import json

import mlflow
import pandas as pd

from agentic_mlops import tracking
from agentic_mlops.testset import GOLDEN_PATH

ap = argparse.ArgumentParser()
ap.add_argument("--version", required=True)
a = ap.parse_args()
exp = tracking.setup(offline=False)
runs = mlflow.search_runs(experiment_names=[exp])
run = runs[runs["tags.prompt_version"] == a.version].sort_values("start_time").iloc[-1]
df = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run["run_id"], artifact_path="regression/responses.csv"))
with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
    for _, r in df.iterrows():
        f.write(json.dumps({"id": r["id"], "query": r["query"], "reference": r["response"]}) + "\n")
print(f"Wrote {len(df)} golden references from {a.version} to {GOLDEN_PATH}")
