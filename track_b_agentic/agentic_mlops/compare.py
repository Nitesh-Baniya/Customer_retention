"""Export the MLflow run comparison + pick the version to promote."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"

COLS = [
    ("tags.prompt_version", "version"), ("tags.promotion_gate", "gate"),
    ("metrics.task_completion_rate", "completion"), ("metrics.strict_completion_rate", "strict_completion"),
    ("metrics.tool_call_correctness", "tool_correct"), ("metrics.verification_recall", "verif_recall"),
    ("metrics.entity_coverage_rate", "coverage"), ("metrics.error_rate", "error_rate"),
    ("metrics.pct_tests_passed", "pct_tests_passed"), ("metrics.correctness_pass_rate", "correctness"),
    ("metrics.completeness_pass_rate", "completeness"), ("metrics.avg_llm_iterations", "avg_iters"),
    ("metrics.avg_tool_calls_per_query", "avg_tool_calls"), ("metrics.avg_tokens_per_query", "avg_tokens"),
    ("metrics.avg_latency_s", "avg_latency_s"), ("run_id", "run_id"),
]


def export_comparison(experiment: str, offline: bool) -> Path:
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time ASC"])
    runs = runs[runs["tags.mode"] == ("offline_stub" if offline else "live")]
    # keep only the latest run per prompt version
    runs = runs.sort_values("start_time").groupby("tags.prompt_version", as_index=False).tail(1)
    table = runs[[c for c, _ in COLS if c in runs.columns]].rename(columns=dict(COLS))
    suffix = "_OFFLINE" if offline else ""
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / f"run_comparison{suffix}.csv", index=False)
    (REPORTS / f"run_comparison{suffix}.md").write_text(
        ("> OFFLINE STUB RUN - scripted policy, not a real model. Plumbing check only.\n\n" if offline else "")
        + table.drop(columns=["run_id"]).round(3).to_markdown(index=False) + "\n", encoding="utf-8")
    (REPORTS / f"promotion_decision{suffix}.json").write_text(
        json.dumps(choose_best(table), indent=2), encoding="utf-8")
    _export_traces(runs, suffix)
    return REPORTS / f"run_comparison{suffix}.md"


def _export_traces(runs: pd.DataFrame, suffix: str) -> None:
    """Copy each run's representative traces into the repo so reviewers can read them."""
    for _, r in runs.iterrows():
        dest = REPORTS / f"traces{suffix}" / r["tags.prompt_version"]
        dest.mkdir(parents=True, exist_ok=True)
        local = mlflow.artifacts.download_artifacts(run_id=r["run_id"], artifact_path="traces/representative")
        for f in Path(local).glob("*.json"):
            (dest / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")


def choose_best(table: pd.DataFrame) -> dict:
    """Promote only versions that pass the regression gate; among them highest strict
    completion, then highest judge pass rate, then fewest tokens."""
    ok = table[table["gate"] == "PASS"]
    if ok.empty:
        return {"promoted": None, "reason": "no version passed the Evidently regression gate"}
    best = ok.sort_values(["strict_completion", "pct_tests_passed", "avg_tokens"],
                          ascending=[False, False, True]).iloc[0]
    return {"promoted": best["version"], "run_id": best["run_id"],
            "reason": "passed regression gate; best strict completion / judge pass rate / token cost",
            "strict_completion": float(best["strict_completion"]),
            "pct_tests_passed": float(best["pct_tests_passed"]), "avg_tokens": float(best["avg_tokens"])}


if __name__ == "__main__":
    import argparse
    from agentic_mlops import tracking
    ap = argparse.ArgumentParser(); ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    print(export_comparison(tracking.setup(a.offline), a.offline))
