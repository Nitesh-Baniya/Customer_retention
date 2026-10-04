"""Sanity-check the LLM judge against a human reading of the same responses.

1. After a run, open the run's ``regression/judge_audit.csv`` (also written to
   ``reports/judge_audit_<version>.csv`` by this script) and fill the ``human_correct`` /
   ``human_complete`` columns with ``yes`` / ``no`` after reading each response yourself.
2. Re-run this script: it reports raw agreement and lists every disagreement - those are
   cases where the judge (or your reading) is wrong/biased.

    uv run python -m agentic_mlops.judge_audit --version v3 [--offline]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from agentic_mlops import tracking
from agentic_mlops.compare import REPORTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    exp = tracking.setup(a.offline)
    runs = mlflow.search_runs(experiment_names=[exp])
    runs = runs[runs["tags.prompt_version"] == a.version].sort_values("start_time")
    run_id = runs.iloc[-1]["run_id"]
    src = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="regression/judge_audit.csv"))
    dst = REPORTS / f"judge_audit_{a.version}{'_OFFLINE' if a.offline else ''}.csv"
    df = pd.read_csv(src)
    if dst.exists():  # keep the human labels already entered
        old = pd.read_csv(dst)
        for col in ("human_correct", "human_complete"):
            if col in old:
                df[col] = df["id"].map(old.set_index("id")[col])
    for col in ("human_correct", "human_complete"):
        if col not in df:
            df[col] = ""
    df.to_csv(dst, index=False)

    labelled = df[df["human_correct"].isin(["yes", "no"]) & df["human_complete"].isin(["yes", "no"])]
    if labelled.empty:
        print(f"Wrote {dst}. Fill human_correct / human_complete (yes/no) and re-run.")
        return
    j_c = labelled["Correctness_verdict"].eq("correct")
    j_p = labelled["Completeness_verdict"].eq("complete")
    h_c, h_p = labelled["human_correct"].eq("yes"), labelled["human_complete"].eq("yes")
    agree = (j_c == h_c).sum() + (j_p == h_p).sum()
    total = 2 * len(labelled)
    print(f"Judge-human agreement: {agree}/{total} = {agree / total:.0%}")
    bad = labelled[(j_c != h_c) | (j_p != h_p)]
    for _, r in bad.iterrows():
        print(f"- DISAGREE {r['id']}: judge=({r['Correctness_verdict']},{r['Completeness_verdict']}) "
              f"human=({r['human_correct']},{r['human_complete']}) | {r['Correctness_reasoning']}")
    if a.version:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("judge_human_agreement", agree / total)


if __name__ == "__main__":
    main()
