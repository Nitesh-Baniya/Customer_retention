"""Trace-driven failure diagnosis: what went wrong in each version, step by step.

    uv run python -m agentic_mlops.diagnose [--offline]

Reads the traces + case scores that each MLflow run logged and writes
``reports/failure_diagnosis[_OFFLINE].md``. This is the input for the *next* prompt revision:
each prompt version must answer a failure listed here, not a speculative tweak.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from agentic_mlops import tracking

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def _latest_runs(experiment: str, offline: bool) -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=[experiment])
    runs = runs[runs["tags.mode"] == ("offline_stub" if offline else "live")]
    return runs.sort_values("start_time").groupby("tags.prompt_version", as_index=False).tail(1) \
               .sort_values("tags.prompt_version")


def _fmt_steps(trace: dict) -> str:
    lines = []
    for s in trace["steps"]:
        tool = f" `{s['tool']}`({json.dumps(s['args'])})" if s.get("tool") else ""
        res = (s.get("result") or "")[:110].replace("\n", " ")
        lines.append(f"  {s['step']}. **{s['phase']}**{tool} - {('reasoning: ' + str(s['reasoning'])[:120]) if s.get('reasoning') else ''}"
                     + (f"\n     result: `{res}`" if res else ""))
    return "\n".join(lines)


def build(experiment: str, offline: bool) -> str:
    out = ["# Failure diagnosis from traces\n"]
    if offline:
        out.append("> OFFLINE STUB RUN - illustrates the workflow; the agent here is a scripted policy.\n")
    for _, run in _latest_runs(experiment, offline).iterrows():
        v = run["tags.prompt_version"]
        scores = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run.run_id, artifact_path="harness/case_scores.csv"))
        traces = {}
        p = mlflow.artifacts.download_artifacts(run_id=run.run_id, artifact_path="traces/all_traces.jsonl")
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            traces[t["case_id"]] = t
        failed = scores[scores["failure_mode"].fillna("") != ""]
        out.append(f"\n## prompt {v} - {len(failed)} / {len(scores)} cases with a failure mode\n")
        if failed.empty:
            out.append("No failures.\n")
        by_mode = failed.groupby("failure_mode")
        for mode, grp in by_mode:
            out.append(f"\n### `{mode}` ({len(grp)} case(s))\n")
            first = grp.iloc[0]
            out.append(f"Diagnosis: {first['diagnosis']}\n")
            out.append("Cases: " + ", ".join(grp["case_id"]) + "\n")
            out.append(f"Example trace `{first['case_id']}` (termination: {traces[first['case_id']]['termination_reason']}):\n")
            out.append(_fmt_steps(traces[first["case_id"]]) + "\n")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    exp = tracking.setup(a.offline)
    text = build(exp, a.offline)
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"failure_diagnosis{'_OFFLINE' if a.offline else ''}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
