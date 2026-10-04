"""Run every prompt/config version over the fixed query set and log to MLflow.

    uv run python -m agentic_mlops.run_experiment            # live: needs HF_TOKEN (+ judge key)
    uv run python -m agentic_mlops.run_experiment --offline  # no network; pipeline smoke test only

Per version (= one MLflow run) this logs
  params    prompt_version, model, temperature, max_iterations, ...
  metrics   W16 harness metrics, token cost, latency, Evidently regression pass rates
  artifacts prompt text, ALL traces (JSONL), representative success/failure traces,
            case scores, responses, Evidently HTML report, judge audit CSV
  traces    MLflow spans (LLM turns + tool calls) linked to the run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
from dataclasses import replace
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.entities import SpanType

from agentic_mlops import tracking
from agentic_mlops.compare import export_comparison
from agentic_mlops.config import EXPERIMENTS, RunConfig, load_prompt
from agentic_mlops.regression import CHECK_PASS_RATE_THRESHOLD, run_regression
from agentic_mlops.scoring import aggregate, failure_summary, to_rows
from agentic_mlops.testset import TEST_CASES, load_golden
from agentic_mlops.traced_agent import QueryTrace, TracedVerificationAgent

logger = logging.getLogger("agentic_mlops")
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def build_runtime(cfg: RunConfig, system_prompt: str, offline: bool):
    from app.core.config import Settings

    if offline:
        from agentic_mlops.offline_stub import ScriptedLLMClient, offline_registry

        settings = Settings(hf_token="offline", llm_temperature=cfg.temperature,
                            llm_max_tool_iterations=cfg.max_iterations)
        return settings, ScriptedLLMClient(system_prompt), offline_registry()

    from app.tools.builtin import create_default_tool_registry
    from agentic_mlops.usage_client import UsageTrackingLLMClient

    settings = Settings(llm_temperature=cfg.temperature, llm_max_tool_iterations=cfg.max_iterations)
    return settings, UsageTrackingLLMClient(settings), create_default_tool_registry(settings)


def pick_representative(traces: list[QueryTrace], success_ids: set[str]) -> list[QueryTrace]:
    """1 clean success + up to 2 failures (the failures drive the next prompt revision)."""
    ok = [t for t in traces if t.case_id in success_ids]
    bad = [t for t in traces if t.case_id not in success_ids]
    # prefer a success that used a tool and verification so it is a *rich* example
    ok.sort(key=lambda t: (-len(t.steps)))
    return ok[:1] + bad[:2]


async def run_version(cfg: RunConfig, offline: bool, golden: dict[str, str]) -> dict:
    system_prompt = load_prompt(cfg.prompt_version)
    settings, llm, registry = build_runtime(cfg, system_prompt, offline)
    agent = TracedVerificationAgent(llm, registry, settings, cfg, system_prompt)
    mode = "offline_stub" if offline else "live"

    with mlflow.start_run(run_name=cfg.run_name()) as run:
        mlflow.log_params(cfg.as_params())
        mlflow.log_param("n_test_cases", len(TEST_CASES))
        mlflow.set_tags({"mode": mode, "prompt_version": cfg.prompt_version,
                         "change_summary": cfg.description,
                         "judge": "heuristic_offline" if offline else "llm"})
        mlflow.log_text(system_prompt, f"prompts/prompt_{cfg.prompt_version}.txt")

        traces: list[QueryTrace] = []
        for case in TEST_CASES:
            logger.info("[%s] %s", cfg.prompt_version, case.id)
            with mlflow.start_span(name=f"query:{case.id}", span_type=SpanType.AGENT) as span:
                span.set_inputs({"query": case.query, "prompt_version": cfg.prompt_version})
                tr = await agent.run_traced(case.id, case.query)
                span.set_outputs({"answer": tr.answer, "termination": tr.termination_reason})
            traces.append(tr)
        if hasattr(llm, "close"):
            await llm.close()

        metrics, scores = aggregate(list(TEST_CASES), traces)
        success_ids = {s.case_id for s in scores if s.strict_success}

        # --- Evidently regression suite (LLM judge) -------------------------------
        responses = pd.DataFrame(
            [{"id": t.case_id, "query": t.query, "response": t.answer} for t in traces])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            reg = run_regression(responses, golden, tmp / f"regression_{cfg.prompt_version}.html", offline)
            metrics.update(reg.metrics)
            mlflow.log_metrics(metrics)

            gate = reg.gate_passed
            mlflow.set_tags({"promotion_gate": "PASS" if gate else "FAIL",
                             "gate_rule": f"each judge check pass-rate >= {CHECK_PASS_RATE_THRESHOLD}"})

            # --- artifacts ---------------------------------------------------------
            (tmp / "all_traces.jsonl").write_text(
                "\n".join(json.dumps(t.to_dict(), default=str) for t in traces), encoding="utf-8")
            mlflow.log_artifact(str(tmp / "all_traces.jsonl"), "traces")
            for t in pick_representative(traces, success_ids):
                label = "success" if t.case_id in success_ids else "failure"
                mlflow.log_dict(t.to_dict(), f"traces/representative/{label}_{t.case_id}.json")
            pd.DataFrame(to_rows(scores)).to_csv(tmp / "case_scores.csv", index=False)
            mlflow.log_artifact(str(tmp / "case_scores.csv"), "harness")
            mlflow.log_dict(failure_summary(scores), "harness/failure_modes.json")
            responses.to_csv(tmp / "responses.csv", index=False)
            mlflow.log_artifact(str(tmp / "responses.csv"), "regression")
            reg.rows.to_csv(tmp / "judge_audit.csv", index=False)
            mlflow.log_artifact(str(tmp / "judge_audit.csv"), "regression")
            mlflow.log_dict(reg.suite_tests, "regression/suite_tests.json")
            mlflow.log_artifact(str(reg.report_path), "regression")

            # keep a copy of each version's Evidently report in the repo
            REPORTS.mkdir(exist_ok=True)
            (REPORTS / f"evidently_regression_{cfg.prompt_version}{'_OFFLINE' if offline else ''}.html").write_text(
                reg.report_path.read_text(encoding="utf-8"), encoding="utf-8")

        return {"run_id": run.info.run_id, "version": cfg.prompt_version, "metrics": metrics,
                "failures": failure_summary(scores), "gate": gate}


async def main_async(args: argparse.Namespace) -> None:
    experiment = tracking.setup(args.offline)
    golden = load_golden()
    configs = [c for c in EXPERIMENTS if not args.versions or c.prompt_version in args.versions]
    if args.temperature is not None:
        configs = [replace(c, temperature=args.temperature) for c in configs]
    if args.model:
        configs = [replace(c, model=args.model) for c in configs]
    results = []
    for cfg in configs:
        results.append(await run_version(cfg, args.offline, golden))
    path = export_comparison(experiment, offline=args.offline)
    print(f"\nExperiment: {experiment}")
    for r in results:
        print(f"  {r['version']}: gate={'PASS' if r['gate'] else 'FAIL'}  "
              f"completion={r['metrics']['task_completion_rate']:.2f}  "
              f"strict={r['metrics']['strict_completion_rate']:.2f}  "
              f"pct_tests_passed={r['metrics']['pct_tests_passed']}  failures={r['failures']}")
    print(f"Comparison written to {path}")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offline", action="store_true", help="scripted LLM + canned weather + heuristic judge (no network)")
    p.add_argument("--versions", nargs="*", help="subset of prompt versions, e.g. v1 v3")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--model", default=None, help="override agent model (disables W15 fallback)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    cli()
