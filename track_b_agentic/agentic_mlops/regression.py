"""Evidently regression test suite with an LLM-as-a-judge.

New responses (one per fixed query) are compared with approved golden references using two
judge checks:

1. ``Correctness`` (reference-based): does the response contradict, or lose information
   present in, the reference answer?  (``BinaryClassificationPromptTemplate``)
2. ``Completeness`` (question-based): does the response address every part of the query?

Per-row verdicts become boolean test columns (``<check>: equals correct``); suite-level
Evidently tests then gate on the pass rate of each check. ``pct_tests_passed`` is the share
of (case x check) tests that passed - that is what is logged to MLflow.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.descriptors import LLMEval
from evidently.legacy.options.base import Options
from evidently.legacy.utils.llm.wrapper import LLMResult, LLMWrapper, RateLimits, llm_provider
from evidently.llm.models import LLMMessage
from evidently.llm.templates import BinaryClassificationPromptTemplate
from evidently.llm.utils.wrapper import OpenAIKey
from evidently.metrics import MeanValue
from evidently.presets import TextEvals
from evidently.tests import eq, gte

# A version is promotable only if BOTH suite-level pass-rate tests hold.
CHECK_PASS_RATE_THRESHOLD = 0.8
CHECKS = ("Correctness", "Completeness")


def _templates() -> dict[str, BinaryClassificationPromptTemplate]:
    return {
        "Correctness": BinaryClassificationPromptTemplate(
            criteria=(
                "You are checking a NEW assistant response against an approved REFERENCE answer.\n"
                "REFERENCE: {reference}\n"
                "The response is INCORRECT if it contradicts the reference, or drops a key fact, "
                "number, entity or unit that the reference contains. Different wording, extra "
                "correct detail, and different *live* values (current weather, current time) are "
                "fine as long as the response reports the same kind of information the reference "
                "describes and does not claim it could not obtain it."
            ),
            target_category="incorrect", non_target_category="correct",
            uncertainty="unknown", include_reasoning=True,
            pre_messages=[LLMMessage.system(
                "You are a strict, impartial QA reviewer. Judge only what is written.")],
        ),
        "Completeness": BinaryClassificationPromptTemplate(
            criteria=(
                "The user's QUESTION was: {query}\n"
                "The response is INCOMPLETE if any named entity or sub-question in the QUESTION "
                "(e.g. one of several cities or countries, or the second half of a two-part "
                "question) is not addressed with concrete information."
            ),
            target_category="incomplete", non_target_category="complete",
            uncertainty="unknown", include_reasoning=True,
            pre_messages=[LLMMessage.system(
                "You are a strict, impartial QA reviewer. Judge only what is written.")],
        ),
    }


# Judge verdict that counts as a PASS for each check.
PASS_LABEL = {"Correctness": "correct", "Completeness": "complete"}


# ---- offline heuristic judge (CI only; NOT an LLM) -------------------------------------
@llm_provider("heuristic_offline", None)
class HeuristicJudge(LLMWrapper):
    """Deterministic keyword/number judge used by ``--offline``. Not a substitute for an LLM."""

    def __init__(self, model: str, options: Any) -> None:
        self.model = model

    async def complete(self, messages: List[LLMMessage], seed: Optional[int] = None) -> LLMResult[str]:
        text = "\n".join(m.content for m in messages)
        resp = text.split("___text_starts_here___")[-1].split("___text_ends_here___")[0].strip().lower()
        if "REFERENCE:" in text:
            ref = text.split("REFERENCE:")[1].split("\nThe response is")[0].lower()
            nums = re.findall(r"\d+(?:\.\d+)?", ref)
            names = [w for w in re.findall(r"\b(new york|london|tokyo|paris|brazil|india)\b", ref)]
            nums = [n for n in nums if len(n) > 2]
            resp_nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", resp)]
            close = lambda n: any(abs(float(n) - r) <= 0.005 * max(1.0, abs(float(n))) for r in resp_nums)  # noqa: E731
            missing_n = nums if nums and not any(n in resp or close(n) for n in nums) else []
            missing_names = [n for n in names if n not in resp]
            gave_up = "do not have" in resp or "can help with that" in resp
            bad = bool(missing_names) or gave_up or bool(missing_n)
            cat = "incorrect" if bad else "correct"
            why = f"missing={missing_names or missing_n}; gave_up={gave_up}"
        else:
            q = text.split("The user's QUESTION was:")[1].split("\n")[0].lower()
            ents = [c for c in ("new york", "london", "tokyo", "paris", "brazil", "india") if c in q]
            miss = [e for e in ents if e not in resp]
            cat = "incomplete" if miss else "complete"
            why = f"entities not addressed: {miss}"
        return LLMResult(json.dumps({"category": cat, "reasoning": why}), 0, 0)

    def get_limits(self) -> RateLimits:
        return RateLimits()


@dataclass
class RegressionResult:
    metrics: dict[str, float]
    rows: pd.DataFrame
    report_path: Path
    gate_passed: bool
    suite_tests: list[dict[str, Any]]


def judge_settings(offline: bool) -> tuple[str, str, Options]:
    if offline:
        return "heuristic_offline", "heuristic", Options()
    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set JUDGE_API_KEY (or HF_TOKEN / OPENAI_API_KEY) for the LLM judge, "
                           "or run with --offline.")
    url = os.environ.get("JUDGE_BASE_URL", "https://router.huggingface.co/v1")
    model = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra")
    return "openai", model, Options(custom={OpenAIKey: OpenAIKey(api_key=key, api_url=url)})


def run_regression(
    responses: pd.DataFrame,          # columns: id, query, response
    golden: dict[str, str],
    out_html: Path,
    offline: bool = False,
) -> RegressionResult:
    provider, model, options = judge_settings(offline)
    df = responses.copy()
    df["reference"] = df["id"].map(golden)
    if df["reference"].isna().any():
        raise KeyError(f"No golden reference for: {df[df.reference.isna()].id.tolist()}")
    df["response"] = df["response"].fillna("").replace("", "(empty response)")

    templates = _templates()
    descriptors = [
        LLMEval(
            "response", template=templates["Correctness"], provider=provider, model=model,
            alias="Correctness", additional_columns={"reference": "reference"},
            tests=[eq(PASS_LABEL["Correctness"], column="Correctness")],
        ),
        LLMEval(
            "response", template=templates["Completeness"], provider=provider, model=model,
            alias="Completeness", additional_columns={"query": "query"},
            tests=[eq(PASS_LABEL["Completeness"], column="Completeness")],
        ),
    ]
    dataset = Dataset.from_pandas(
        df,
        data_definition=DataDefinition(text_columns=["query", "response", "reference"]),
        descriptors=descriptors,
        options=options,
    )
    out = dataset.as_dataframe()

    pass_cols = {c: f"{c}: equals {PASS_LABEL[c]}" for c in CHECKS}
    report = Report([
        TextEvals(),
        *[MeanValue(column=col, tests=[gte(CHECK_PASS_RATE_THRESHOLD)]) for col in pass_cols.values()],
    ])
    snapshot = report.run(dataset, None)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_html))

    suite = [{"name": t.name, "status": str(t.status.value), "description": t.description}
             for t in snapshot.tests_results]
    gate = bool(suite) and all(t["status"] == "SUCCESS" for t in suite)

    flags = pd.DataFrame({c: out[col].astype(bool) for c, col in pass_cols.items()})
    n_tests = flags.size
    metrics = {
        "pct_tests_passed": round(float(flags.values.sum()) / n_tests * 100, 2),
        "pct_cases_passed": round(float(flags.all(axis=1).mean()) * 100, 2),
        **{f"{c.lower()}_pass_rate": round(float(flags[c].mean()), 4) for c in CHECKS},
        "suite_gate_passed": float(gate),
        "n_failed_cases": float((~flags.all(axis=1)).sum()),
    }
    rows = pd.DataFrame({
        "id": df["id"].values, "query": df["query"].values,
        "response": df["response"].values, "reference": df["reference"].values,
        **{f"{c}_verdict": out[c].values for c in CHECKS},
        **{f"{c}_reasoning": out[f"{c} reasoning"].values for c in CHECKS},
        "passed_all": flags.all(axis=1).values,
    })
    return RegressionResult(metrics, rows, out_html, gate, suite)
