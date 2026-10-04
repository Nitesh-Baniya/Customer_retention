"""Harness metrics (W16 definitions, kept comparable) + failure diagnosis from traces."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from agentic_mlops.testset import TestCase
from agentic_mlops.traced_agent import ERROR, MAX_ITER, QueryTrace


@dataclass
class CaseScore:
    case_id: str
    success: bool            # W16 definition: answer + expected tool + verification-if-expected
    strict_success: bool     # success AND one tool call per entity
    tool_ok: bool
    verification_ok: bool
    coverage_ok: bool
    unneeded_verification: bool
    failure_mode: str        # "" when success
    diagnosis: str


def score_case(case: TestCase, tr: QueryTrace) -> CaseScore:
    tools = tr.tools_used
    answered = len(tr.answer.strip()) >= 10 and tr.termination_reason != ERROR
    tool_ok = any(t in tools for t in case.expected_tools)
    verification_ok = (not case.expected_verification) or tr.verification_performed
    n_expected_calls = sum(1 for t in tools if t in case.expected_tools)
    coverage_ok = n_expected_calls >= case.min_tool_calls
    success = answered and tool_ok and verification_ok
    mode, why = "", ""
    if not success:
        mode, why = _diagnose(case, tr, answered, tool_ok, verification_ok)
    elif not coverage_ok:
        mode = "incomplete_entity_coverage"
        why = (f"Made {n_expected_calls} call(s) to {case.expected_tools[0]} but the question names "
               f"{case.min_tool_calls} entities; the agent stopped after the first tool result.")
    return CaseScore(
        case_id=case.id, success=success, strict_success=success and coverage_ok,
        tool_ok=tool_ok, verification_ok=verification_ok, coverage_ok=coverage_ok,
        unneeded_verification=(not case.expected_verification) and tr.verification_performed,
        failure_mode=mode, diagnosis=why,
    )


def _diagnose(case: TestCase, tr: QueryTrace, answered: bool, tool_ok: bool, verif_ok: bool) -> tuple[str, str]:
    if tr.termination_reason == ERROR:
        return "crash", f"Loop raised: {tr.error}"
    if tr.termination_reason == MAX_ITER:
        return "max_iterations", f"Hit the iteration cap ({tr.llm_iterations}) without a final answer."
    if not answered:
        return "empty_answer", "Final answer empty/too short."
    if not tool_ok:
        used = tr.tools_used or "no tool"
        return "answered_from_memory", (
            f"Expected one of {list(case.expected_tools)} but the agent used {used}; "
            "it answered from parametric knowledge instead of calling a tool.")
    if not verif_ok:
        decision = next((s for s in tr.steps if s.phase == "verification_decision"), None)
        reason = decision.reasoning if decision else "no verification decision recorded"
        return "verifier_skipped", f"Verification was expected but not run. Verifier said: {reason}"
    return "other", "Unclassified."


def aggregate(cases: list[TestCase], traces: list[QueryTrace]) -> tuple[dict[str, float], list[CaseScore]]:
    scores = [score_case(c, t) for c, t in zip(cases, traces)]
    n = max(1, len(scores))
    exp_verif = [s for c, s in zip(cases, scores) if c.expected_verification]
    metrics = {
        "task_completion_rate": sum(s.success for s in scores) / n,
        "strict_completion_rate": sum(s.strict_success for s in scores) / n,
        "tool_call_correctness": sum(s.tool_ok for s in scores) / n,
        "verification_recall": (sum(s.verification_ok for s in exp_verif) / len(exp_verif)) if exp_verif else 1.0,
        "entity_coverage_rate": sum(s.coverage_ok for s in scores) / n,
        "unneeded_verification_rate": sum(s.unneeded_verification for s in scores) / n,
        "error_rate": sum(t.termination_reason == ERROR for t in traces) / n,
        "max_iterations_rate": sum(t.termination_reason == MAX_ITER for t in traces) / n,
        "avg_llm_iterations": sum(t.llm_iterations for t in traces) / n,
        "avg_steps_per_query": sum(len(t.steps) for t in traces) / n,
        "avg_tool_calls_per_query": sum(len(t.tools_used) for t in traces) / n,
        "total_tokens": float(sum(t.usage.get("total_tokens", 0) for t in traces)),
        "avg_tokens_per_query": sum(t.usage.get("total_tokens", 0) for t in traces) / n,
        "total_llm_calls": float(sum(t.usage.get("llm_calls", 0) for t in traces)),
        "avg_latency_s": sum(t.latency_s for t in traces) / n,
    }
    return {k: round(v, 4) for k, v in metrics.items()}, scores


def failure_summary(scores: list[CaseScore]) -> dict[str, int]:
    return dict(Counter(s.failure_mode for s in scores if s.failure_mode))


def to_rows(scores: list[CaseScore]) -> list[dict[str, Any]]:
    return [s.__dict__ for s in scores]
