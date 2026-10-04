"""Instrumented version of the W16 ``VerificationAssistantAgent`` loop.

It reuses the W16 agent's building blocks (``_evaluate_verification_need``,
``_perform_verification``, tool registry, skill manager) but runs the loop itself so that
every step is written out as a structured record::

    {step, iteration, phase, tool, args, result, success, reasoning, latency_s}

and the loop ends with an explicit ``termination_reason``. Every LLM / tool step is also
emitted as an MLflow span so the trace shows up in the MLflow UI next to the run.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
from mlflow.entities import SpanType

from app.assistant.prompts import build_system_prompt  # noqa: F401  (documented dependency)
from app.assistant.verification_agent import VerificationAssistantAgent
from app.llm.client import LLMError
from app.schemas.chat import AssistantOutput
from app.tools.registry import ToolRegistry

from agentic_mlops.config import RunConfig

MAX_RESULT_CHARS = 4000

# termination reasons
SUCCESS_DIRECT = "success_no_verification_needed"
SUCCESS_VERIFIED = "success_after_verification"
VERIFY_BUDGET = "success_verification_budget_exhausted"
MAX_ITER = "max_iterations_reached"
ERROR = "error"


@dataclass
class Step:
    step: int
    iteration: int
    phase: str            # tool_call | draft_answer | verification_decision | verification_search | final_answer
    tool: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    success: bool | None = None
    reasoning: str | None = None
    latency_s: float = 0.0


@dataclass
class QueryTrace:
    case_id: str
    query: str
    prompt_version: str
    answer: str = ""
    termination_reason: str = ""
    error: str | None = None
    llm_iterations: int = 0
    verification_rounds: int = 0
    steps: list[Step] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    model: str | None = None

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps if s.tool and s.phase in ("tool_call", "verification_search")]

    @property
    def verification_performed(self) -> bool:
        return self.verification_rounds > 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tools_used"] = self.tools_used
        d["verification_performed"] = self.verification_performed
        return d


def _cut(text: str | None) -> str | None:
    if text is None:
        return None
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "...[truncated]"


def _message_reasoning(message: Any) -> str | None:
    """Best-effort 'why': assistant text emitted alongside a tool call, or the
    provider's reasoning field (gpt-oss / DeepSeek expose ``reasoning(_content)``)."""
    for attr in ("reasoning", "reasoning_content"):
        val = getattr(message, attr, None)
        if val:
            return str(val)
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_content"):
        if extra.get(key):
            return str(extra[key])
    return message.content or None


class TracedVerificationAgent(VerificationAssistantAgent):
    def __init__(
        self,
        llm_client: Any,
        tool_registry: ToolRegistry,
        settings: Any,
        run_config: RunConfig,
        system_prompt: str,
        skills_dir: Path | None = None,
    ) -> None:
        super().__init__(llm_client, tool_registry, settings, skills_dir)
        self._cfg = run_config
        self._system_prompt = system_prompt
        self._max_iterations = run_config.max_iterations
        self._max_verification_steps = run_config.max_verification_rounds

    # Same as W16 ``_build_messages`` but with the *versioned* system prompt.
    def _build_messages(self, question, history, context):  # type: ignore[override]
        skill_context = self._skill_manager.get_context_augmentation(question)
        enhanced = f"{context}\n\n{skill_context}" if context else skill_context
        prompt = (
            self._system_prompt
            + "\n\n<document_context>\n"
            + enhanced
            + "\n</document_context>"
        )
        msgs: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
        msgs.extend(m.model_dump() for m in history or [])
        msgs.append({"role": "user", "content": question})
        return msgs

    async def run_traced(self, case_id: str, question: str) -> QueryTrace:
        trace = QueryTrace(case_id=case_id, query=question, prompt_version=self._cfg.prompt_version)
        usage_before = dict(self._llm.usage.snapshot()) if hasattr(self._llm, "usage") else {}
        t_start = time.perf_counter()
        try:
            await self._loop(question, trace)
        except LLMError as exc:
            trace.termination_reason, trace.error = ERROR, f"LLMError: {exc}"
        except Exception as exc:  # keep the harness going; failure is itself a data point
            trace.termination_reason, trace.error = ERROR, f"{type(exc).__name__}: {exc}"
        trace.latency_s = round(time.perf_counter() - t_start, 3)
        if hasattr(self._llm, "usage"):
            after = self._llm.usage.snapshot()
            trace.usage = {k: after[k] - usage_before.get(k, 0) for k in after}
        return trace

    def _add(self, trace: QueryTrace, iteration: int, phase: str, **kw: Any) -> Step:
        step = Step(step=len(trace.steps) + 1, iteration=iteration, phase=phase, **kw)
        trace.steps.append(step)
        return step

    async def _loop(self, question: str, trace: QueryTrace) -> None:
        messages = self._build_messages(question, None, None)
        active_model: str | None = self._cfg.model
        executions: list[Any] = []

        for iteration in range(1, self._max_iterations + 1):
            trace.llm_iterations = iteration
            t0 = time.perf_counter()
            with mlflow.start_span(name=f"llm_turn_{iteration}", span_type=SpanType.CHAT_MODEL) as span:
                span.set_inputs({"n_messages": len(messages)})
                completion = await self._llm.complete(
                    messages,
                    response_model=AssistantOutput,
                    tools=self._tools.schemas(),
                    model=active_model,
                )
                span.set_outputs({"tool_calls": [tc.function.name for tc in completion.message.tool_calls or []]})
            trace.model = completion.model
            if completion.used_fallback:
                active_model = completion.model
            llm_latency = time.perf_counter() - t0

            tool_calls = completion.message.tool_calls or []
            if tool_calls:
                reasoning = _message_reasoning(completion.message)
                messages.append(completion.message.model_dump(exclude_none=True))
                import json as _json

                for tc in tool_calls:
                    if tc.type != "function":
                        continue
                    t1 = time.perf_counter()
                    with mlflow.start_span(name=tc.function.name, span_type=SpanType.TOOL) as span:
                        span.set_inputs({"arguments": tc.function.arguments})
                        execution = await self._tools.execute(tc.function.name, tc.function.arguments)
                        span.set_outputs({"output": _cut(execution.output), "success": execution.success})
                    executions.append(execution)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id,
                         "content": _json.dumps(execution.model_dump())}
                    )
                    self._add(
                        trace, iteration, "tool_call",
                        tool=tc.function.name, args=execution.arguments,
                        result=_cut(execution.output), success=execution.success,
                        reasoning=reasoning, latency_s=round(llm_latency + time.perf_counter() - t1, 3),
                    )
                continue

            # ---- model stopped calling tools: this is the draft answer -------------
            draft = completion.message.content or ""
            self._add(
                trace, iteration, "draft_answer", result=_cut(draft),
                reasoning="Model returned no tool call -> loop proposes to stop.",
                latency_s=round(llm_latency, 3),
            )
            await self._verify_and_finish(trace, messages, completion, draft, executions, iteration, active_model)
            return

        trace.termination_reason = MAX_ITER

    async def _verify_and_finish(
        self, trace: QueryTrace, messages: list[dict[str, Any]], completion: Any,
        draft: str, executions: list[Any], iteration: int, model: str | None,
    ) -> None:
        # W16 passed ``completion.parsed`` which is always None on tool-enabled turns, so
        # the verifier saw an empty draft. ``pass_draft_to_verifier`` feeds it the draft.
        probe = (
            SimpleNamespace(parsed=SimpleNamespace(answer=draft))
            if self._cfg.pass_draft_to_verifier else completion
        )
        t0 = time.perf_counter()
        with mlflow.start_span(name="verification_decision", span_type=SpanType.CHAT_MODEL) as span:
            decision = await self._evaluate_verification_need(messages, probe, model)
            span.set_outputs({k: v for k, v in decision.items() if k != "used_fallback"})
        self._add(
            trace, iteration, "verification_decision",
            args={k: decision.get(k) for k in ("needs_verification", "confidence", "suggested_queries")},
            reasoning=str(decision.get("reason")), latency_s=round(time.perf_counter() - t0, 3),
        )

        verified_any = False
        if decision.get("needs_verification"):
            if self._max_verification_steps <= 0:
                trace.termination_reason = VERIFY_BUDGET
            else:
                t1 = time.perf_counter()
                result = await self._perform_verification(messages, decision, executions, 0, model)
                trace.verification_rounds = result["steps"]
                messages.extend(result["additional_messages"])
                n = max(1, len(result["additional_executions"]))
                for ex in result["additional_executions"]:
                    with mlflow.start_span(name=ex.name, span_type=SpanType.TOOL) as span:
                        span.set_inputs({"arguments": ex.arguments})
                        span.set_outputs({"output": _cut(ex.output), "success": ex.success})
                    self._add(
                        trace, iteration, "verification_search", tool=ex.name, args=ex.arguments,
                        result=_cut(ex.output), success=ex.success,
                        reasoning=f"verification: {decision.get('reason')}",
                        latency_s=round((time.perf_counter() - t1) / n, 3),
                    )
                executions.extend(result["additional_executions"])
                verified_any = bool(result["additional_executions"])

        t2 = time.perf_counter()
        with mlflow.start_span(name="final_answer", span_type=SpanType.CHAT_MODEL) as span:
            final = await self._llm.complete(messages, response_model=AssistantOutput, model=model)
            span.set_outputs({"answer": _cut(final.parsed.answer if final.parsed else None)})
        trace.model = final.model
        if final.parsed is None:
            raise LLMError("Model did not return a structured final answer")
        trace.answer = final.parsed.answer
        self._add(
            trace, iteration, "final_answer", result=_cut(final.parsed.answer),
            reasoning=f"confidence={final.parsed.confidence}", latency_s=round(time.perf_counter() - t2, 3),
        )
        if not trace.termination_reason:
            trace.termination_reason = SUCCESS_VERIFIED if verified_any else SUCCESS_DIRECT
