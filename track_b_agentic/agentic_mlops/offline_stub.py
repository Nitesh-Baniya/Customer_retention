"""Offline stand-ins so the whole MLOps pipeline can be smoke-tested with NO network/API keys.

*** ScriptedLLMClient is NOT a language model. ***
It is a small rule-based policy whose behaviour depends on a few phrases in the system
prompt. It exists only to exercise the plumbing (traces -> MLflow -> Evidently -> gate ->
Airflow) in CI. Never report its numbers as experiment results: runs made with it are
tagged ``mode=offline_stub`` and logged to a separate MLflow experiment.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import BaseModel

from app.llm.client import LLMCompletion
from app.schemas.chat import AssistantOutput
from app.tools.calculator import create_calculator_tool
from app.tools.current_time import create_current_time_tool
from app.tools.registry import RegisteredTool, ToolRegistry
from app.tools.weather import WeatherInput
from app.tools.web_search import create_web_search_tool

from agentic_mlops.usage_client import Usage

CITIES = ("New York", "London", "Tokyo", "Paris")
_FAKE_WEATHER = {
    "New York": (18.4, "Partly cloudy"), "London": (14.2, "Slight rain"),
    "Tokyo": (22.1, "Mainly clear"), "Paris": (16.0, "Overcast"),
}


def _offline_weather_tool() -> RegisteredTool:
    def handler(inp: BaseModel) -> str:
        loc = WeatherInput.model_validate(inp.model_dump()).location
        city = next((c for c in CITIES if c.lower() in loc.lower()), loc)
        temp, cond = _FAKE_WEATHER.get(city, (20.0, "Clear sky"))
        return json.dumps({"location": city, "condition": cond, "temperature": temp,
                           "temperature_unit": "°C", "relative_humidity_percent": 60,
                           "wind_speed": 11.0, "wind_speed_unit": "km/h"})

    return RegisteredTool("get_current_weather", "Get the current weather for a city or place.",
                          WeatherInput, handler)


def offline_registry() -> ToolRegistry:
    """Real calculator/time/web_search (all local) + canned weather instead of Open-Meteo."""
    return ToolRegistry([create_calculator_tool(), create_current_time_tool(),
                         _offline_weather_tool(), create_web_search_tool()])


class ScriptedLLMClient:
    def __init__(self, system_prompt: str) -> None:
        self.usage = Usage()
        self._models = ["scripted-offline"]
        self.routing = "ALWAYS call" in system_prompt
        self.completeness = "ONE tool call PER entity" in system_prompt

    def reset_usage(self) -> None:
        self.usage = Usage()

    async def close(self) -> None:  # parity with LLMClient
        return None

    # ---- helpers ------------------------------------------------------------------
    @staticmethod
    def _query(messages: list[dict[str, Any]]) -> str:
        return next(m["content"] for m in messages if m["role"] == "user")

    def _plan(self, q: str) -> list[tuple[str, dict[str, Any]]]:
        ql = q.lower()
        if "calculate" in ql:
            expr = "25 * 17 + 43" if "25 * 17" in q else "pi * 5 ** 2"
            return [("calculator", {"expression": expr})]
        if "weather" in ql or "temperature" in ql:
            found = sorted((ql.find(c.lower()), c) for c in CITIES if c.lower() in ql)
            cities = [c for _, c in found] or ["Tokyo"]
            cities = cities if self.completeness else cities[:1]
            return [("get_current_weather", {"location": c}) for c in cities]
        if "time" in ql and "utc" in ql:
            return [("current_utc_time", {})]
        if self.routing and ("population" in ql or "latest" in ql):
            queries = ["Brazil population", "India population"] if "population" in ql and self.completeness \
                else [q[:80]]
            return [("web_search", {"query": x}) for x in queries]
        return []

    @staticmethod
    def _category(q: str) -> str:
        ql = q.lower()
        if "weather" in ql or "temperature" in ql:
            return "weather"
        if "population" in ql:
            return "population"
        if "latest" in ql:
            return "news"
        return "stable"

    def _answer_text(self, q: str, tool_msgs: list[dict[str, Any]]) -> str:
        outs = [json.loads(m["content"]) for m in tool_msgs]
        parts: list[str] = []
        for o in outs:
            name, out = o["name"], o["output"]
            if name == "calculator":
                parts.append(f"The result of `{json.loads(out)['expression']}` is {json.loads(out)['result']}.")
            elif name == "get_current_weather":
                w = json.loads(out)
                parts.append(f"{w['location']}: {w['temperature']}{w['temperature_unit']}, {w['condition']}.")
            elif name == "current_utc_time":
                parts.append(f"The current UTC time is {json.loads(out)['utc_time']}.")
            elif name == "web_search":
                parts.append("Search result: " + json.loads(out)["results"][0]["snippet"])
        if parts:
            return " ".join(parts)
        cat = self._category(q)
        if cat == "population":
            return "Brazil has roughly 215 million people and India about 1.4 billion."
        if cat == "news":
            return "Renewable energy keeps growing, but I do not have specific recent developments."
        return "I can help with that."

    def _tick(self, messages: list[dict[str, Any]], out: str) -> None:
        self.usage.llm_calls += 1
        self.usage.prompt_tokens += len(json.dumps(messages, default=str)) // 4
        self.usage.completion_tokens += max(1, len(out) // 4)

    # ---- LLMClient interface ------------------------------------------------------
    async def complete(self, messages, *, response_model, tools=None, model=None):  # type: ignore[no-untyped-def]
        q = self._query(messages)
        tool_msgs = [m for m in messages if m["role"] == "tool" and not str(m.get("tool_call_id", "")).startswith("verify_")]
        verify_msgs = [m for m in messages if str(m.get("tool_call_id", "")).startswith("verify_")]
        all_tool = [m for m in messages if m["role"] == "tool"]

        if tools:  # ---- tool-selection turn
            plan = self._plan(q)
            idx = len(tool_msgs)
            if idx < len(plan):
                name, args = plan[idx]
                msg = ChatCompletionMessage(
                    role="assistant", content=f"I need `{name}` for this question.",
                    tool_calls=[ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}", type="function",
                        function=Function(name=name, arguments=json.dumps(args)))],
                )
                self._tick(messages, msg.content or "")
                return LLMCompletion(message=msg, parsed=None, model="scripted-offline", used_fallback=False)
            text = self._answer_text(q, tool_msgs)
            self._tick(messages, text)
            return LLMCompletion(message=ChatCompletionMessage(role="assistant", content=text),
                                 parsed=None, model="scripted-offline", used_fallback=False)

        if response_model is None:  # ---- verification decision (raw JSON)
            draft_visible = bool(str(messages[-2].get("content", "")).strip())
            already = bool(verify_msgs)
            needs = self._category(q) != "stable" and draft_visible and not already
            reason = ("No draft answer was provided to evaluate." if not draft_visible
                      else "Claim is time-varying; cross-check with a second source." if needs
                      else "Already verified or stable fact.")
            out = json.dumps({"needs_verification": needs, "confidence": 0.6, "reason": reason,
                              "suggested_queries": [q[:80]] if needs else None})
            self._tick(messages, out)
            return LLMCompletion(message=ChatCompletionMessage(role="assistant", content=out),
                                 parsed=None, model="scripted-offline", used_fallback=False)

        text = self._answer_text(q, all_tool)  # ---- final structured answer
        parsed = AssistantOutput(answer=text, cited_chunk_ids=[], follow_up_questions=[],
                                 confidence="medium" if all_tool else "low")
        out = parsed.model_dump_json()
        self._tick(messages, out)
        return LLMCompletion(message=ChatCompletionMessage(role="assistant", content=out),
                             parsed=parsed, model="scripted-offline", used_fallback=False)
