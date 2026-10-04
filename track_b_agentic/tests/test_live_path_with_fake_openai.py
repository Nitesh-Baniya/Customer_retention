"""Exercises the *live* code path (UsageTrackingLLMClient + TracedVerificationAgent) against a
fake OpenAI endpoint, so the W16 ``response_model=None`` crash fix and token accounting are
verified without network access."""
import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from agentic_mlops.config import RunConfig, load_prompt
from agentic_mlops.offline_stub import offline_registry
from agentic_mlops.traced_agent import SUCCESS_VERIFIED, TracedVerificationAgent
from agentic_mlops.usage_client import UsageTrackingLLMClient
from app.core.config import Settings


def _completion(message: dict, prompt=10, completion=5) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    })


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        if kw.get("tools"):
            has_tool_result = any(m["role"] == "tool" for m in kw["messages"])
            if not has_tool_result:
                return _completion({"role": "assistant", "content": "need weather", "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "get_current_weather", "arguments": json.dumps({"location": "Tokyo"})}}]})
            return _completion({"role": "assistant", "content": "Tokyo is 22C."})
        fmt = kw["response_format"]["type"]
        if fmt == "json_object":  # verification decision (response_model=None)
            return _completion({"role": "assistant", "content": json.dumps(
                {"needs_verification": True, "confidence": 0.5, "reason": "live data", "suggested_queries": ["tokyo weather"]})})
        return _completion({"role": "assistant", "content": json.dumps(
            {"answer": "Tokyo is 22C.", "cited_chunk_ids": [], "follow_up_questions": [], "confidence": "high"})})


@pytest.mark.asyncio
async def test_live_path_handles_none_response_model_and_counts_tokens():
    settings = Settings(hf_token="fake")
    llm = UsageTrackingLLMClient(settings)
    fake = FakeCompletions()
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=fake), close=lambda: None)  # type: ignore[assignment]
    cfg = RunConfig(prompt_version="v3", pass_draft_to_verifier=True)
    agent = TracedVerificationAgent(llm, offline_registry(), settings, cfg, load_prompt("v3"))

    tr = await agent.run_traced("weather_tokyo", "What is the current weather in Tokyo?")

    assert tr.error is None, tr.error
    assert tr.termination_reason == SUCCESS_VERIFIED
    assert [s.phase for s in tr.steps] == [
        "tool_call", "draft_answer", "verification_decision", "verification_search", "final_answer"]
    assert tr.usage["llm_calls"] == 4 and tr.usage["total_tokens"] == 60
    assert tr.steps[0].args == {"location": "Tokyo"} and tr.steps[0].reasoning == "need weather"
    # the verifier now sees the draft answer (W16 sent an empty string)
    verify_call = next(c for c in fake.calls if c.get("response_format", {}).get("type") == "json_object")
    assert verify_call["messages"][-2] == {"role": "assistant", "content": "Tokyo is 22C."}
