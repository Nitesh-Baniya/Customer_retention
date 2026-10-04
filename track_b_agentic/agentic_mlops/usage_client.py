"""LLM client wrapper used for experiments.

Two things the W16 ``LLMClient`` does not do that MLOps needs:

1. **Token accounting** - every request's ``usage`` is accumulated so that cost can be
   logged to MLflow (the W16 harness only *estimated* tokens as ``len(text)/4``).
2. **``response_model=None`` support** - ``VerificationAssistantAgent`` calls
   ``llm.complete(..., response_model=None)`` to get raw JSON for its "does this answer
   need verification?" decision, but ``LLMClient._response_format(None)`` dereferences
   ``None.__name__`` and raises ``AttributeError`` (not caught by ``complete``). In a live
   run that means *every* query that reaches the verification step crashes. Here a ``None``
   response model falls back to plain ``json_object`` mode and returns the raw message.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, cast

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from app.llm.client import ChatMessageParam, LLMClient, LLMCompletion, LLMError

logger = logging.getLogger(__name__)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }


class UsageTrackingLLMClient(LLMClient):
    """``LLMClient`` that records token usage and tolerates ``response_model=None``."""

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.usage = Usage()

    def reset_usage(self) -> None:
        self.usage = Usage()

    def _record(self, completion: Any) -> None:
        self.usage.llm_calls += 1
        usage = getattr(completion, "usage", None)
        if usage is not None:
            self.usage.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage.completion_tokens += int(
                getattr(usage, "completion_tokens", 0) or 0
            )

    async def _request_with_tools(self, model, messages, tools):  # type: ignore[override]
        completion = await super()._request_with_tools(model, messages, tools)
        self._record(completion)
        return completion

    async def _request_structured_output(self, model, messages, response_model):  # type: ignore[override]
        if response_model is None:
            completion = await self._client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),
                response_format={"type": "json_object"},
                temperature=self._settings.llm_temperature,
                top_p=self._settings.llm_top_p,
                max_tokens=self._settings.llm_max_output_tokens,
            )
        else:
            completion = await super()._request_structured_output(
                model, messages, response_model
            )
        self._record(completion)
        return completion

    async def complete(  # type: ignore[override]
        self,
        messages: list[ChatMessageParam],
        *,
        response_model: type[BaseModel] | None,
        tools: Any = None,
        model: str | None = None,
    ) -> LLMCompletion[Any]:
        if response_model is not None:
            return await super().complete(
                messages, response_model=response_model, tools=tools, model=model
            )

        errors: list[str] = []
        for candidate in self._candidate_models(model):
            try:
                message = await self._request(candidate, messages, None, tools)  # type: ignore[arg-type]
                return LLMCompletion(
                    message=message,
                    parsed=None,
                    model=candidate,
                    used_fallback=candidate != self._models[0],
                )
            except (
                APIError,
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
                json.JSONDecodeError,
                ValidationError,
                ValueError,
            ) as exc:
                errors.append(f"{candidate}: {exc}")
                logger.warning("LLM attempt failed for model %s: %s", candidate, exc)
        raise LLMError("All configured models failed: " + " | ".join(errors))
