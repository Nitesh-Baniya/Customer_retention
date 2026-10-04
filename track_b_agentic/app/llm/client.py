from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, ValidationError

from app.core.config import LLMBackend, Settings

logger = logging.getLogger(__name__)

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)
ChatMessageParam = dict[str, Any]


class LLMError(RuntimeError):
    """Raised when neither the primary nor fallback model can respond."""


@dataclass(frozen=True, slots=True)
class LLMCompletion(Generic[ResponseModelT]):
    message: ChatCompletionMessage
    parsed: ResponseModelT | None
    model: str
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class LLMTextChunk:
    content: str
    model: str
    used_fallback: bool
    is_complete: bool = False


@dataclass(frozen=True, slots=True)
class _BackendConfig:
    api_key: str
    base_url: str
    models: list[str]


class LLMClient:
    """OpenAI-compatible client for Hugging Face Router and local vLLM."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        backend = self._backend_config(settings)
        self._models = backend.models
        self._client = AsyncOpenAI(
            api_key=backend.api_key,
            base_url=backend.base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=1,
        )

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _backend_config(settings: Settings) -> _BackendConfig:
        if settings.llm_backend is LLMBackend.HUGGINGFACE:
            if settings.hf_token is None:  # Guard for static type narrowing.
                raise ValueError("HF_TOKEN is required for the Hugging Face backend")
            return _BackendConfig(
                api_key=settings.hf_token.get_secret_value(),
                base_url=str(settings.hf_base_url).rstrip("/"),
                models=[settings.hf_model, settings.hf_fallback_model],
            )

        return _BackendConfig(
            api_key=settings.vllm_api_key.get_secret_value(),
            base_url=str(settings.vllm_base_url).rstrip("/"),
            models=[settings.vllm_model],
        )

    async def complete(
        self,
        messages: list[ChatMessageParam],
        *,
        response_model: type[ResponseModelT],
        tools: list[ChatCompletionToolParam] | None = None,
        model: str | None = None,
    ) -> LLMCompletion[ResponseModelT]:
        """Try configured models until one returns tools or valid structured output."""

        candidate_models = self._candidate_models(model)
        errors: list[str] = []

        for candidate in candidate_models:
            try:
                # Step 1: Send the conversation and schema to this model.
                message = await self._request(
                    candidate,
                    messages,
                    response_model,
                    tools,
                )

                # Step 2: Validate content when this is a final-answer turn.
                parsed = self._parse_final_answer(
                    message,
                    response_model,
                    tools_enabled=bool(tools),
                )

                # Step 3: Return the SDK message and validated model output.
                return LLMCompletion(
                    message=message,
                    parsed=parsed,
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

    async def stream_text(
        self,
        messages: list[ChatMessageParam],
        *,
        model: str | None = None,
    ) -> AsyncIterator[LLMTextChunk]:
        """Stream plain text from the first available configured model."""

        errors: list[str] = []

        for candidate in self._candidate_models(model):
            received_content = False

            try:
                stream = await self._client.chat.completions.create(
                    model=candidate,
                    messages=cast(Any, messages),
                    temperature=self._settings.llm_temperature,
                    top_p=self._settings.llm_top_p,
                    max_tokens=self._settings.llm_max_output_tokens,
                    stream=True,
                )

                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if not content:
                        continue

                    received_content = True
                    yield LLMTextChunk(
                        content=content,
                        model=candidate,
                        used_fallback=candidate != self._models[0],
                    )

                yield LLMTextChunk(
                    content="",
                    model=candidate,
                    used_fallback=candidate != self._models[0],
                    is_complete=True,
                )
                return
            except (
                APIError,
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            ) as exc:
                if received_content:
                    raise LLMError(
                        f"Model stream interrupted after output began: {exc}"
                    ) from exc

                errors.append(f"{candidate}: {exc}")
                logger.warning("LLM stream failed for model %s: %s", candidate, exc)

        raise LLMError("All configured models failed: " + " | ".join(errors))

    def _candidate_models(self, requested_model: str | None) -> list[str]:
        return [requested_model] if requested_model else self._models

    async def _request(
        self,
        model: str,
        messages: list[ChatMessageParam],
        response_model: type[BaseModel],
        tools: list[ChatCompletionToolParam] | None,
    ) -> ChatCompletionMessage:
        if tools:
            completion = await self._request_with_tools(model, messages, tools)
        else:
            completion = await self._request_structured_output(
                model,
                messages,
                response_model,
            )

        return completion.choices[0].message

    async def _request_with_tools(
        self,
        model: str,
        messages: list[ChatMessageParam],
        tools: list[ChatCompletionToolParam],
    ) -> ChatCompletion:
        """Ask the model to either call a tool or continue without one."""

        return await self._client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            tools=tools,
            tool_choice="auto",
            temperature=self._settings.llm_temperature,
            top_p=self._settings.llm_top_p,
            max_tokens=self._settings.llm_max_output_tokens,
        )

    async def _request_structured_output(
        self,
        model: str,
        messages: list[ChatMessageParam],
        response_model: type[BaseModel],
    ) -> ChatCompletion:
        """Ask for JSON without sending tool fields to the provider."""

        return await self._client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            response_format=self._response_format(response_model),
            temperature=self._settings.llm_temperature,
            top_p=self._settings.llm_top_p,
            max_tokens=self._settings.llm_max_output_tokens,
        )

    def _parse_final_answer(
        self,
        message: ChatCompletionMessage,
        response_model: type[ResponseModelT],
        *,
        tools_enabled: bool,
    ) -> ResponseModelT | None:
        # Tool-selection turns cannot also use JSON mode on some providers.
        if tools_enabled:
            return None
        return self._parse_response(message.content, response_model)

    @staticmethod
    def _response_format(response_model: type[BaseModel]) -> Any:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": response_model.model_json_schema(),
            },
        }

    @staticmethod
    def _parse_response(
        content: str | None,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        if not content:
            raise ValueError("Model returned an empty response")
        return response_model.model_validate_json(content)
