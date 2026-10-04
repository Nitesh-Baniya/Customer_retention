from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.tools.registry import RegisteredTool


class MonidDiscoverInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=2,
        max_length=500,
        description="The external information or API capability to search for.",
    )


class MonidInspectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=100)
    endpoint: str = Field(min_length=1, max_length=500)


class MonidRunInput(MonidInspectInput):
    input: dict[str, Any] = Field(
        description="Input object that exactly follows the inspected endpoint schema."
    )


class MonidService:
    """Call Monid's discover, inspect, and run API sequence."""

    MAX_RESPONSE_CHARACTERS = 100_000

    def __init__(self, settings: Settings) -> None:
        if settings.monid_api_key is None:
            raise ValueError("MONID_API_KEY is required to use Monid tools")

        self._api_key = settings.monid_api_key.get_secret_value()
        self._base_url = str(settings.monid_base_url).rstrip("/")
        self._timeout = settings.monid_timeout_seconds

    async def discover(self, query: str) -> str:
        return await self._post("/v1/discover", {"query": query})

    async def inspect(self, provider: str, endpoint: str) -> str:
        return await self._post(
            "/v1/inspect",
            {"provider": provider, "endpoint": endpoint},
        )

    async def run(
        self,
        provider: str,
        endpoint: str,
        endpoint_input: dict[str, Any],
    ) -> str:
        return await self._post(
            "/v1/run",
            {
                "provider": provider,
                "endpoint": endpoint,
                "input": endpoint_input,
            },
        )

    async def _post(self, path: str, payload: dict[str, object]) -> str:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            ) as client:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                result = json.dumps(response.json())
                if len(result) > self.MAX_RESPONSE_CHARACTERS:
                    return result[: self.MAX_RESPONSE_CHARACTERS] + "... [truncated]"
                return result
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise ValueError(
                f"Monid returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise ValueError("Monid is currently unavailable") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Monid returned an invalid JSON response") from exc


def create_monid_tools(settings: Settings) -> list[RegisteredTool]:
    if settings.monid_api_key is None:
        return []

    service = MonidService(settings)

    async def discover(input_data: BaseModel) -> str:
        tool_input = MonidDiscoverInput.model_validate(input_data.model_dump())
        return await service.discover(tool_input.query)

    async def inspect(input_data: BaseModel) -> str:
        tool_input = MonidInspectInput.model_validate(input_data.model_dump())
        return await service.inspect(tool_input.provider, tool_input.endpoint)

    async def run(input_data: BaseModel) -> str:
        tool_input = MonidRunInput.model_validate(input_data.model_dump())
        return await service.run(
            tool_input.provider,
            tool_input.endpoint,
            tool_input.input,
        )

    return [
        RegisteredTool(
            name="monid_discover",
            description=(
                "Find a read-only external API when current or specialized "
                "information is needed. Inspect a result before running it."
            ),
            input_model=MonidDiscoverInput,
            handler=discover,
        ),
        RegisteredTool(
            name="monid_inspect",
            description=(
                "Inspect a Monid endpoint returned by monid_discover to learn "
                "its purpose and required input schema before using monid_run."
            ),
            input_model=MonidInspectInput,
            handler=inspect,
        ),
        RegisteredTool(
            name="monid_run",
            description=(
                "Run a previously discovered and inspected read-only Monid "
                "endpoint. Never use endpoints that publish, purchase, or "
                "modify external data or accounts."
            ),
            input_model=MonidRunInput,
            handler=run,
            strict=False,
        ),
    ]
