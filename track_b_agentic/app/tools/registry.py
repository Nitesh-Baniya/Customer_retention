from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, ValidationError

from app.schemas.chat import ToolExecution

ToolHandler = Callable[[BaseModel], str | Awaitable[str]]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    strict: bool = True

    def as_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
                "strict": self.strict,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[RegisteredTool] | None = None) -> None:
        self._tools = {tool.name: tool for tool in tools or []}

    def schemas(self) -> list[ChatCompletionToolParam]:
        return [tool.as_chat_tool() for tool in self._tools.values()]

    async def execute(self, name: str, raw_arguments: str) -> ToolExecution:
        """Find, validate, and execute one model-requested tool call."""

        # Step 1: Reject tools that were not registered by the application.
        tool = self._tools.get(name)
        if tool is None:
            return self._failure(name, {}, f"Unknown tool: {name}")

        arguments: dict[str, object] = {}
        try:
            # Step 2: Parse JSON and validate the tool's input schema.
            arguments = self._parse_arguments(raw_arguments)
            validated = tool.input_model.model_validate(arguments)
            arguments = validated.model_dump()

            # Step 3: Run either a synchronous or asynchronous handler.
            output = await self._run_handler(tool, validated)

            # Step 4: Return a serializable result to the assistant loop.
            return ToolExecution(
                name=name,
                arguments=arguments,
                output=output,
                success=True,
            )
        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            ArithmeticError,
        ) as exc:
            return self._failure(name, arguments, f"Tool error: {exc}")

    @staticmethod
    def _parse_arguments(raw_arguments: str) -> dict[str, object]:
        decoded: object = json.loads(raw_arguments)
        if not isinstance(decoded, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return decoded

    @staticmethod
    async def _run_handler(tool: RegisteredTool, input_data: BaseModel) -> str:
        result = tool.handler(input_data)
        return await result if inspect.isawaitable(result) else result

    @staticmethod
    def _failure(
        name: str,
        arguments: dict[str, object],
        message: str,
    ) -> ToolExecution:
        return ToolExecution(
            name=name,
            arguments=arguments,
            output=message,
            success=False,
        )
