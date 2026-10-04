from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessage

from app.assistant.prompts import build_system_prompt
from app.core.config import Settings
from app.llm.client import ChatMessageParam, LLMClient, LLMError
from app.schemas.chat import (
    AssistantMetadata,
    AssistantOutput,
    ChatMessage,
    ToolExecution,
)
from app.tools.registry import ToolRegistry
from app.skills import SkillManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    """Decision model for cross-source verification."""
    needs_verification: bool
    confidence: float
    reason: str
    suggested_queries: list[str] | None = None


@dataclass(frozen=True, slots=True)
class VerificationAgentResult:
    output: AssistantOutput
    tools_used: list[ToolExecution]
    verification_steps: int
    model: str
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class VerificationAgentToolEvent:
    execution: ToolExecution


@dataclass(frozen=True, slots=True)
class VerificationAgentDeltaEvent:
    content: str


@dataclass(frozen=True, slots=True)
class VerificationAgentCompleteEvent:
    answer: str
    metadata: AssistantMetadata
    tools_used: list[ToolExecution]
    verification_steps: int
    model: str
    used_fallback: bool


VerificationAgentStreamEvent = (
    VerificationAgentToolEvent 
    | VerificationAgentDeltaEvent 
    | VerificationAgentCompleteEvent
)


class VerificationAssistantAgent:
    """
    Enhanced agent with cross-source verification capabilities.
    
    This agent can evaluate whether information needs verification from multiple sources
    and decide whether to perform additional searches before responding.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        settings: Settings,
        skills_dir: Path | None = None,
    ) -> None:
        self._llm = llm_client
        self._tools = tool_registry
        self._max_iterations = settings.llm_max_tool_iterations
        self._max_verification_steps = 3  # Maximum verification rounds
        self._skill_manager = SkillManager(skills_dir)

    async def run(
        self,
        question: str,
        *,
        history: list[ChatMessage] | None = None,
        context: str | None = None,
    ) -> VerificationAgentResult:
        """
        Run the agent with cross-source verification loop.
        
        The agent will:
        1. Process the question normally
        2. Evaluate if the answer needs verification
        3. If needed, perform additional searches
        4. Update the answer based on verification results
        """
        messages = self._build_messages(question, history, context)
        executions: list[ToolExecution] = []
        active_model: str | None = None
        used_fallback = False
        verification_steps = 0

        # Initial response generation
        for iteration in range(self._max_iterations):
            completion = await self._llm.complete(
                messages,
                response_model=AssistantOutput,
                tools=self._tools.schemas(),
                model=active_model,
            )
            used_fallback = used_fallback or completion.used_fallback
            if completion.used_fallback:
                active_model = completion.model

            if completion.message.tool_calls:
                await self._execute_tools(
                    completion.message,
                    messages,
                    executions,
                )
                continue

            # Check if verification is needed
            verification_decision = await self._evaluate_verification_need(
                messages,
                completion,
                model=active_model,
            )
            used_fallback = used_fallback or verification_decision.get("used_fallback", False)

            if not verification_decision["needs_verification"]:
                # No verification needed, return initial result
                final_completion = await self._llm.complete(
                    messages,
                    response_model=AssistantOutput,
                    model=active_model,
                )
                used_fallback = used_fallback or final_completion.used_fallback

                if final_completion.parsed is None:
                    raise LLMError("Model did not return a structured final answer")

                return VerificationAgentResult(
                    output=final_completion.parsed,
                    tools_used=executions,
                    verification_steps=verification_steps,
                    model=final_completion.model,
                    used_fallback=used_fallback,
                )

            # Perform verification loop
            verification_result = await self._perform_verification(
                messages,
                verification_decision,
                executions,
                verification_steps,
                active_model,
            )
            
            verification_steps = verification_result["steps"]
            used_fallback = used_fallback or verification_result["used_fallback"]
            executions.extend(verification_result["additional_executions"])
            
            # Update messages with verification context
            messages.extend(verification_result["additional_messages"])

            # Generate final answer with verification context
            final_completion = await self._llm.complete(
                messages,
                response_model=AssistantOutput,
                model=active_model,
            )
            used_fallback = used_fallback or final_completion.used_fallback

            if final_completion.parsed is None:
                raise LLMError("Model did not return a structured final answer")

            return VerificationAgentResult(
                output=final_completion.parsed,
                tools_used=executions,
                verification_steps=verification_steps,
                model=final_completion.model,
                used_fallback=used_fallback,
            )

        raise LLMError("Maximum tool-call iterations reached")

    async def stream(
        self,
        question: str,
        *,
        history: list[ChatMessage] | None = None,
        context: str | None = None,
    ) -> AsyncIterator[VerificationAgentStreamEvent]:
        """Stream the response with verification capabilities."""

        messages = self._build_messages(question, history, context)
        executions: list[ToolExecution] = []
        active_model: str | None = None
        used_fallback = False
        verification_steps = 0

        for iteration in range(self._max_iterations):
            completion = await self._llm.complete(
                messages,
                response_model=AssistantOutput,
                tools=self._tools.schemas(),
                model=active_model,
            )
            used_fallback = used_fallback or completion.used_fallback
            if completion.used_fallback:
                active_model = completion.model

            if completion.message.tool_calls:
                messages.append(completion.message.model_dump(exclude_none=True))

                for tool_call in completion.message.tool_calls:
                    if tool_call.type != "function":
                        continue

                    execution = await self._tools.execute(
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    executions.append(execution)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(execution.model_dump()),
                        }
                    )
                    yield VerificationAgentToolEvent(execution=execution)
                continue

            # Check if verification is needed
            verification_decision = await self._evaluate_verification_need(
                messages,
                completion,
                model=active_model,
            )
            used_fallback = used_fallback or verification_decision.get("used_fallback", False)

            if verification_decision["needs_verification"]:
                # Perform verification
                verification_result = await self._perform_verification(
                    messages,
                    verification_decision,
                    executions,
                    verification_steps,
                    active_model,
                )
                
                verification_steps = verification_result["steps"]
                used_fallback = used_fallback or verification_result["used_fallback"]
                executions.extend(verification_result["additional_executions"])
                messages.extend(verification_result["additional_messages"])
                
                # Emit verification tool events
                for execution in verification_result["additional_executions"]:
                    yield VerificationAgentToolEvent(execution=execution)

            # Stream final answer
            answer_parts: list[str] = []
            final_model = completion.model
            final_messages = self._build_stream_final_messages(messages)

            async for chunk in self._llm.stream_text(
                final_messages,
                model=active_model,
            ):
                final_model = chunk.model
                used_fallback = used_fallback or chunk.used_fallback
                if chunk.content:
                    answer_parts.append(chunk.content)
                    yield VerificationAgentDeltaEvent(content=chunk.content)

            answer = "".join(answer_parts).strip()
            if not answer:
                raise LLMError("Model returned an empty streamed answer")

            # Extract metadata
            metadata_messages = [
                *messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "Return only structured metadata for the answer above: "
                        "cited chunk IDs, up to three follow-up questions, and "
                        "confidence. Do not rewrite the answer."
                    ),
                },
            ]
            metadata_completion = await self._llm.complete(
                metadata_messages,
                response_model=AssistantMetadata,
                model=final_model,
            )
            used_fallback = used_fallback or metadata_completion.used_fallback
            if metadata_completion.parsed is None:
                raise LLMError("Model did not return structured stream metadata")

            yield VerificationAgentCompleteEvent(
                answer=answer,
                metadata=metadata_completion.parsed,
                tools_used=executions,
                verification_steps=verification_steps,
                model=metadata_completion.model,
                used_fallback=used_fallback,
            )
            return

        raise LLMError("Maximum tool-call iterations reached")

    async def _evaluate_verification_need(
        self,
        messages: list[ChatMessageParam],
        completion: Any,
        model: str | None,
    ) -> dict[str, Any]:
        """Evaluate if the current answer needs verification."""
        
        verification_prompt = [
            *messages,
            {"role": "assistant", "content": completion.parsed.answer if completion.parsed else ""},
            {
                "role": "user",
                "content": (
                    "Evaluate if this answer needs verification from additional sources. "
                    "Consider: 1) Factual claims that could be outdated, 2) Statistics or data, "
                    "3) Information that varies by source. Return JSON with: "
                    '{"needs_verification": boolean, "confidence": float (0-1), "reason": string, '
                    '"suggested_queries": [string] or null}.'
                ),
            },
        ]

        try:
            result = await self._llm.complete(
                verification_prompt,
                response_model=None,  # Get raw JSON
                model=model,
            )
            
            # Parse the decision
            content = result.message.content if result.message else ""
            decision = json.loads(content) if content else {
                "needs_verification": False,
                "confidence": 0.0,
                "reason": "Unable to evaluate",
                "suggested_queries": None
            }
            
            decision["used_fallback"] = result.used_fallback
            return decision
            
        except (json.JSONDecodeError, LLMError) as exc:
            logger.warning(f"Verification evaluation failed: {exc}")
            return {
                "needs_verification": False,
                "confidence": 0.0,
                "reason": f"Evaluation failed: {exc}",
                "suggested_queries": None,
                "used_fallback": False,
            }

    async def _perform_verification(
        self,
        messages: list[ChatMessageParam],
        decision: dict[str, Any],
        existing_executions: list[ToolExecution],
        current_step: int,
        model: str | None,
    ) -> dict[str, Any]:
        """Perform verification by searching additional sources."""
        
        if current_step >= self._max_verification_steps:
            return {
                "steps": current_step,
                "additional_executions": [],
                "additional_messages": [],
                "used_fallback": False,
            }

        additional_executions: list[ToolExecution] = []
        additional_messages: list[ChatMessageParam] = []
        used_fallback = False

        # Use suggested queries or generate them
        queries = decision.get("suggested_queries", [])
        if not queries:
            # Generate a verification query based on the original context
            verification_query = self._generate_verification_query(messages)
            queries = [verification_query] if verification_query else []

        # Perform web searches for verification
        for query in queries[:2]:  # Limit to 2 searches per verification step
            try:
                execution = await self._tools.execute(
                    "web_search",
                    json.dumps({"query": query}),
                )
                additional_executions.append(execution)
                
                additional_messages.append({
                    "role": "assistant",
                    "content": f"Performing verification search: {query}",
                })
                additional_messages.append({
                    "role": "tool",
                    "tool_call_id": f"verify_{len(additional_executions)}",
                    "content": json.dumps(execution.model_dump()),
                })
                
            except Exception as exc:
                logger.warning(f"Verification search failed: {exc}")

        # Add a summary message about verification
        additional_messages.append({
            "role": "system",
            "content": (
                f"Verification step {current_step + 1} completed. "
                f"Found {len(additional_executions)} additional sources. "
                "Consider this information when formulating your final answer."
            ),
        })

        return {
            "steps": current_step + 1,
            "additional_executions": additional_executions,
            "additional_messages": additional_messages,
            "used_fallback": used_fallback,
        }

    def _generate_verification_query(self, messages: list[ChatMessageParam]) -> str:
        """Generate a verification query from the conversation context."""
        
        # Extract the original user question
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if content and not content.startswith("Return only structured"):
                    return f"verify: {content[:100]}"
        
        return "verify current information"

    @staticmethod
    def _build_stream_final_messages(
        messages: list[ChatMessageParam],
    ) -> list[ChatMessageParam]:
        """Convert tool protocol messages into context for the prose pass."""

        final_messages: list[ChatMessageParam] = []
        tool_results: list[str] = []

        for message in messages:
            if message.get("role") == "tool":
                content = message.get("content")
                if content:
                    tool_results.append(str(content))
                continue

            if message.get("role") == "assistant" and message.get("tool_calls"):
                continue

            final_messages.append(message)

        final_messages.append(
            {
                "role": "user",
                "content": (
                    "Answer the original question now using the available context "
                    "and tool results below. Include any verification information gathered. "
                    "Do not call any tools.\n\n"
                    f"Tool results:\n{json.dumps(tool_results)}"
                ),
            }
        )
        return final_messages

    def _build_messages(
        self,
        question: str,
        history: list[ChatMessage] | None,
        context: str | None,
    ) -> list[ChatMessageParam]:
        # Apply progressive disclosure through Skills
        skill_context = self._skill_manager.get_context_augmentation(question)
        enhanced_context = f"{context}\n\n{skill_context}" if context else skill_context
        
        messages: list[ChatMessageParam] = [
            {"role": "system", "content": build_system_prompt(enhanced_context)}
        ]
        messages.extend(message.model_dump() for message in history or [])
        messages.append({"role": "user", "content": question})
        return messages

    async def _execute_tools(
        self,
        assistant_message: ChatCompletionMessage,
        messages: list[ChatMessageParam],
        executions: list[ToolExecution],
    ) -> None:
        messages.append(assistant_message.model_dump(exclude_none=True))

        for tool_call in assistant_message.tool_calls or []:
            if tool_call.type != "function":
                continue

            execution = await self._tools.execute(
                tool_call.function.name,
                tool_call.function.arguments,
            )
            executions.append(execution)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(execution.model_dump()),
                }
            )