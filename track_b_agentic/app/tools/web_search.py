from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import RegisteredTool

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=2,
        max_length=200,
        description="Search query to find information on the web.",
    )


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchService:
    """Simulated web search service for cross-source verification."""

    async def search(self, query: str) -> dict[str, object]:
        """Perform a web search and return results."""
        
        # For this implementation, we'll use a simulated search
        # In production, this would connect to a real search API
        simulated_results = self._simulated_search(query)
        
        return {
            "query": query,
            # Track B fix: SearchResult is a plain dataclass (W16 called .model_dump(), which raised).
            "results": [asdict(result) for result in simulated_results],
            "total_results": len(simulated_results),
        }

    def _simulated_search(self, query: str) -> list[SearchResult]:
        """Simulate search results for demonstration purposes."""
        
        # This is a simplified simulation - in production, use a real search API
        query_lower = query.lower()
        
        # Generate some contextually relevant simulated results
        results = []
        
        if "weather" in query_lower or "temperature" in query_lower:
            results.append(SearchResult(
                title="Weather Information Service",
                url="https://example.com/weather",
                snippet="Current weather conditions and forecasts for locations worldwide."
            ))
        elif "calculator" in query_lower or "math" in query_lower:
            results.append(SearchResult(
                title="Mathematical Calculator Guide",
                url="https://example.com/math",
                snippet="Comprehensive guide to mathematical calculations and formulas."
            ))
        elif "time" in query_lower or "date" in query_lower:
            results.append(SearchResult(
                title="World Time Service",
                url="https://example.com/time",
                snippet="Current time and date information for timezones around the world."
            ))
        else:
            results.append(SearchResult(
                title="General Information Database",
                url="https://example.com/info",
                snippet=f"Information related to: {query}"
            ))
        
        return results


async def web_search(input_data: BaseModel) -> str:
    search_input = WebSearchInput.model_validate(input_data.model_dump())
    result = await WebSearchService().search(search_input.query)
    return json.dumps(result)


def create_web_search_tool() -> RegisteredTool:
    return RegisteredTool(
        name="web_search",
        description="Search the web for information to verify facts or find additional sources.",
        input_model=WebSearchInput,
        handler=web_search,
    )
