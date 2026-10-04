"""Fixed query set (the 8 W16 harness cases) + golden references for regression testing.

Differences from the W16 ``tests/evaluation_harness.py`` cases:
* every case has a stable ``id`` (needed to join responses to golden references);
* the time case expects ``current_utc_time`` - the registered tool name. The W16 harness
  expected ``current_time``, which no tool is called, so that case could never pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GOLDEN_PATH = DATA_DIR / "golden.jsonl"


@dataclass(frozen=True)
class TestCase:
    __test__ = False  # not a pytest class

    id: str
    query: str
    expected_tools: tuple[str, ...]
    expected_verification: bool
    description: str
    min_tool_calls: int = 1  # one call per named entity (e.g. 3 cities -> 3)


TEST_CASES: tuple[TestCase, ...] = (
    TestCase("weather_tokyo", "What is the current weather in Tokyo?",
             ("get_current_weather",), True, "Single-city live weather"),
    TestCase("calc_arith", "Calculate 25 * 17 + 43",
             ("calculator",), False, "Arithmetic - tool, no verification"),
    TestCase("time_utc", "What time is it now in UTC?",
             ("current_utc_time",), False, "Live time - tool, no verification"),
    TestCase("weather_compare", "Compare the weather in New York, London, and Tokyo",
             ("get_current_weather",), True, "Multi-entity: needs one call per city", 3),
    TestCase("population_compare",
             "What is the population of Brazil and how does it compare to India?",
             ("web_search",), True, "Statistics comparison", 2),
    TestCase("calc_circle", "Calculate the area of a circle with radius 5",
             ("calculator",), False, "Geometry via calculator"),
    TestCase("renewables_latest", "What are the latest developments in renewable energy?",
             ("web_search",), True, "Time-sensitive topic"),
    TestCase("weather_paris", "What is the current temperature in Paris and is it raining?",
             ("get_current_weather",), True, "Weather with a condition check"),
)


def load_golden(path: Path = GOLDEN_PATH) -> dict[str, str]:
    """Return {case_id: approved reference answer}."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row["reference"]
    return out
