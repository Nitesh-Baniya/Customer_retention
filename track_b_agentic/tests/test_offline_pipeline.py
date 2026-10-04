import asyncio

from agentic_mlops.config import EXPERIMENTS, load_prompt
from agentic_mlops.offline_stub import ScriptedLLMClient, offline_registry
from agentic_mlops.scoring import aggregate
from agentic_mlops.testset import TEST_CASES, load_golden
from agentic_mlops.traced_agent import TracedVerificationAgent
from app.core.config import Settings


def _run(cfg):
    prompt = load_prompt(cfg.prompt_version)
    s = Settings(hf_token="x")
    agent = TracedVerificationAgent(ScriptedLLMClient(prompt), offline_registry(), s, cfg, prompt)

    async def go():
        return [await agent.run_traced(c.id, c.query) for c in TEST_CASES]
    return asyncio.run(go())


def test_golden_covers_every_case():
    assert set(load_golden()) == {c.id for c in TEST_CASES}


def test_web_search_tool_no_longer_crashes():
    from app.tools.web_search import WebSearchService
    out = asyncio.run(WebSearchService().search("anything"))
    assert out["total_results"] == 1


def test_every_trace_has_termination_reason_and_ordered_steps():
    for cfg in EXPERIMENTS:
        for tr in _run(cfg):
            assert tr.termination_reason, tr.case_id
            assert [s.step for s in tr.steps] == list(range(1, len(tr.steps) + 1))


def test_v3_is_not_worse_than_v1_on_strict_completion():
    m = {c.prompt_version: aggregate(list(TEST_CASES), _run(c))[0] for c in EXPERIMENTS}
    assert m["v3"]["strict_completion_rate"] >= m["v1"]["strict_completion_rate"]
