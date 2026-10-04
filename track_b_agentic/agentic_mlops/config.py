"""Run configuration: everything we vary between MLflow runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@dataclass(frozen=True)
class RunConfig:
    prompt_version: str = "v1"
    model: str | None = None            # None -> Settings.hf_model (with W15 fallback)
    temperature: float = 0.2
    max_iterations: int = 5             # agentic tool loop (Settings.llm_max_tool_iterations)
    max_verification_rounds: int = 3    # W16 hard-coded this to 3
    pass_draft_to_verifier: bool = False  # W16 behaviour: verifier never sees the draft
    description: str = ""

    def run_name(self) -> str:
        return f"prompt_{self.prompt_version}"

    def as_params(self) -> dict[str, object]:
        d = asdict(self)
        d.pop("description")
        return d


def load_prompt(version: str) -> str:
    path = PROMPTS_DIR / f"prompt_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file for version {version!r}: {path}")
    return path.read_text(encoding="utf-8").strip()


# The three experiment configurations. Each one answers a specific failure that the
# traces of the previous one exposed (see README "Experiment Tracking Strategy").
EXPERIMENTS: tuple[RunConfig, ...] = (
    RunConfig(
        prompt_version="v1",
        description="W15/W16 baseline system prompt, W16 agent behaviour unchanged.",
    ),
    RunConfig(
        prompt_version="v2",
        description="v1 + explicit tool-routing rules (answers from memory instead of "
        "calling a tool).",
    ),
    RunConfig(
        prompt_version="v3",
        description="v2 + completeness/stop rules, and the verifier is shown the draft "
        "answer (it previously received an empty string).",
        pass_draft_to_verifier=True,
    ),
)
