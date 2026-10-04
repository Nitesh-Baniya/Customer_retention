from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Skill:
    """Represents a capability that can be progressively disclosed."""
    name: str
    summary: str  # Concise description for initial context
    full_instructions: str  # Detailed instructions loaded when relevant
    trigger_keywords: list[str]  # Keywords that indicate this skill is relevant


class SkillInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(
        description="Name of the skill to load full instructions for.",
    )


class SkillManager:
    """
    Manages progressive disclosure of skill instructions.
    
    This implements the context engineering technique of progressive disclosure
    through Skills, where concise descriptions are loaded initially and full
    instructions are provided only when the model determines the skill is relevant.
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self._loaded_full_instructions: set[str] = set()
        
        if skills_dir and skills_dir.exists():
            self._load_skills_from_directory(skills_dir)
        else:
            self._load_default_skills()

    def _load_default_skills(self) -> None:
        """Load default skills for the system."""
        
        self._skills["verification"] = Skill(
            name="verification",
            summary="Cross-source verification to validate factual claims from multiple sources.",
            full_instructions=(
                "When answering questions that involve factual claims, statistics, or information "
                "that may vary by source, use the verification process:\n"
                "1. Generate an initial answer based on available context\n"
                "2. Evaluate if the answer contains claims that need verification\n"
                "3. If verification is needed, perform web searches to cross-check information\n"
                "4. Compare results from multiple sources\n"
                "5. Update the answer to reflect verified information and note any discrepancies\n"
                "6. Be transparent about the verification process and confidence levels\n\n"
                "Use the web_search tool for verification. Limit verification to 2-3 rounds maximum."
            ),
            trigger_keywords=["verify", "check", "confirm", "validate", "fact", "source", "accurate"],
        )
        
        self._skills["research"] = Skill(
            name="research",
            summary="Multi-source research for comprehensive information gathering.",
            full_instructions=(
                "For complex research tasks:\n"
                "1. Break down the research question into key components\n"
                "2. Identify different types of sources needed (academic, news, data, etc.)\n"
                "3. Perform targeted searches for each component\n"
                "4. Synthesize information from multiple sources\n"
                "5. Identify agreements and disagreements between sources\n"
                "6. Provide a comprehensive answer with source attribution\n"
                "7. Note any gaps or limitations in the available information"
            ),
            trigger_keywords=["research", "study", "investigate", "analyze", "comprehensive", "multiple sources"],
        )
        
        self._skills["comparison"] = Skill(
            name="comparison",
            summary="Systematic comparison of multiple options or alternatives.",
            full_instructions=(
                "When comparing multiple options:\n"
                "1. Identify clear comparison criteria relevant to the user's needs\n"
                "2. Gather information about each option against those criteria\n"
                "3. Present results in a structured format (table or list)\n"
                "4. Highlight key differences and trade-offs\n"
                "5. Provide a recommendation based on the analysis\n"
                "6. Explain the reasoning behind the recommendation\n"
                "7. Note any assumptions or limitations in the comparison"
            ),
            trigger_keywords=["compare", "versus", "vs", "difference", "better", "best", "alternative"],
        )

    def _load_skills_from_directory(self, skills_dir: Path) -> None:
        """Load skills from SKILL.md files in a directory."""
        
        for skill_file in skills_dir.glob("*/SKILL.md"):
            try:
                skill_name = skill_file.parent.name
                content = skill_file.read_text()
                
                # Parse the skill file (expected format: summary first, then full instructions)
                sections = content.split("\n\n", 1)
                summary = sections[0].strip()
                full_instructions = sections[1].strip() if len(sections) > 1 else summary
                
                # Extract trigger keywords from the summary
                trigger_keywords = self._extract_keywords(summary)
                
                self._skills[skill_name] = Skill(
                    name=skill_name,
                    summary=summary,
                    full_instructions=full_instructions,
                    trigger_keywords=trigger_keywords,
                )
                
                logger.info(f"Loaded skill: {skill_name}")
                
            except Exception as exc:
                logger.warning(f"Failed to load skill from {skill_file}: {exc}")

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract potential trigger keywords from skill description."""
        
        # Simple keyword extraction - in production, use more sophisticated NLP
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        words = text.lower().split()
        keywords = [word.strip(".,!?") for word in words if word.lower() not in common_words and len(word) > 3]
        
        return keywords[:10]  # Limit to top 10 keywords

    def get_skill_summaries(self) -> str:
        """Get concise summaries of all available skills for initial context."""
        
        summaries = []
        for skill in self._skills.values():
            summaries.append(f"- {skill.name}: {skill.summary}")
        
        return "\n".join(summaries)

    def get_full_skill_instructions(self, skill_name: str) -> str | None:
        """Get full instructions for a specific skill."""
        
        skill = self._skills.get(skill_name)
        if skill:
            self._loaded_full_instructions.add(skill_name)
            return skill.full_instructions
        
        return None

    def find_relevant_skills(self, query: str) -> list[str]:
        """Find skills that are relevant to a given query based on trigger keywords."""
        
        query_lower = query.lower()
        relevant_skills = []
        
        for skill_name, skill in self._skills.items():
            # Check if any trigger keywords appear in the query
            if any(keyword in query_lower for keyword in skill.trigger_keywords):
                relevant_skills.append(skill_name)
        
        return relevant_skills

    def should_load_full_instructions(self, query: str, skill_name: str) -> bool:
        """Determine if full instructions should be loaded for a skill."""
        
        # Load full instructions if the skill is relevant and not already loaded
        if skill_name in self._loaded_full_instructions:
            return False
        
        relevant_skills = self.find_relevant_skills(query)
        return skill_name in relevant_skills

    def get_context_augmentation(self, query: str) -> str:
        """
        Get context augmentation for a query.
        
        Returns concise summaries initially, and adds full instructions
        for relevant skills when needed.
        """
        
        relevant_skills = self.find_relevant_skills(query)
        
        if not relevant_skills:
            # No relevant skills, return just summaries
            return f"Available skills:\n{self.get_skill_summaries()}"
        
        # Add full instructions for relevant skills
        augmentations = [f"Available skills:\n{self.get_skill_summaries()}"]
        
        for skill_name in relevant_skills:
            full_instructions = self.get_full_skill_instructions(skill_name)
            if full_instructions:
                augmentations.append(f"\n\n--- Full instructions for {skill_name} ---\n{full_instructions}")
        
        return "\n".join(augmentations)