import logging
import os
from typing import Any, Dict, Optional
from jinja2 import Template

logger = logging.getLogger(__name__)

PROMPTS_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts")

DEFAULT_PROMPTS: Dict[str, str] = {
    "query_understanding": """You are the Query Understanding Agent in a medical question answering system.
Analyze the user's medical question:
{{ question }}
Identify:
1. Medical specialty
2. User intent
3. Important abbreviations
4. Expanded medical terminology
5. An optimized retrieval query
Return ONLY valid JSON:
{
    "medical_specialty": "",
    "intent": "",
    "expanded_query": "",
    "abbreviations": {},
    "confidence": 0.0
}""",
    "iterative_retrieval": """You are the Medical Gap Analysis Agent in a medical question answering system.
Question: {{ question }}
Retrieved evidence: {{ evidence_text }}
Determine if evidence is sufficient to safely answer.
Return ONLY valid JSON:
{
    "sufficient": true,
    "reasoning": "",
    "missing_information": [],
    "sub_query": null,
    "confidence": 0.0
}""",
    "generator": """You are the Medical Answer Generator in a medical question-answering system.
Answer the user's question using ONLY the provided evidence:
User question: {{ question }}
Evidence: {{ evidence_text }}
Return only the final grounded answer with citation markers.""",
    "critic": """You are the Medical Answer Critic in a medical question-answering system.
Question: {{ question }}
Generated answer: {{ answer }}
Evidence: {{ evidence_text }}
Evaluate factual consistency, citation support, and hallucination risk.
Return ONLY valid JSON:
{
    "passed": true,
    "factual_consistency": 0.0,
    "citation_support": 0.0,
    "hallucination_risk": 0.0,
    "issues": [],
    "revision_instructions": null,
    "confidence": 0.0
}""",
    "confidence": """You are the Confidence Calibration Agent.
Calculate 4-component composite confidence score.""",
    "abstention": """You are the Abstention Decision Agent.
Determine whether to PROCEED or ABSTAIN based on safety thresholds.""",
}


def load_prompt_template(agent_key: str, version: str = "v1") -> str:
    """
    Load prompt template from prompts/{version}/{agent_key}.txt if available,
    otherwise return fallback default template.
    """
    file_path = os.path.join(PROMPTS_BASE_DIR, version, f"{agent_key}.txt")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content and not content.startswith("# Prompt Template Placeholder"):
                    return content
        except Exception as e:
            logger.warning(f"Failed to load prompt file {file_path}: {e}")

    return DEFAULT_PROMPTS.get(agent_key, "")


def render_prompt(agent_key: str, context: Dict[str, Any], version: str = "v1") -> str:
    """
    Render prompt template using Jinja2 with provided context.
    """
    raw_template = load_prompt_template(agent_key, version=version)
    if not raw_template:
        raise ValueError(f"No prompt template found for agent key: {agent_key} (version {version})")

    template = Template(raw_template)
    return template.render(**context)
