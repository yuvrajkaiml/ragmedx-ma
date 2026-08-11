import os
import logging
from typing import Dict, Any, Optional
from jinja2 import Template

logger = logging.getLogger(__name__)

# Default prompt templates fallback directory
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts", "v1")


DEFAULT_PROMPTS: Dict[str, str] = {
    "query_understanding": """You are Agent 1 (Query Understanding Agent) in the RAGMedX-MA medical decision support pipeline.
Analyze the following user medical question and output a structured JSON evaluation.

User Question: {{ question }}

{% if specialty_hint %}
Specialty Hint: {{ specialty_hint }}
{% endif %}

Instructions:
1. Detect primary medical specialty (e.g., Cardiology, Oncology, Neurology, General Practice).
2. Classify intent (e.g., diagnosis, treatment, pharmacology, prognosis).
3. Resolve medical abbreviations to full terms.
4. Expand key terms with UMLS/MeSH standard terminology.
5. Provide an optimized search query string suitable for PubMed/BM25 retrieval.

Output JSON Format:
{
    "original_query": "{{ question }}",
    "processed_query": "Expanded query with resolved medical terms",
    "detected_intent": "treatment",
    "detected_specialty": "Cardiology",
    "expanded_terms": {"term": ["synonym1", "synonym2"]},
    "resolved_abbreviations": {"CAD": "Coronary Artery Disease"},
    "optimized_search_query": "coronary artery disease management evidence"
}
""",

    "generator": """You are Agent 4 (Generator Agent) in the RAGMedX-MA medical decision support pipeline.
Generate an evidence-based medical answer to the user's question using ONLY the provided evidence passages.

User Question: {{ question }}

Retrieved Evidence Snippets:
{% for ev in evidence %}
[{{ ev.citation_marker }}] {{ ev.title }} ({{ ev.journal }}, {{ ev.publication_year }})
{{ ev.cited_text }}
{% endfor %}

Instructions:
1. Synthesize a comprehensive, clinically accurate answer strictly supported by the provided evidence.
2. Cite all facts and claims using inline citation markers like [1], [2] matching the evidence provided.
3. Do NOT make claims that are not supported by the evidence passages.

Draft Answer:
""",

    "critic": """You are Agent 5 (Critic Agent) in the RAGMedX-MA peer-review pipeline.
Review the draft answer generated for the user question against the provided evidence passages.

User Question: {{ question }}

Draft Answer:
{{ draft_answer }}

Retrieved Evidence Snippets:
{% for ev in evidence %}
[{{ ev.citation_marker }}] {{ ev.cited_text }}
{% endfor %}

Evaluate:
1. Factual consistency (0.0 to 1.0)
2. Medical correctness (0.0 to 1.0)
3. Evidence alignment (0.0 to 1.0)
4. Are there hallucinated or unsupported claims?
5. Select verdict: 'verified', 'revised', 'rejected', or 'needs_revision'.

Output JSON Format:
{
    "verdict": "verified",
    "factual_consistency_score": 0.95,
    "medical_correctness_score": 0.90,
    "evidence_alignment_score": 0.92,
    "hallucination_detected": false,
    "unsupported_claims": [],
    "critic_feedback": "The answer accurately reflects the cited sources.",
    "suggested_revisions": [],
    "review_round": {{ review_round|default(1) }}
}
""",

    "confidence": """You are Agent 6 (Confidence Calibration Agent) in the RAGMedX-MA pipeline.
Calculate the composite confidence score for the candidate answer based on:
1. Retrieval Confidence (precision & similarity of evidence)
2. Evidence Quality Score (GRADE level, publication temporal score)
3. Answer Consistency Score (NLI factuality & critic evaluation)
4. Inter-Source Agreement (consensus across evidence sources)

Evaluation Metrics:
- Retrieval Confidence: {{ retrieval_confidence }}
- Evidence Quality Score: {{ evidence_quality_score }}
- Answer Consistency Score: {{ answer_consistency_score }}
- Inter Source Agreement: {{ inter_source_agreement }}

Compute:
composite_confidence = (0.25 * retrieval_confidence) + (0.30 * evidence_quality_score) + (0.25 * answer_consistency_score) + (0.20 * inter_source_agreement)
Passes Threshold: true if composite_confidence >= 0.65 else false

Output JSON Format:
{
    "retrieval_confidence": {{ retrieval_confidence }},
    "evidence_quality_score": {{ evidence_quality_score }},
    "answer_consistency_score": {{ answer_consistency_score }},
    "inter_source_agreement": {{ inter_source_agreement }},
    "composite_confidence": {{ composite_confidence }},
    "calibrated_confidence": {{ composite_confidence }},
    "confidence_category": "{{ 'high' if composite_confidence >= 0.80 else ('moderate' if composite_confidence >= 0.65 else 'low') }}",
    "passes_threshold": {{ 'true' if composite_confidence >= 0.65 else 'false' }},
    "threshold_used": 0.65
}
""",

    "abstention": """You are Agent 7 (Abstention Decision Agent) in the RAGMedX-MA pipeline.
The system confidence score for answering the query is below the required safety threshold (tau = 0.65).
Formulate a safe, transparent abstention response explaining why the system is declining to provide a definitive answer.

User Question: {{ question }}
Composite Confidence Score: {{ confidence_score }}
Reason for Low Confidence: {{ abstention_reason }}

Formulate a response explaining the insufficient evidence and suggest next steps or alternative queries.
"""
}


def load_prompt_template(agent_key: str) -> str:
    """
    Load prompt template from file if available, otherwise return default inline template.
    """
    file_path = os.path.join(PROMPTS_DIR, f"{agent_key}.txt")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content and not content.startswith("# Prompt Template Placeholder"):
                    return content
        except Exception as e:
            logger.warning(f"Failed to load prompt file {file_path}: {e}")

    return DEFAULT_PROMPTS.get(agent_key, "")


def render_prompt(agent_key: str, context: Dict[str, Any]) -> str:
    """
    Render prompt template using Jinja2 with provided context.
    """
    raw_template = load_prompt_template(agent_key)
    if not raw_template:
        raise ValueError(f"No prompt template found for agent key: {agent_key}")

    template = Template(raw_template)
    return template.render(**context)
