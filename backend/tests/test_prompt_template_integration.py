"""Agent prompt rendering must remain bound to versioned v1 template files."""

import pytest

from app.agents.critic import CriticAgent
from app.agents.evidence_reranking import EvidenceRerankingAgent
from app.agents.generator import GeneratorAgent
from app.agents.iterative_retrieval import IterativeRetrievalAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.llm.prompt_templates import load_prompt_template


class PromptRecordingLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt, **_):
        self.prompts.append(prompt)
        return self.response


EVIDENCE = [{
    "chunk_id": "chunk-1", "document_id": "doc-1", "title": "Source",
    "journal": "Journal", "publication_year": 2025,
    "cited_text": "A sufficiently detailed medical evidence span for this test.",
    "citation_marker": "[Doc 1]", "relevance_score": 0.9,
    "cross_encoder_score": 0.9,
}]


@pytest.mark.parametrize(
    ("agent_key", "agent_factory", "state", "response"),
    [
        ("query_understanding", QueryUnderstandingAgent,
         {"original_question": "What is hypertension?"},
         '{"medical_specialty":"cardiology","intent":"definition","expanded_query":"hypertension","abbreviations":{},"confidence":0.9}'),
        ("iterative_retrieval", IterativeRetrievalAgent,
         {"original_question": "What is hypertension?", "evidence": EVIDENCE},
         '{"sufficient":true,"reasoning":"enough","missing_information":[],"sub_query":null,"confidence":0.9}'),
        ("evidence_reranking", EvidenceRerankingAgent,
         {"original_question": "What is hypertension?", "evidence": EVIDENCE},
         '{"contradictory":false,"selected_markers":["[Doc 1]"],"reasoning":"relevant"}'),
        ("generator", GeneratorAgent,
         {"original_question": "What is hypertension?", "evidence": EVIDENCE},
         "Hypertension is high blood pressure [Doc 1]."),
        ("critic", CriticAgent,
         {"original_question": "What is hypertension?", "evidence": EVIDENCE,
          "answer": "Hypertension is high blood pressure [Doc 1]."},
         '{"passed":true,"factual_consistency":0.9,"citation_support":0.9,"hallucination_risk":0.1,"issues":[],"revision_instructions":null,"confidence":0.9}'),
    ],
)
async def test_llm_agents_render_their_versioned_template(agent_key, agent_factory, state, response):
    llm = PromptRecordingLLM(response)
    await agent_factory(llm_wrapper=llm).run(state)

    template = load_prompt_template(agent_key)
    assert template
    # The first non-templated phrase is deliberately distinct per v1 file.
    static_prefix = template.split("{{", maxsplit=1)[0].strip()
    assert static_prefix in llm.prompts[0]


async def test_generator_revision_prompt_contains_critic_instructions_and_changes_output():
    llm = PromptRecordingLLM("Revised answer uses the evidence only [Doc 1].")
    result = await GeneratorAgent(llm_wrapper=llm).run({
        "original_question": "What is hypertension?",
        "evidence": EVIDENCE,
        "revision_instructions": "Remove the unsupported treatment claim.",
    })

    assert result["answer"] == "Revised answer uses the evidence only [Doc 1]."
    assert "Remove the unsupported treatment claim." in llm.prompts[0]
