import asyncio
import logging
from datetime import datetime, timezone
import sys
import os

# Ensure backend root is on sys.path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from app.llm.local_llm import LocalLLMWrapper
from app.llm.tokenizer import count_tokens, truncate_to_max_tokens
from app.llm.prompt_templates import render_prompt
from app.agents.base import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_local_llm")


class DummyAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    async def run(self, state):
        started_at = datetime.now(timezone.utc)

        prompt = (
            "What is the first-line treatment for "
            "uncomplicated essential hypertension?"
        )

        response = self.llm_wrapper.generate(
            prompt=prompt,
            max_new_tokens=100,
        )

        trace_step = self.create_trace_step(
            status="completed",
            started_at=started_at,
            details={
                "prompt": prompt,
                "response_length": len(response),
            },
        )

        return {
            "response": response,
            "trace": trace_step,
        }


def test_tokenizer():
    logger.info("=== Testing Tokenizer ===")

    sample_text = (
        "Hypertension is a chronic medical condition in which "
        "blood pressure in the arteries is persistently elevated."
    )

    token_cnt = count_tokens(sample_text)

    logger.info(
        f"Sample text char count: {len(sample_text)}, "
        f"Token count estimate: {token_cnt}"
    )

    assert token_cnt > 0

    truncated = truncate_to_max_tokens(
        sample_text,
        max_tokens=5,
    )

    logger.info(
        f"Truncated text (5 tokens): '{truncated}'"
    )

    assert len(truncated) <= len(sample_text)

    print("[OK] Tokenizer test passed.")


def test_prompt_rendering():
    logger.info("=== Testing Prompt Templates ===")

    rendered = render_prompt(
        "query_understanding",
        {
            "question": "How to treat type 2 diabetes?"
        },
    )

    logger.info(
        f"Rendered Prompt Snippet:\n{rendered[:200]}..."
    )

    assert "How to treat type 2 diabetes?" in rendered

    print("[OK] Prompt template test passed.")


async def test_llm_generation():
    logger.info("=== Testing Local LLM Wrapper ===")

    # Use the project's configured local model.
    # The wrapper automatically selects CUDA when available
    # and CPU otherwise.
    llm = LocalLLMWrapper(
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"
    )

    llm.load_model()

    response = llm.generate(
        prompt=(
            "What are the primary symptoms of "
            "acute myocardial infarction?"
        ),
        max_new_tokens=64,
    )

    logger.info(
        f"LLM Response: {response}"
    )

    assert response is not None
    assert isinstance(response, str)
    assert len(response.strip()) > 0

    print("[OK] LLM Wrapper generation test passed.")


async def test_base_agent():
    logger.info(
        "=== Testing BaseAgent & Pipeline State Integration ==="
    )

    llm = LocalLLMWrapper(
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"
    )

    agent = DummyAgent(
        agent_name="DummyTestAgent",
        agent_type="query_understanding",
        llm_wrapper=llm,
    )

    state_update = await agent.run({})

    logger.info(
        f"Agent State Update: {state_update}"
    )

    assert "response" in state_update
    assert isinstance(state_update["response"], str)

    assert "trace" in state_update
    assert state_update["trace"]["status"] == "completed"
    assert state_update["trace"]["agent_type"] == "query_understanding"

    print("[OK] BaseAgent integration test passed.")


if __name__ == "__main__":
    print(
        "Running RAGMedX-MA Local LLM & Base Agent Smoke Tests..."
    )

    test_tokenizer()
    test_prompt_rendering()

    asyncio.run(test_llm_generation())
    asyncio.run(test_base_agent())

    print(
        "\nALL SMOKE TESTS PASSED SUCCESSFULLY!"
    )