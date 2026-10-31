"""
Unit tests for the OLLAMA runtime path of LocalLLMWrapper.

These tests patch the ``requests`` HTTP calls so they run without a live Ollama
server (safe for CI). They cover mode selection, think-block stripping, and the
strict readiness check that raises ModelLoadError instead of silently falling
back to mock.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper
from app.llm.parser import ModelLoadError


def _tags_response(model_names):
    """Build a fake /api/tags response object."""
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"models": [{"name": name} for name in model_names]},
    )


def _chat_response(content, prompt_eval_count=11, eval_count=7):
    """Build a fake /api/chat response object."""
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
        },
    )


def test_ollama_mode_selected_and_not_mock():
    wrapper = LocalLLMWrapper(
        mode=LLMRuntimeMode.OLLAMA,
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
        allow_fallback=False,
    )

    with patch("requests.get", return_value=_tags_response(["qwen3:8b"])):
        assert wrapper.load_model() is True

    assert wrapper.active_runtime_mode == LLMRuntimeMode.OLLAMA
    assert wrapper.is_loaded is True

    with patch("requests.post", return_value=_chat_response("ok")):
        text, usage = wrapper.generate("Reply with the single word: ok")

    assert text == "ok"
    assert usage["is_mock"] is False
    assert usage["mode"] == "ollama"
    assert usage["model_name"] == "qwen3:8b"
    assert "[MOCK_GENERATED_ANSWER" not in text
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 18


def test_ollama_strips_think_block():
    wrapper = LocalLLMWrapper(
        mode=LLMRuntimeMode.OLLAMA,
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
        allow_fallback=False,
    )

    with patch("requests.get", return_value=_tags_response(["qwen3:8b"])):
        wrapper.load_model()

    content = "<think>Let me reason about this.</think>Metformin is first-line therapy."
    with patch("requests.post", return_value=_chat_response(content)):
        text, _ = wrapper.generate("What is first-line therapy for T2D?")

    assert "<think>" not in text
    assert "reason about this" not in text
    assert text == "Metformin is first-line therapy."


def test_ollama_load_raises_when_unreachable():
    wrapper = LocalLLMWrapper(
        mode=LLMRuntimeMode.OLLAMA,
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
        allow_fallback=False,
    )

    def _raise(*args, **kwargs):
        raise ConnectionError("connection refused")

    with patch("requests.get", side_effect=_raise):
        with pytest.raises(ModelLoadError) as exc_info:
            wrapper.load_model()

    assert "not reachable" in str(exc_info.value)
    assert wrapper.is_loaded is False


def test_ollama_load_raises_when_tag_missing():
    wrapper = LocalLLMWrapper(
        mode=LLMRuntimeMode.OLLAMA,
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
        allow_fallback=False,
    )

    with patch("requests.get", return_value=_tags_response(["llama3:8b"])):
        with pytest.raises(ModelLoadError) as exc_info:
            wrapper.load_model()

    assert "not available" in str(exc_info.value)
    assert wrapper.is_loaded is False
