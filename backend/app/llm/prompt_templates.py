"""Backward-compatible imports for prompts now owned by :mod:`app.rag.prompts`."""

from app.rag.prompts.qa_prompt import load_prompt_template, render_prompt

__all__ = ["load_prompt_template", "render_prompt"]
