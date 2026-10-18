"""
Canonical citation-marker contract for RAGMedX-MA.

Single source of truth for how in-text citation markers are written and
extracted, shared by the GeneratorAgent (which emits/validates them) and the
CriticAgent (which verifies them). Keeping one regex here prevents the class of
bug where the generator prompt instructs the model to emit ``[Doc 1]`` but the
extractor only matched ``[E1]``-style markers, so correct citations were dropped
from ``answer_citations`` and hallucinated ``[Doc N]`` markers slipped past the
generator's own unsupported-citation guard.

Supported marker forms (verified against the live retriever + stub + tests):
- ``[Doc 1]``, ``[Doc 12]``  -- canonical form emitted by CitationGenerator /
  RAGPipeline for native FAISS evidence (``_build_citation_id`` -> ``[Doc N]``).
- ``[Doc E1]``                -- alpha-prefixed variant observed from Qwen3-8B;
  handled by the optional single letter before the digits in the ``Doc`` branch.
- ``[E1]``, ``[E2]``         -- legacy/synthetic form used by the temporary
  retrieval stub and existing unit tests.
- ``[A-Za-z0-9_-]+``         -- generic single-token markers.

A matched marker is always returned normalized WITH its surrounding brackets
(e.g. ``"[Doc 1]"``), matching the ``citation_marker`` value stored on evidence
items so that extracted markers compare equal to the evidence set.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set

# Order matters: try the multi-word ``[Doc N]`` form before the generic
# single-token form so ``[Doc 1]`` is captured whole rather than failing.
# The optional ``[A-Za-z]?`` before the digits lets the ``Doc`` branch accept
# alpha-prefixed ids like ``[Doc E1]`` (emitted by Qwen3-8B) in addition to
# plain ``[Doc 1]`` / ``[Doc 12]``, while legacy ``[E1]`` and generic
# single-token markers still match via the second branch.
CITATION_MARKER_PATTERN = re.compile(r"\[(?:Doc\s+[A-Za-z]?\d+|[A-Za-z0-9_-]+)\]")


def normalize_marker(marker: str) -> str:
    """Return a marker in canonical bracketed form.

    ``"E1"`` -> ``"[E1]"``; ``"[Doc 1]"`` -> ``"[Doc 1]"``. Internal whitespace
    is collapsed to a single space so ``"[Doc  1]"`` and ``"[Doc 1]"`` compare
    equal. ``None``/empty returns ``""``.
    """
    if not marker:
        return ""
    text = str(marker).strip()
    if not text:
        return ""
    if not text.startswith("["):
        text = f"[{text}]"
    # Collapse internal whitespace (e.g. "[Doc   1]" -> "[Doc 1]").
    return re.sub(r"\s+", " ", text).strip()


def extract_markers(text: str) -> Set[str]:
    """Extract all citation markers found in ``text`` as a normalized set.

    Returns markers WITH brackets, e.g. ``{"[Doc 1]", "[E2]"}``.
    """
    if not text:
        return set()
    return {normalize_marker(m) for m in CITATION_MARKER_PATTERN.findall(text)}


def evidence_markers(evidence: Iterable) -> List[str]:
    """Return the canonical marker for each evidence item, in order.

    Falls back to a positional ``[Doc i]`` marker (1-based) when an item carries
    no ``citation_marker``. Accepts dict items or objects with attributes.
    """
    markers: List[str] = []
    for index, item in enumerate(evidence or [], start=1):
        if isinstance(item, dict):
            raw = item.get("citation_marker")
        else:
            raw = getattr(item, "citation_marker", None)
        markers.append(normalize_marker(raw) if raw else f"[Doc {index}]")
    return markers
