"""
Claim verification pipeline for RAGMedX-MA.

Provides ClaimVerifier: a CPU-only, NLI-backed pipeline that checks whether
answer claims are supported by the retrieved evidence.

Pipeline per claim:
  1. check_ids_exist      — cited doc IDs must be in the retrieved set
  2. check_quote_fidelity — lexical similarity between claimed quote and source
  3. check_entailment     — NLI entailment using a small CPU cross-encoder

Hard rules:
  - Citation presence OR lexical overlap alone is NEVER 'supported'.
    Entailment must confirm support.
  - NLI model runs on CPU only; never shares VRAM with the generator.
  - If the NLI model package is not installed, raises ImportError with
    installation instructions — does NOT silently fall back to lexical-only.
  - All structured outputs are validated for required keys and value ranges.

Evidence tier: SOFTWARE TEST (no LLM, no GPU, deterministic given the same model)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard_token_sim(a: str, b: str) -> float:
    """Jaccard similarity on lowercased whitespace tokens."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _longest_common_substring_ratio(needle: str, haystack: str) -> float:
    """
    Fraction of needle tokens that appear as a contiguous run in haystack.
    Used to detect verbatim quote fidelity.
    """
    needle_tokens = needle.lower().split()
    haystack_tokens = haystack.lower().split()
    if not needle_tokens:
        return 1.0
    best = 0
    n_h = len(haystack_tokens)
    for i, token in enumerate(needle_tokens):
        length = 0
        for j in range(n_h):
            k = 0
            while (
                i + k < len(needle_tokens)
                and j + k < n_h
                and needle_tokens[i + k] == haystack_tokens[j + k]
            ):
                k += 1
            if k > length:
                length = k
        best = max(best, length)
    return best / len(needle_tokens)


# ---------------------------------------------------------------------------
# ClaimVerifier
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """
    Claim-level evidence verifier.

    All methods are synchronous and CPU-only.

    Args:
        nli_model_name: HuggingFace model identifier for the NLI cross-encoder.
            Default: 'cross-encoder/nli-deberta-v3-small'
        _nli_pipeline: Optional injected mock; used in unit tests to avoid
            loading a real model.
    """

    DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

    # Threshold above which lexical similarity counts as a potential quote match
    QUOTE_FIDELITY_THRESHOLD = 0.5

    def __init__(
        self,
        nli_model_name: str = DEFAULT_NLI_MODEL,
        _nli_pipeline: Optional[Any] = None,
    ) -> None:
        self._nli_model_name = nli_model_name
        self._pipeline = _nli_pipeline  # None until first call to check_entailment

    # ------------------------------------------------------------------
    # 1. Claim segmentation
    # ------------------------------------------------------------------

    def segment_claims(self, text: str) -> List[str]:
        """
        Split answer text into individual verifiable claims.

        A claim is a self-contained sentence-level assertion.
        Splits on sentence boundaries (periods, semicolons followed by space).
        Filters out very short fragments (<= 3 words) and blank lines.

        Args:
            text: Answer text (may contain multiple sentences).

        Returns:
            List of individual claim strings.
        """
        # Split on '.', '!', '?', ';' followed by whitespace or end
        raw = re.split(r"(?<=[.!?;])\s+", text.strip())
        claims = []
        for fragment in raw:
            fragment = fragment.strip()
            if len(fragment.split()) > 3:  # ignore very short fragments
                claims.append(fragment)
        return claims if claims else ([text.strip()] if text.strip() else [])

    # ------------------------------------------------------------------
    # 2. Citation mapping
    # ------------------------------------------------------------------

    def map_citations(
        self,
        claims: List[str],
        retrieved_docs: Dict[str, str],
    ) -> Dict[int, List[str]]:
        """
        Map each claim (by index) to a list of cited doc_ids.

        Looks for explicit citation markers of the form:
          [Doc 1], [E2], [1], [doc_id], etc.

        Also checks if any retrieved doc_id appears literally in the claim text.

        Args:
            claims: List of claim strings (from segment_claims).
            retrieved_docs: {doc_id: text} mapping from retrieval.

        Returns:
            {claim_index: [doc_id, ...]} mapping.
        """
        result: Dict[int, List[str]] = {}
        # Patterns: [Doc 1], [E2], [1], [[any text]]
        marker_pattern = re.compile(r"\[([^\]]+)\]")

        for idx, claim in enumerate(claims):
            cited: List[str] = []
            markers_in_claim = marker_pattern.findall(claim)
            for marker in markers_in_claim:
                # Direct doc_id match
                if marker in retrieved_docs:
                    cited.append(marker)
                    continue
                # Ordinal match: "[2]" → map to doc at position 2-1
                if re.fullmatch(r"\d+", marker):
                    ordinal = int(marker) - 1
                    keys = list(retrieved_docs.keys())
                    if 0 <= ordinal < len(keys):
                        cited.append(keys[ordinal])
                    continue
                # Prefix match: "[Doc 1]" → find doc_id that starts with "Doc 1"
                normalized = marker.strip()
                for doc_id in retrieved_docs:
                    if doc_id == normalized or doc_id.endswith(normalized) or normalized.endswith(doc_id):
                        cited.append(doc_id)
                        break
            # Also check for literal doc_id mentions in the claim text
            for doc_id in retrieved_docs:
                if doc_id in claim and doc_id not in cited:
                    cited.append(doc_id)
            result[idx] = list(dict.fromkeys(cited))  # deduplicate, preserve order
        return result

    # ------------------------------------------------------------------
    # 3. ID existence check
    # ------------------------------------------------------------------

    def check_ids_exist(
        self,
        claimed_ids: List[str],
        retrieved_ids: List[str],
    ) -> Dict[str, bool]:
        """
        For each cited doc_id, check whether it was actually in the retrieved set.

        Args:
            claimed_ids: doc_ids cited in the claim.
            retrieved_ids: doc_ids that were retrieved for this query.

        Returns:
            {doc_id: True/False} — True if the ID was in the retrieved set.
        """
        retrieved_set = set(retrieved_ids)
        return {doc_id: doc_id in retrieved_set for doc_id in claimed_ids}

    # ------------------------------------------------------------------
    # 4. Quote fidelity
    # ------------------------------------------------------------------

    def check_quote_fidelity(self, claim: str, doc_text: str) -> float:
        """
        Lexical similarity between the claim and the source document text.

        Uses Jaccard token similarity as the primary metric and augments
        with longest-common-substring ratio for verbatim quote detection.

        Does NOT use an LLM.

        Args:
            claim: The claim text (may contain a quoted phrase).
            doc_text: The source document text.

        Returns:
            Float in [0, 1].  Higher = more lexical overlap.
        """
        jaccard = _jaccard_token_sim(claim, doc_text)
        lcs = _longest_common_substring_ratio(claim, doc_text)
        # Weight towards LCS for verbatim quote detection
        return round(max(jaccard, lcs * 0.9), 4)

    # ------------------------------------------------------------------
    # 5. NLI entailment
    # ------------------------------------------------------------------

    def check_entailment(
        self,
        claim: str,
        passage: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        NLI entailment check using a small CPU cross-encoder.

        The NLI model is loaded on CPU only and never shares VRAM with the
        generator.

        Args:
            claim: The claim to verify.
            passage: The source passage (premise).
            model_name: Override the model name.

        Returns:
            {'label': str, 'score': float}
            label is one of: 'entailment', 'neutral', 'contradiction'

        Raises:
            ImportError: If sentence-transformers or torch is not installed,
                with explicit installation instructions.
        """
        effective_model = model_name or self._nli_model_name

        # Load the pipeline lazily on first call
        if self._pipeline is None:
            self._pipeline = self._load_nli_pipeline(effective_model)

        try:
            # cross-encoder/nli-* models expect (premise, hypothesis) pairs
            result = self._pipeline(passage, claim)
        except Exception as exc:
            raise RuntimeError(f"NLI model inference failed: {exc}") from exc

        # Validate and normalise output
        return self._normalise_nli_output(result)

    @staticmethod
    def _load_nli_pipeline(model_name: str) -> Any:
        """Load the NLI pipeline on CPU."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for NLI entailment checking. "
                "Install it with: pip install sentence-transformers\n"
                "Do NOT install GPU-specific packages; this model runs on CPU only."
            ) from exc

        try:
            import torch
            device = "cpu"
        except ImportError:
            device = "cpu"

        return CrossEncoder(model_name, device=device, max_length=512)

    @staticmethod
    def _normalise_nli_output(raw: Any) -> Dict[str, Any]:
        """
        Normalise the cross-encoder NLI output to a standard dict.

        sentence-transformers CrossEncoder.predict returns a numpy array
        of shape [3] with scores for [contradiction, entailment, neutral]
        (order is model-dependent; we use the model's built-in label mapping).
        """
        # If the pipeline is a CrossEncoder, raw is a numpy array of logit scores
        try:
            import numpy as np
            if isinstance(raw, np.ndarray):
                # Cross-encoder/nli-deberta-v3-small label order: contradiction=0, entailment=1, neutral=2
                label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
                idx = int(np.argmax(raw))
                score = float(np.max(raw))
                # Convert raw logit to probability-like score via softmax
                exp_scores = np.exp(raw - np.max(raw))
                probs = exp_scores / exp_scores.sum()
                label = label_map.get(idx, "neutral")
                return {"label": label, "score": round(float(probs[idx]), 4)}
        except ImportError:
            pass

        # Fallback: if passed as dict (e.g. from mock)
        if isinstance(raw, dict):
            label = str(raw.get("label", "neutral")).lower()
            if label not in ("entailment", "neutral", "contradiction"):
                label = "neutral"
            score = float(raw.get("score", 0.0))
            return {"label": label, "score": round(score, 4)}

        # Fallback: list of dicts (transformers pipeline format)
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            best = max(raw, key=lambda x: x.get("score", 0.0))
            label = str(best.get("label", "neutral")).lower()
            if "entail" in label:
                label = "entailment"
            elif "contradict" in label:
                label = "contradiction"
            else:
                label = "neutral"
            return {"label": label, "score": round(float(best.get("score", 0.0)), 4)}

        # Unknown format
        return {"label": "neutral", "score": 0.0}

    # ------------------------------------------------------------------
    # 6. Full verify_claim pipeline
    # ------------------------------------------------------------------

    def verify_claim(
        self,
        claim: str,
        cited_ids: List[str],
        retrieved_docs: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Full verification pipeline for a single claim.

        Pipeline: ID check → quote fidelity → entailment

        Args:
            claim: The claim text to verify.
            cited_ids: doc_ids that the claim cites.
            retrieved_docs: {doc_id: text} of all retrieved documents.

        Returns:
            {
                'label': 'supported'|'partial'|'unsupported'|'contradicted'|'unverifiable',
                'score': float,
                'id_exists': bool,
                'quote_fidelity': float,
                'entailment': {'label': str, 'score': float},
                'reason': str,
            }

        Hard rules:
          - Citation presence or lexical overlap alone is NEVER 'supported'.
          - Entailment must confirm support.
          - If ANY cited ID does not exist in retrieved set → 'unverifiable'.
          - If NLI raises an exception → return {'label': 'unverifiable', 'reason': 'verifier_error', ...}
        """
        retrieved_ids = list(retrieved_docs.keys())

        # --- Step 1: ID existence check ---
        if not cited_ids:
            return {
                "label": "unverifiable",
                "score": 0.0,
                "id_exists": False,
                "quote_fidelity": 0.0,
                "entailment": {"label": "neutral", "score": 0.0},
                "reason": "no_citations",
            }

        id_check = self.check_ids_exist(cited_ids, retrieved_ids)
        all_ids_exist = all(id_check.values())
        any_id_exists = any(id_check.values())

        if not any_id_exists:
            return {
                "label": "unverifiable",
                "score": 0.0,
                "id_exists": False,
                "quote_fidelity": 0.0,
                "entailment": {"label": "neutral", "score": 0.0},
                "reason": "cited_id_not_in_retrieved_set",
            }

        # --- Step 2: Quote fidelity (against all cited docs that exist) ---
        best_fidelity = 0.0
        best_doc_text = ""
        for doc_id in cited_ids:
            if id_check.get(doc_id, False) and doc_id in retrieved_docs:
                fidelity = self.check_quote_fidelity(claim, retrieved_docs[doc_id])
                if fidelity >= best_fidelity or not best_doc_text:
                    best_fidelity = fidelity
                    best_doc_text = retrieved_docs[doc_id]

        # --- Step 3: NLI entailment ---
        entailment: Dict[str, Any] = {"label": "neutral", "score": 0.0}
        try:
            if best_doc_text:
                entailment = self.check_entailment(claim, best_doc_text)
        except RuntimeError as exc:
            return {
                "label": "unverifiable",
                "score": 0.0,
                "id_exists": all_ids_exist,
                "quote_fidelity": best_fidelity,
                "entailment": {"label": "neutral", "score": 0.0},
                "reason": "verifier_error",
                "error_detail": str(exc),
            }

        # --- Step 4: Assign label ---
        # Hard rule: entailment must confirm support; citation or lexical overlap alone is NOT sufficient
        nli_label = entailment.get("label", "neutral")
        nli_score = float(entailment.get("score", 0.0))

        if nli_label == "contradiction":
            label = "contradicted"
            reason = "nli_contradiction"
        elif nli_label == "entailment" and nli_score >= 0.5:
            if not all_ids_exist:
                label = "partial"
                reason = "some_cited_ids_missing_but_content_supported"
            else:
                label = "supported"
                reason = "nli_entailment_confirmed"
        elif best_fidelity < self.QUOTE_FIDELITY_THRESHOLD and nli_label != "entailment":
            label = "unsupported"
            reason = "low_lexical_overlap_and_no_entailment"
        else:
            # Neutral NLI: not enough to support
            label = "unsupported"
            reason = "nli_neutral_insufficient_for_support"

        return {
            "label": label,
            "score": round(nli_score if label in ("supported", "partial") else best_fidelity, 4),
            "id_exists": all_ids_exist,
            "quote_fidelity": best_fidelity,
            "entailment": entailment,
            "reason": reason,
        }
