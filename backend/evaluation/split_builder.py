"""
Dataset split builder and leakage detection for RAGMedX-MA evaluation.

Functions
---------
build_splits              -- seeded, stratified split into dev/calibration/test
check_near_duplicates     -- near-duplicate detection between two splits
check_question_leakage    -- check if questions appear in the corpus
check_answer_leakage      -- check if reference answers appear in the corpus
check_prompt_leakage      -- check if reference answers appear in prompt templates

Classes
-------
IndexLeakageGuard         -- guards against accidentally using the production FAISS index
IndexLeakageError         -- raised when production index is detected
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# IndexLeakageError
# ---------------------------------------------------------------------------

class IndexLeakageError(Exception):
    """Raised when an index path resolves to the protected production index."""
    pass


# ---------------------------------------------------------------------------
# IndexLeakageGuard
# ---------------------------------------------------------------------------

_PRODUCTION_INDEX_CANDIDATES = [
    # Normalised absolute path of the production index root
    "backend\\data\\faiss_index",
    "backend/data/faiss_index",
]

# The allowed eval split path fragment — must be a child of faiss_index/eval_split
_EVAL_SPLIT_FRAGMENT = "eval_split"


class IndexLeakageGuard:
    """Guard against accidentally pointing evaluation at the production FAISS index.

    Usage
    -----
    ::

        guard = IndexLeakageGuard()
        guard.assert_safe(r"d:\\Ragmed_2\\backend\\data\\faiss_index\\eval_split")  # OK
        guard.assert_safe(r"d:\\Ragmed_2\\backend\\data\\faiss_index")  # raises

    Rules
    -----
    The path is safe if and only if ALL of the following hold:
      1. The resolved absolute path contains the ``eval_split`` directory
         component somewhere in its path.
      2. The resolved absolute path is NOT equal to (or a parent of) the
         production index root ``backend/data/faiss_index``.
      3. Symlinks are fully resolved (via ``Path.resolve()``).

    If the path does not exist on disk the guard still performs the string
    analysis on the normalised path to catch misconfigurations.
    """

    def __init__(self, repo_root: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        repo_root : str, optional
            Absolute path to the repository root.  If None, inferred from
            this file's location (two levels up from backend/evaluation/).
        """
        if repo_root is None:
            # This file lives at backend/evaluation/split_builder.py
            # so repo_root = parent of parent of parent
            self._repo_root = Path(__file__).resolve().parent.parent.parent
        else:
            self._repo_root = Path(repo_root).resolve()

    def _production_root(self) -> Path:
        """Return the resolved absolute path of the production FAISS index."""
        return (self._repo_root / "backend" / "data" / "faiss_index").resolve()

    def assert_safe(self, index_path: str) -> None:
        """Assert that index_path is safe to use as an eval index.

        Parameters
        ----------
        index_path : str — path to the index directory being checked

        Raises
        ------
        IndexLeakageError if the path is the production index or a non-eval_split
        subpath of the production index.
        ValueError if index_path is None or empty.
        """
        if not index_path:
            raise ValueError("IndexLeakageGuard.assert_safe: index_path must not be empty")

        resolved = Path(index_path).resolve()
        production_root = self._production_root()

        # Check 1: is this path equal to the production root?
        try:
            if resolved == production_root:
                raise IndexLeakageError(
                    f"PRODUCTION INDEX PATH REJECTED: '{index_path}' resolves to "
                    f"the production index root '{production_root}'. "
                    "Use backend/data/faiss_index/eval_split/ instead."
                )
        except (OSError, ValueError):
            pass  # resolve may fail on non-existent paths; fall through to string check

        # Check 2: is this path a parent of (or equal to) the production root?
        # i.e., is it *above* the production root? (less restrictive, always bad)
        try:
            production_root.relative_to(resolved)
            # If we reach here, resolved is a parent of production_root — dangerous
            raise IndexLeakageError(
                f"PRODUCTION INDEX PATH REJECTED: '{index_path}' resolves to "
                f"'{resolved}' which is an ancestor of the production index. "
                "You must use a path under eval_split/."
            )
        except ValueError:
            pass  # Not a parent — good

        # Check 3: is this a sub-path of production root WITHOUT eval_split?
        try:
            relative = resolved.relative_to(production_root)
            # It IS under the production root — only safe if eval_split is the first component
            parts = relative.parts
            if not parts or parts[0].lower() != _EVAL_SPLIT_FRAGMENT.lower():
                raise IndexLeakageError(
                    f"PRODUCTION INDEX PATH REJECTED: '{index_path}' resolves to "
                    f"a subpath of the production index root that is NOT under "
                    f"eval_split/. Resolved: '{resolved}'. "
                    "Only backend/data/faiss_index/eval_split/ (and its children) are allowed."
                )
            # eval_split sub-path → safe
        except ValueError:
            # Not under production root at all → safe (external path)
            pass

        # Check 4: string-level fallback for non-existent paths
        # Normalise separators
        normalised = str(resolved).replace("\\", "/").lower()
        prod_str = str(production_root).replace("\\", "/").lower()

        if normalised == prod_str:
            raise IndexLeakageError(
                f"PRODUCTION INDEX PATH REJECTED (string check): '{index_path}'"
            )

        # Check that if it contains the production path segment, eval_split appears too
        prod_fragment = "backend/data/faiss_index"
        if prod_fragment in normalised:
            # Must also have eval_split after the production fragment
            after = normalised.split(prod_fragment, 1)[1]
            if _EVAL_SPLIT_FRAGMENT not in after.lower():
                raise IndexLeakageError(
                    f"PRODUCTION INDEX PATH REJECTED (string check): '{index_path}' "
                    f"contains the production index path without eval_split component."
                )


# ---------------------------------------------------------------------------
# Split Builder
# ---------------------------------------------------------------------------

def build_splits(
    dataset: List[Dict[str, Any]],
    dev_n: int,
    cal_n: int,
    test_n: int,
    seed: int = 42,
    stratify_by: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build seeded, stratified dev / calibration / test splits.

    Parameters
    ----------
    dataset     : list of dicts, each must have a unique 'id' field (or 'question_id').
    dev_n       : int — number of samples for the development split
    cal_n       : int — number of samples for the calibration split
    test_n      : int — number of samples for the test split
    seed        : int — random seed; default 42
    stratify_by : str, optional — field name to stratify on (e.g. 'type' for BioASQ)

    Returns
    -------
    (dev_split, cal_split, test_split) : tuple of lists

    Raises
    ------
    ValueError if insufficient data or duplicate IDs.
    """
    import random as _random

    total_needed = dev_n + cal_n + test_n
    if len(dataset) < total_needed:
        raise ValueError(
            f"build_splits: dataset has {len(dataset)} items but "
            f"dev_n={dev_n} + cal_n={cal_n} + test_n={test_n} = {total_needed} required"
        )

    # Check for duplicate IDs
    id_field = "id" if dataset and "id" in dataset[0] else "question_id"
    ids = [item.get(id_field, i) for i, item in enumerate(dataset)]
    if len(ids) != len(set(ids)):
        raise ValueError("build_splits: dataset contains duplicate IDs")

    rng = _random.Random(seed)

    if stratify_by is not None:
        # Group by stratum
        strata: Dict[Any, List[int]] = {}
        for i, item in enumerate(dataset):
            key = item.get(stratify_by, "__unknown__")
            strata.setdefault(key, []).append(i)

        # Compute per-stratum allocation proportional to stratum size
        n_total = len(dataset)
        dev_indices: List[int] = []
        cal_indices: List[int] = []
        test_indices: List[int] = []

        # Two-pass: first allocate proportionally, then fill remainder
        stratum_keys = sorted(strata.keys(), key=str)
        allocations: Dict[Any, Tuple[int, int, int]] = {}
        for key in stratum_keys:
            frac = len(strata[key]) / n_total
            d = max(1, round(frac * dev_n))
            c = max(1, round(frac * cal_n))
            t = max(1, round(frac * test_n))
            allocations[key] = (d, c, t)

        # Adjust to exact totals
        def _adjust(alloc_key: int, target: int) -> None:
            vals = [allocations[k][alloc_key] for k in stratum_keys]
            diff = sum(vals) - target
            # Add/remove from the largest stratum first
            strata_by_size = sorted(stratum_keys, key=lambda k: len(strata[k]), reverse=True)
            idx = 0
            while diff > 0:
                k = strata_by_size[idx % len(strata_by_size)]
                old = list(allocations[k])
                if old[alloc_key] > 1:
                    old[alloc_key] -= 1
                    allocations[k] = tuple(old)  # type: ignore[assignment]
                    diff -= 1
                idx += 1
            while diff < 0:
                k = strata_by_size[idx % len(strata_by_size)]
                old = list(allocations[k])
                old[alloc_key] += 1
                allocations[k] = tuple(old)  # type: ignore[assignment]
                diff += 1
                idx += 1

        _adjust(0, dev_n)
        _adjust(1, cal_n)
        _adjust(2, test_n)

        for key in stratum_keys:
            stratum_indices = strata[key][:]
            rng.shuffle(stratum_indices)
            d, c, t = allocations[key]
            needed = d + c + t
            if len(stratum_indices) < needed:
                raise ValueError(
                    f"build_splits: stratum '{key}' has {len(stratum_indices)} items "
                    f"but {needed} needed for stratified split"
                )
            dev_indices.extend(stratum_indices[:d])
            cal_indices.extend(stratum_indices[d:d + c])
            test_indices.extend(stratum_indices[d + c:d + c + t])
    else:
        # Non-stratified: shuffle and slice
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        dev_indices = indices[:dev_n]
        cal_indices = indices[dev_n:dev_n + cal_n]
        test_indices = indices[dev_n + cal_n:dev_n + cal_n + test_n]

    dev_split  = [dataset[i] for i in dev_indices]
    cal_split  = [dataset[i] for i in cal_indices]
    test_split = [dataset[i] for i in test_indices]

    return dev_split, cal_split, test_split


# ---------------------------------------------------------------------------
# Near-Duplicate Detection
# ---------------------------------------------------------------------------

def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between token sets of two strings (lowercased)."""
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def check_near_duplicates(
    split_a: List[Dict[str, Any]],
    split_b: List[Dict[str, Any]],
    threshold: float = 0.9,
    text_field: str = "question",
) -> List[Tuple[Any, Any, float]]:
    """Find near-duplicate pairs between two splits.

    Similarity is measured as Jaccard similarity on whitespace-split tokens
    (lowercased). Pairs with similarity >= threshold are returned.

    Parameters
    ----------
    split_a    : list of dicts with 'id' (or 'question_id') and text_field
    split_b    : list of dicts with the same fields
    threshold  : float — similarity threshold in [0, 1]; default 0.9
    text_field : str   — field containing the question text; default 'question'

    Returns
    -------
    List of (id_a, id_b, similarity) tuples where similarity >= threshold.
    Empty list if no near-duplicates found.
    """
    results: List[Tuple[Any, Any, float]] = []
    id_field = "id"

    def _get_id(item: Dict[str, Any]) -> Any:
        return item.get("id", item.get("question_id", id(item)))

    for item_a in split_a:
        text_a = str(item_a.get(text_field, ""))
        id_a = _get_id(item_a)
        for item_b in split_b:
            text_b = str(item_b.get(text_field, ""))
            id_b = _get_id(item_b)
            sim = _token_jaccard(text_a, text_b)
            if sim >= threshold:
                results.append((id_a, id_b, round(sim, 4)))

    return results


# ---------------------------------------------------------------------------
# Question Leakage Check
# ---------------------------------------------------------------------------

def check_question_leakage(
    questions: List[Dict[str, Any]],
    corpus_texts: List[str],
    threshold: float = 0.95,
    text_field: str = "question",
) -> List[Any]:
    """Check if any question appears verbatim or near-verbatim in the eval corpus.

    Uses Jaccard token similarity. Returns IDs of leaking questions.

    Parameters
    ----------
    questions    : list of dicts with 'id' (or 'question_id') and text_field
    corpus_texts : list of str — corpus document/chunk texts to check against
    threshold    : float — similarity threshold; default 0.95
    text_field   : str   — field containing question text

    Returns
    -------
    List of question IDs where similarity >= threshold with any corpus text.
    """
    leaking_ids: List[Any] = []

    def _get_id(item: Dict[str, Any]) -> Any:
        return item.get("id", item.get("question_id", id(item)))

    for item in questions:
        q_text = str(item.get(text_field, ""))
        q_id = _get_id(item)
        for doc_text in corpus_texts:
            sim = _token_jaccard(q_text, doc_text)
            if sim >= threshold:
                leaking_ids.append(q_id)
                break  # Only report once per question

    return leaking_ids


# ---------------------------------------------------------------------------
# Answer Leakage Check
# ---------------------------------------------------------------------------

def check_answer_leakage(
    reference_answers: List[Dict[str, Any]],
    corpus_texts: List[str],
    threshold: float = 0.95,
    answer_field: str = "answer",
) -> List[Any]:
    """Check if reference answers appear near-verbatim in the eval corpus.

    Uses Jaccard token similarity. Returns IDs of leaking answers.

    Parameters
    ----------
    reference_answers : list of dicts with 'id' (or 'question_id') and answer_field
    corpus_texts      : list of str — corpus document/chunk texts
    threshold         : float — similarity threshold; default 0.95
    answer_field      : str   — field containing the reference answer

    Returns
    -------
    List of question IDs where the reference answer similarity >= threshold
    with any corpus text.
    """
    leaking_ids: List[Any] = []

    def _get_id(item: Dict[str, Any]) -> Any:
        return item.get("id", item.get("question_id", id(item)))

    for item in reference_answers:
        ans_text = str(item.get(answer_field, ""))
        item_id = _get_id(item)
        if not ans_text.strip():
            continue
        for doc_text in corpus_texts:
            sim = _token_jaccard(ans_text, doc_text)
            if sim >= threshold:
                leaking_ids.append(item_id)
                break

    return leaking_ids


# ---------------------------------------------------------------------------
# Prompt Leakage Check
# ---------------------------------------------------------------------------

def check_prompt_leakage(
    reference_answers: List[Dict[str, Any]],
    prompt_templates: List[str],
    answer_field: str = "answer",
) -> List[Any]:
    """Check if any reference answer text appears in any prompt template.

    Uses exact substring containment (case-insensitive).

    Parameters
    ----------
    reference_answers : list of dicts with 'id' (or 'question_id') and answer_field
    prompt_templates  : list of str — prompt template strings to check
    answer_field      : str         — field containing the reference answer

    Returns
    -------
    List of question IDs whose reference answer appears in any prompt template.
    """
    leaking_ids: List[Any] = []

    def _get_id(item: Dict[str, Any]) -> Any:
        return item.get("id", item.get("question_id", id(item)))

    templates_lower = [t.lower() for t in prompt_templates]

    for item in reference_answers:
        ans_text = str(item.get(answer_field, "")).strip().lower()
        item_id = _get_id(item)
        if not ans_text:
            continue
        for tmpl in templates_lower:
            if ans_text in tmpl:
                leaking_ids.append(item_id)
                break

    return leaking_ids
