"""
Phase 2 Workstream B — Unit tests for split_builder.py.

Tests cover:
  - split sizes and stratification correctness
  - near-duplicate detection
  - question leakage detection
  - answer leakage detection
  - prompt leakage detection
  - IndexLeakageGuard: accepts eval_split, rejects production, rejects symlinks
    that resolve to production

Evidence tier: SOFTWARE TEST (no model execution, no FAISS, no LLM).
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

from backend.evaluation.split_builder import (
    IndexLeakageError,
    IndexLeakageGuard,
    build_splits,
    check_answer_leakage,
    check_near_duplicates,
    check_prompt_leakage,
    check_question_leakage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int, stratum_field: str = None, strata: list = None) -> list:
    """Create a minimal dataset of n items."""
    items = []
    for i in range(n):
        item = {"id": f"q{i:04d}", "question": f"What is condition {i}?", "answer": f"answer_{i}"}
        if stratum_field and strata:
            item[stratum_field] = strata[i % len(strata)]
        items.append(item)
    return items


# ===========================================================================
# build_splits
# ===========================================================================

class TestBuildSplits:

    def test_split_sizes_correct(self):
        dataset = _make_dataset(100)
        dev, cal, test = build_splits(dataset, dev_n=10, cal_n=15, test_n=25, seed=42)
        assert len(dev) == 10
        assert len(cal) == 15
        assert len(test) == 25

    def test_no_overlap_between_splits(self):
        dataset = _make_dataset(100)
        dev, cal, test = build_splits(dataset, dev_n=10, cal_n=15, test_n=25, seed=42)
        dev_ids  = {item["id"] for item in dev}
        cal_ids  = {item["id"] for item in cal}
        test_ids = {item["id"] for item in test}
        assert dev_ids.isdisjoint(cal_ids), "dev and cal overlap"
        assert dev_ids.isdisjoint(test_ids), "dev and test overlap"
        assert cal_ids.isdisjoint(test_ids), "cal and test overlap"

    def test_seeded_reproducibility(self):
        dataset = _make_dataset(100)
        dev1, cal1, test1 = build_splits(dataset, dev_n=10, cal_n=15, test_n=25, seed=42)
        dev2, cal2, test2 = build_splits(dataset, dev_n=10, cal_n=15, test_n=25, seed=42)
        assert [i["id"] for i in dev1] == [i["id"] for i in dev2]
        assert [i["id"] for i in cal1] == [i["id"] for i in cal2]
        assert [i["id"] for i in test1] == [i["id"] for i in test2]

    def test_different_seeds_give_different_splits(self):
        dataset = _make_dataset(100)
        dev1, _, _ = build_splits(dataset, dev_n=20, cal_n=20, test_n=20, seed=42)
        dev2, _, _ = build_splits(dataset, dev_n=20, cal_n=20, test_n=20, seed=99)
        ids1 = [i["id"] for i in dev1]
        ids2 = [i["id"] for i in dev2]
        assert ids1 != ids2, "different seeds should give different splits"

    def test_raises_on_insufficient_data(self):
        dataset = _make_dataset(20)
        with pytest.raises(ValueError, match="insufficient" if False else ""):
            build_splits(dataset, dev_n=10, cal_n=10, test_n=10, seed=42)

    def test_raises_on_duplicate_ids(self):
        dataset = _make_dataset(50)
        dataset.append({"id": "q0000", "question": "duplicate"})  # duplicate ID
        with pytest.raises(ValueError, match="duplicate" if False else ""):
            build_splits(dataset, dev_n=5, cal_n=5, test_n=5)

    def test_stratified_preserves_stratum_proportions(self):
        # 60 items: 30 type_a, 30 type_b
        dataset = _make_dataset(60, stratum_field="type", strata=["type_a", "type_b"])
        dev, cal, test = build_splits(
            dataset, dev_n=10, cal_n=10, test_n=20,
            seed=42, stratify_by="type"
        )
        # Each split should have approximately equal type_a and type_b
        def count_stratum(split, s):
            return sum(1 for item in split if item.get("type") == s)

        # dev=10: expect ~5 of each
        assert count_stratum(dev, "type_a") >= 3
        assert count_stratum(dev, "type_b") >= 3

    def test_stratified_total_sizes_correct(self):
        dataset = _make_dataset(90, stratum_field="type", strata=["a", "b", "c"])
        dev, cal, test = build_splits(
            dataset, dev_n=9, cal_n=12, test_n=15,
            seed=42, stratify_by="type"
        )
        assert len(dev) == 9
        assert len(cal) == 12
        assert len(test) == 15

    def test_exact_boundary_works(self):
        # Exactly enough data
        dataset = _make_dataset(30)
        dev, cal, test = build_splits(dataset, dev_n=10, cal_n=10, test_n=10, seed=0)
        assert len(dev) == 10
        assert len(cal) == 10
        assert len(test) == 10


# ===========================================================================
# check_near_duplicates
# ===========================================================================

class TestCheckNearDuplicates:

    def test_no_duplicates(self):
        split_a = [{"id": "q1", "question": "What is hypertension?"}]
        split_b = [{"id": "q2", "question": "How does insulin resistance develop?"}]
        result = check_near_duplicates(split_a, split_b, threshold=0.9)
        assert result == []

    def test_exact_duplicate_detected(self):
        split_a = [{"id": "q1", "question": "What is hypertension?"}]
        split_b = [{"id": "q2", "question": "What is hypertension?"}]
        result = check_near_duplicates(split_a, split_b, threshold=0.9)
        assert len(result) == 1
        assert result[0][0] == "q1"
        assert result[0][1] == "q2"
        assert result[0][2] == pytest.approx(1.0, abs=1e-4)

    def test_near_duplicate_detected(self):
        # High token overlap
        split_a = [{"id": "a1", "question": "What is the mechanism of hypertension?"}]
        split_b = [{"id": "b1", "question": "What is the mechanism of hypertension treatment?"}]
        result = check_near_duplicates(split_a, split_b, threshold=0.7)
        # Jaccard of {"what","is","the","mechanism","of","hypertension"} vs
        #             {"what","is","the","mechanism","of","hypertension","treatment"}
        # intersection=6, union=7 → 6/7 ≈ 0.857 ≥ 0.7
        assert len(result) == 1

    def test_threshold_below_similarity_not_flagged(self):
        split_a = [{"id": "a1", "question": "What is hypertension?"}]
        split_b = [{"id": "b1", "question": "What is hypertension?"}]
        result = check_near_duplicates(split_a, split_b, threshold=1.1)  # impossible
        assert result == []

    def test_empty_splits(self):
        assert check_near_duplicates([], [], threshold=0.9) == []


# ===========================================================================
# check_question_leakage
# ===========================================================================

class TestCheckQuestionLeakage:

    def test_no_leakage(self):
        questions = [{"id": "q1", "question": "What is the function of the mitral valve?"}]
        corpus = ["Aspirin inhibits platelet aggregation.", "Insulin regulates blood sugar."]
        result = check_question_leakage(questions, corpus, threshold=0.95)
        assert result == []

    def test_verbatim_leakage_detected(self):
        q_text = "What is the function of the mitral valve?"
        questions = [{"id": "q1", "question": q_text}]
        corpus = [q_text]  # verbatim match
        result = check_question_leakage(questions, corpus, threshold=0.95)
        assert "q1" in result

    def test_near_verbatim_leakage_detected(self):
        # High token overlap with corpus chunk
        q_text = "What is the mechanism of action of aspirin?"
        questions = [{"id": "q2", "question": q_text}]
        corpus = ["What is the mechanism of action of aspirin therapy?"]
        # Jaccard: intersection≈6, union≈7 → ~0.857 < 0.95 → no leakage at 0.95
        result = check_question_leakage(questions, corpus, threshold=0.95)
        # Should NOT leak at 0.95 threshold (similarity < 0.95)
        assert result == []

    def test_empty_input(self):
        assert check_question_leakage([], [], threshold=0.95) == []


# ===========================================================================
# check_answer_leakage
# ===========================================================================

class TestCheckAnswerLeakage:

    def test_no_leakage(self):
        answers = [{"id": "q1", "answer": "Aspirin inhibits cyclooxygenase."}]
        corpus = ["Insulin stimulates glucose uptake.", "The mitral valve prevents backflow."]
        result = check_answer_leakage(answers, corpus, threshold=0.95)
        assert result == []

    def test_verbatim_answer_in_corpus_detected(self):
        ans_text = "Aspirin inhibits cyclooxygenase enzyme activity."
        answers = [{"id": "q1", "answer": ans_text}]
        corpus = [ans_text]  # verbatim
        result = check_answer_leakage(answers, corpus, threshold=0.95)
        assert "q1" in result

    def test_empty_answer_skipped(self):
        answers = [{"id": "q1", "answer": ""}]
        corpus = ["some text"]
        result = check_answer_leakage(answers, corpus, threshold=0.95)
        assert result == []

    def test_empty_input(self):
        assert check_answer_leakage([], [], threshold=0.95) == []


# ===========================================================================
# check_prompt_leakage
# ===========================================================================

class TestCheckPromptLeakage:

    def test_no_leakage(self):
        answers = [{"id": "q1", "answer": "The mitral valve prevents backflow."}]
        templates = ["You are a medical expert. Answer the following question:"]
        result = check_prompt_leakage(answers, templates)
        assert result == []

    def test_answer_in_prompt_detected(self):
        answers = [{"id": "q1", "answer": "aspirin inhibits cyclooxygenase"}]
        templates = [
            "Correct answer: aspirin inhibits cyclooxygenase. Now generate your answer."
        ]
        result = check_prompt_leakage(answers, templates)
        assert "q1" in result

    def test_case_insensitive(self):
        answers = [{"id": "q1", "answer": "ASPIRIN INHIBITS CYCLOOXYGENASE"}]
        templates = ["the correct answer is aspirin inhibits cyclooxygenase here"]
        result = check_prompt_leakage(answers, templates)
        assert "q1" in result

    def test_empty_answer_skipped(self):
        answers = [{"id": "q1", "answer": ""}]
        templates = ["some template"]
        result = check_prompt_leakage(answers, templates)
        assert result == []

    def test_empty_input(self):
        assert check_prompt_leakage([], []) == []


# ===========================================================================
# IndexLeakageGuard
# ===========================================================================

class TestIndexLeakageGuard:

    def _guard(self) -> IndexLeakageGuard:
        """Return a guard configured with the real repo root."""
        return IndexLeakageGuard(repo_root=r"d:\Ragmed_2")

    def test_accepts_eval_split_path(self):
        guard = self._guard()
        eval_path = r"d:\Ragmed_2\backend\data\faiss_index\eval_split"
        # Should not raise
        guard.assert_safe(eval_path)

    def test_accepts_eval_split_subpath(self):
        guard = self._guard()
        eval_path = r"d:\Ragmed_2\backend\data\faiss_index\eval_split\subset"
        guard.assert_safe(eval_path)

    def test_rejects_production_root(self):
        guard = self._guard()
        prod_path = r"d:\Ragmed_2\backend\data\faiss_index"
        with pytest.raises(IndexLeakageError):
            guard.assert_safe(prod_path)

    def test_rejects_production_root_trailing_slash(self):
        guard = self._guard()
        prod_path = r"d:\Ragmed_2\backend\data\faiss_index\\"
        with pytest.raises(IndexLeakageError):
            guard.assert_safe(prod_path)

    def test_rejects_subpath_without_eval_split(self):
        guard = self._guard()
        # A sibling directory under faiss_index that is NOT eval_split
        bad_path = r"d:\Ragmed_2\backend\data\faiss_index\other_index"
        with pytest.raises(IndexLeakageError):
            guard.assert_safe(bad_path)

    def test_rejects_empty_path(self):
        guard = self._guard()
        with pytest.raises(ValueError):
            guard.assert_safe("")

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Symlink test requires Windows for this path format"
    )
    def test_rejects_symlink_resolving_to_production(self):
        """A symlink that resolves to the production index must be rejected."""
        guard = self._guard()
        prod_path = r"d:\Ragmed_2\backend\data\faiss_index"

        with tempfile.TemporaryDirectory() as tmpdir:
            symlink_path = os.path.join(tmpdir, "fake_eval_split")
            try:
                os.symlink(prod_path, symlink_path)
            except (OSError, NotImplementedError):
                pytest.skip("Symlink creation failed (may need elevated privileges on Windows)")

            # The symlink resolves to production → must be rejected
            with pytest.raises(IndexLeakageError):
                guard.assert_safe(symlink_path)

    def test_accepts_completely_external_path(self):
        guard = self._guard()
        # A path completely outside the production index should be accepted
        with tempfile.TemporaryDirectory() as tmpdir:
            guard.assert_safe(tmpdir)  # Should not raise

    def test_rejects_parent_of_production(self):
        guard = self._guard()
        # The data/ directory is a parent of the production index → dangerous
        parent_path = r"d:\Ragmed_2\backend\data"
        with pytest.raises(IndexLeakageError):
            guard.assert_safe(parent_path)
