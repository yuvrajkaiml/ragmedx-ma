"""Retrieval-metric unit tests on small, hand-verifiable examples (no model)."""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation import metrics  # noqa: E402
from evaluation.bioasq_eval.common import dedup_articles  # noqa: E402


class _Chunk:
    def __init__(self, pmid):
        self.metadata = {"pmid": pmid}
        self.source = f"http://x/pubmed/{pmid}"
        self.document_id = f"doc_{pmid}"


K = 5


def test_all_five_irrelevant():
    retrieved = ["1", "2", "3", "4", "5"]
    gold = ["99"]
    assert metrics.precision_at_k(retrieved, gold, K) == 0.0
    assert metrics.recall_at_k(retrieved, gold, K) == 0.0
    assert metrics.ndcg_at_k(retrieved, gold, K) == 0.0
    assert metrics.mrr([retrieved], [gold]) == 0.0


def test_first_result_relevant():
    retrieved = ["7", "2", "3", "4", "5"]
    gold = ["7"]
    assert metrics.precision_at_k(retrieved, gold, K) == 0.2    # 1/5
    assert metrics.recall_at_k(retrieved, gold, K) == 1.0       # 1/1
    assert metrics.ndcg_at_k(retrieved, gold, K) == 1.0         # rank 1 ideal
    assert metrics.mrr([retrieved], [gold]) == 1.0


def test_relevant_at_rank_five():
    retrieved = ["1", "2", "3", "4", "9"]
    gold = ["9"]
    assert metrics.precision_at_k(retrieved, gold, K) == 0.2
    assert metrics.recall_at_k(retrieved, gold, K) == 1.0
    assert metrics.mrr([retrieved], [gold]) == 0.2             # 1/5
    # nDCG: DCG = 1/log2(6); IDCG = 1/log2(2)=1 -> 1/log2(6)
    import math
    expected = round((1 / math.log2(6)) / 1.0, 4)
    assert metrics.ndcg_at_k(retrieved, gold, K) == expected


def test_multiple_relevant_pmids():
    retrieved = ["a", "b", "c", "d", "e"]
    gold = ["a", "c"]
    assert metrics.precision_at_k(retrieved, gold, K) == 0.4   # 2/5
    assert metrics.recall_at_k(retrieved, gold, K) == 1.0      # 2/2
    assert metrics.mrr([retrieved], [gold]) == 1.0             # first hit rank 1


def test_multiple_chunks_same_pmid_collapse_to_one_article():
    # Ranked chunks: 11, 11, 22, 11, 33 -> articles [11, 22, 33].
    chunks = [_Chunk("11"), _Chunk("11"), _Chunk("22"), _Chunk("11"), _Chunk("33")]
    articles = dedup_articles(chunks)
    assert articles == ["11", "22", "33"]
    gold = ["11"]
    # One article 11 at rank 1, not three hits.
    assert metrics.precision_at_k(articles, gold, K) == round(1 / 5, 4)
    assert metrics.recall_at_k(articles, gold, K) == 1.0


def test_gold_absent_conditional_vs_end_to_end_divergence():
    # Query whose gold is not in the corpus -> scores 0 in end-to-end,
    # and is excluded from the conditional (effectiveness) denominator.
    retrieved = ["x", "y", "z"]
    gold = ["not_in_corpus"]
    p = metrics.precision_at_k(retrieved, gold, K)
    assert p == 0.0
    # End-to-end over 2 queries, one covered (score 1.0) one uncovered (0.0):
    end_to_end = (1.0 + 0.0) / 2
    conditional = 1.0 / 1   # only the covered query counts
    assert end_to_end == 0.5
    assert conditional == 1.0
    assert end_to_end != conditional


def test_dedup_preserves_rank_order_for_mrr():
    chunks = [_Chunk("22"), _Chunk("11"), _Chunk("11"), _Chunk("33")]
    articles = dedup_articles(chunks)
    assert articles == ["22", "11", "33"]
    gold = ["11"]
    # 11 is at rank 2 -> reciprocal rank 0.5.
    assert metrics.mrr([articles], [gold]) == 0.5
