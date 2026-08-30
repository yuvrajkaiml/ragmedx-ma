"""Cross-encoder reranker.

Takes the top-K candidates from a first-stage retriever (dense, sparse, or
hybrid) and re-scores them with a cross-encoder model that jointly encodes
(query, passage) pairs for higher precision.

Model
-----
Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
A 6-layer MiniLM fine-tuned on MS MARCO passage ranking.

Score contract
--------------
`CrossEncoder.predict` with `activation_fn=torch.sigmoid` returns scores
bounded in `(0, 1)`, satisfying the schema's `Field(ge=0.0, le=1.0)`
constraint without any post-processing.

Import safety
-------------
`CrossEncoder` is imported from `sentence_transformers.cross_encoder`
(not the top-level `sentence_transformers` package) to avoid triggering the
`datasets` -> `pyarrow._dataset` import chain.

Usage::

    from app.rag.rerankers.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked  = reranker.rerank(query, results)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from sentence_transformers.cross_encoder import CrossEncoder

from app.schemas.retrieval import RetrievalResult
from app.utils.logger import logger

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_MAX_LENGTH = 512


class CrossEncoderReranker:
    """Re-rank first-stage retrieval results using a cross-encoder model.

    Parameters
    ----------
    model_name:
        HuggingFace model id or local path.  Defaults to
        `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    device:
        `"cpu"`, `"cuda"`, or `None` (auto-detect).
    max_length:
        Maximum token length passed to the tokeniser.  Default 512.
    batch_size:
        Number of (query, passage) pairs per forward pass.  Default 32.
    cache_folder:
        Optional local cache directory for model weights.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = _DEFAULT_MAX_LENGTH,
        batch_size: int = 32,
        cache_folder: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or _DEFAULT_MODEL
        self.batch_size = batch_size

        logger.info(
            f"CrossEncoderReranker: loading '{self.model_name}' "
            f"(device={device or 'auto'})"
        )

        self._model = CrossEncoder(
            self.model_name,
            max_length=max_length,
            device=device,
            cache_folder=cache_folder,
        )

        logger.info(f"CrossEncoderReranker ready — model='{self.model_name}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Re-score *results* against *query* and return sorted by score.

        Parameters
        ----------
        query:
            The original user query string.
        results:
            Candidate `RetrievalResult` objects from a first-stage
            retriever.  Each result's `chunk.text` is used as the passage.
        top_k:
            Optional cap on results returned.  `None` returns all.

        Returns
        -------
        list[RetrievalResult]
            New `RetrievalResult` objects with `relevance_score` set to
            the sigmoid cross-encoder score, ordered descending.
            Empty list if *results* is empty.

        Raises
        ------
        ValueError
            If *query* is empty / whitespace-only, or *top_k* is not positive.
        """
        if not query or not query.strip():
            raise ValueError(
                "CrossEncoderReranker.rerank requires a non-empty query string."
            )
        if top_k is not None and top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {top_k}."
            )

        if not results:
            logger.debug("CrossEncoderReranker.rerank: empty candidates — returning [].")
            return []

        pairs = [(query, r.chunk.text) for r in results]

        logger.debug(
            f"CrossEncoderReranker.rerank: "
            f"query='{query[:60]}{'...' if len(query) > 60 else ''}', "
            f"candidates={len(pairs)}"
        )

        raw_scores: np.ndarray = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            activation_fn=torch.sigmoid,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        reranked: list[RetrievalResult] = []
        for result, score in zip(results, raw_scores):
            clamped = float(max(0.0, min(1.0, float(score))))
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    relevance_score=clamped,
                )
            )

        reranked.sort(key=lambda r: r.relevance_score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        if reranked:
            logger.debug(
                f"CrossEncoderReranker.rerank: returned {len(reranked)} result(s), "
                f"top score={reranked[0].relevance_score:.4f}"
            )
        return reranked