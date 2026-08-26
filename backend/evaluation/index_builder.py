"""MedQA index builder -- ingests JSONL dataset files into a FAISSVectorStore.

Usage (CLI)::

    cd backend
    python -m evaluation.index_builder \
        --dataset  data/datasets/medqa/sample.jsonl \
        --index-dir data/faiss_index

Usage (programmatic)::

    from evaluation.index_builder import MedQAIndexBuilder
    builder = MedQAIndexBuilder(dataset_path=Path("data/datasets/medqa/sample.jsonl"))
    store = builder.build()
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Optional

from app.rag.embeddings.bge_embedding import BGEEmbedder
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.schemas.retrieval import DocumentChunk
from app.utils.logger import logger


class MedQAIndexBuilder:
    """Ingest MedQA JSONL records into a BGE-embedded FAISS index.

    Each record is converted to a DocumentChunk where:
    - ``text``     = ``question`` field
    - ``title``    = first 80 chars of ``question``
    - ``metadata`` = answer, answer_idx, options, meta_info (as strings)

    Parameters
    ----------
    dataset_path:
        Path to a MedQA JSONL file.
    index_dir:
        Directory for index.faiss + metadata.json.
        Defaults to ``app.config.FAISS_INDEX_DIR``.
    embedder:
        Optional pre-built BGEEmbedder. Constructed from defaults if None.
    batch_size:
        Records to embed per forward pass. Default 64.
    """

    def __init__(
        self,
        dataset_path: Path,
        index_dir: Optional[Path] = None,
        embedder: Optional[BGEEmbedder] = None,
        batch_size: int = 64,
    ) -> None:
        self.dataset_path = Path(dataset_path)

        if index_dir is None:
            from app.config import FAISS_INDEX_DIR
            index_dir = FAISS_INDEX_DIR
        self.index_dir = Path(index_dir)

        self._embedder = embedder
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, force: bool = False) -> FAISSVectorStore:
        """Load records, embed them, and save the index.

        Parameters
        ----------
        force:
            Rebuild even if an index already exists. Default False.

        Returns
        -------
        FAISSVectorStore
            The populated and saved vector store.
        """
        index_file = self.index_dir / "index.faiss"
        if index_file.exists() and not force:
            logger.info(
                f"MedQAIndexBuilder: index already exists at {self.index_dir} "
                "(pass force=True to rebuild)."
            )
            return FAISSVectorStore.load(self.index_dir)

        records = self._load_records()
        if not records:
            raise ValueError(
                f"MedQAIndexBuilder: no records loaded from {self.dataset_path}"
            )

        chunks = [self._record_to_chunk(i, r) for i, r in enumerate(records)]
        texts = [c.text for c in chunks]

        embedder = self._embedder or BGEEmbedder()
        logger.info(
            f"MedQAIndexBuilder: embedding {len(texts)} records "
            f"(batch_size={self._batch_size})"
        )

        embeddings = embedder.embed_documents(texts)
        store = FAISSVectorStore(dim=embedder.dim, index_dir=self.index_dir)
        store.upsert(chunks, embeddings)
        store.save(self.index_dir)

        logger.info(
            f"MedQAIndexBuilder: saved {store.ntotal} vectors to {self.index_dir}"
        )
        return store

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"MedQAIndexBuilder: dataset not found: {self.dataset_path}"
            )
        records = []
        with open(self.dataset_path, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        f"MedQAIndexBuilder: skipping line {line_num} "
                        f"(JSONDecodeError: {exc})"
                    )
        logger.info(
            f"MedQAIndexBuilder: loaded {len(records)} records "
            f"from {self.dataset_path}"
        )
        return records

    @staticmethod
    def _record_to_chunk(index: int, record: dict) -> DocumentChunk:
        """Convert a MedQA dict to a DocumentChunk."""
        question = record.get("question", "")
        meta: dict[str, str] = {
            "answer":     str(record.get("answer", "")),
            "answer_idx": str(record.get("answer_idx", "")),
            "options":    json.dumps(record.get("options", {})),
            "meta_info":  str(record.get("meta_info", "")),
        }
        return DocumentChunk(
            chunk_id=uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"medqa_chunk_{index}_{question[:40]}",
            ),
            document_id=uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"medqa_doc_{index}_{question[:40]}",
            ),
            source_type="medqa",
            title=question[:80],
            text=question,
            chunk_index=index,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from a MedQA JSONL dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to MedQA JSONL file.")
    parser.add_argument("--index-dir", type=Path, default=None,
                        help="Output directory for index.faiss + metadata.json.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if an index already exists.")
    args = parser.parse_args()

    builder = MedQAIndexBuilder(
        dataset_path=args.dataset,
        index_dir=args.index_dir,
    )
    store = builder.build(force=args.force)
    print(f"Index built: {store.ntotal} vectors at {builder.index_dir}")


if __name__ == "__main__":
    _main()
