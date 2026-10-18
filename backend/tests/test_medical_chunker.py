from pathlib import Path

from app.rag.loaders.pmc_loader import PMCLoader
from app.rag.chunking.semantic_chunker import MedicalChunker

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
MIN_CHUNK_SIZE = 150


def main():
    dataset_path = Path("data/datasets/pmc")
    loader = PMCLoader(dataset_path)
    documents = loader.load()

    print(f"\nTotal documents loaded: {len(documents)}")

    if not documents:
        print("No PMC documents found.")
        return

    chunker = MedicalChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
    )

    chunks = chunker.chunk_documents(documents)
    print(f"\nTotal chunks created: {len(chunks)}")

    if not chunks:
        raise AssertionError("Chunker returned no chunks.")

    oversized = [
        (chunk.id, len(chunk.content))
        for chunk in chunks
        if len(chunk.content) > CHUNK_SIZE
    ]

    missing_ids = [
        chunk.id for chunk in chunks
        if not chunk.id or not chunk.document_id
    ]

    if oversized:
        raise AssertionError(f"Chunks exceed size limit: {oversized}")

    if missing_ids:
        raise AssertionError(f"Chunks have missing IDs: {missing_ids}")

    print("\nBasic chunk validation passed.")
    print(f"Maximum chunk size: {max(len(c.content) for c in chunks)}")
    print(f"Configured overlap: {CHUNK_OVERLAP}")

    for index, chunk in enumerate(chunks[:5], start=1):
        print("\n" + "=" * 80)
        print(f"Chunk {index}: {chunk.id}")
        print(f"Document ID: {chunk.document_id}")
        print(f"Title: {chunk.title}")
        print(f"Section: {chunk.section}")
        print(f"Journal: {chunk.journal}")
        print(f"Publication year: {chunk.publication_year}")
        print(f"Character count: {len(chunk.content)}")
        print(f"\nContent:\n{chunk.content}")

    print("\nPMC chunking demonstration completed successfully.")


if __name__ == "__main__":
    main()