from pathlib import Path

from app.rag.loaders.pmc_loader import PMCLoader
from app.rag.chunking.semantic_chunker import MedicalChunker


def main():
    dataset_path = Path("data/datasets/pmc")

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    loader = PMCLoader(dataset_path)

    documents = loader.load()

    print(
        f"\nTotal documents loaded: "
        f"{len(documents)}"
    )

    if not documents:
        print("No documents found.")
        return

    # ---------------------------------------------------------
    # CHUNK
    # ---------------------------------------------------------

    chunker = MedicalChunker(
        chunk_size=800,
        chunk_overlap=100,
        min_chunk_size=150,
    )

    chunks = chunker.chunk_documents(
        documents
    )

    print(
        f"\nTotal chunks created: "
        f"{len(chunks)}"
    )

    # ---------------------------------------------------------
    # DISPLAY FIRST 5 CHUNKS
    # ---------------------------------------------------------

    for index, chunk in enumerate(
        chunks[:5]
    ):
        print("\n" + "=" * 80)

        print(
            f"Chunk {index + 1}"
        )

        print(
            f"ID: {chunk.id}"
        )

        print(
            f"Document ID: "
            f"{chunk.document_id}"
        )

        print(
            f"Title: "
            f"{chunk.title}"
        )

        print(
            f"Section: "
            f"{chunk.section}"
        )

        print(
            f"Journal: "
            f"{chunk.journal}"
        )

        print(
            f"Publication year: "
            f"{chunk.publication_year}"
        )

        print(
            f"\nContent:\n"
            f"{chunk.content}"
        )

        print(
            f"\nCharacter count: "
            f"{len(chunk.content)}"
        )


if __name__ == "__main__":
    main()