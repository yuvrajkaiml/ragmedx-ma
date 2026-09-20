from pathlib import Path

from app.rag.loaders.pmc_loader import PMCLoader


def main():
    dataset_path = Path("data/datasets/pmc")

    loader = PMCLoader(dataset_path)

    documents = loader.load()

    print(f"\nTotal documents loaded: {len(documents)}")

    if documents:
        first = documents[0]

        print("\nFirst document:")
        print(first)

        print("\nID:")
        print(first.id)

        print("\nTitle:")
        print(first.title)

        print("\nJournal:")
        print(first.journal)

        print("\nPublication year:")
        print(first.publication_year)

        print("\nSource:")
        print(first.source)

        print("\nContent preview:")
        print(first.content[:500])

        print("\nMetadata:")
        print(first.metadata)


if __name__ == "__main__":
    main()