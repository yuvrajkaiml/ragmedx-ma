from pathlib import Path

from app.rag.loaders.pubmed_loader import PubMedLoader


def main():

    dataset_path = Path("data/datasets/pubmed")

    loader = PubMedLoader(dataset_path)

    documents = loader.load()

    print(f"\nTotal documents loaded: {len(documents)}")

    if documents:

        print("\nFirst document:")
        print(documents[0])

        print("\nID:")
        print(documents[0].id)

        print("\nTitle:")
        print(documents[0].title)

        print("\nJournal:")
        print(documents[0].journal)

        print("\nPublication year:")
        print(documents[0].publication_year)

        print("\nSource:")
        print(documents[0].source)

        print("\nMetadata:")
        print(documents[0].metadata)


if __name__ == "__main__":
    main()