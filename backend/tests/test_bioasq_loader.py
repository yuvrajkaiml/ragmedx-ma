from pathlib import Path

from app.rag.loaders.bioasq_loader import BioASQLoader


def main():

    dataset_path = Path("data/datasets/bioasq")

    loader = BioASQLoader(dataset_path)

    documents = loader.load()

    print(f"\nTotal documents loaded: {len(documents)}")

    if documents:

        print("\nFirst document:")
        print(documents[0])

        print("\nSource:")
        print(documents[0].source)

        print("\nMetadata keys:")
        print(documents[0].metadata.keys())


if __name__ == "__main__":
    main()