from pathlib import Path

from app.rag.loaders.pdf_loader import PDFLoader


def main():

    loader = PDFLoader()

    docs = loader.load(
        Path("data/documents/lab_report.pdf")
    )

    print("=" * 80)
    print(f"Pages Loaded: {len(docs)}")
    print("=" * 80)

    print(docs[0].metadata)

    print("=" * 80)

    print(docs[0].text[:800])


if __name__ == "__main__":
    main()