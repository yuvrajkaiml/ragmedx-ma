from pathlib import Path

from app.rag.loaders.medqa_loader import MedQALoader


loader = MedQALoader(
    Path("data/datasets/medqa")
)

documents = loader.load()

print(f"\nTotal documents loaded: {len(documents)}")

# Basic validation
assert len(documents) > 0, "No documents were loaded."

for document in documents:
    assert document.id, "Document ID is missing."
    assert document.title, "Document title is missing."
    assert document.content, "Document content is missing."
    assert document.source == "MedQA", "Incorrect document source."
    assert document.metadata is not None, "Document metadata is missing."
    assert "answer" in document.metadata, "Answer missing from metadata."
    assert "options" in document.metadata, "Options missing from metadata."

print("\n✓ All documents passed validation.")
print("✓ Every document has an ID.")
print("✓ Every document has a title.")
print("✓ Every document has content.")
print("✓ Every document has source = MedQA.")
print("✓ Every document contains answer and options metadata.")