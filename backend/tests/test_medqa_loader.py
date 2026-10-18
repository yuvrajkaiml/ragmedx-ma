
import json
from pathlib import Path

from app.rag.loaders.medqa_loader import MedQALoader


def test_medqa_loader_loads_jsonl_documents(tmp_path):
    dataset_file = tmp_path / "sample.jsonl"

    record = {
        "question": "What is hypertension?",
        "answer": "Persistently elevated blood pressure",
        "answer_idx": "A",
        "options": {"A": "Elevated blood pressure"},
        "meta_info": "medicine",
    }

    dataset_file.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    loader = MedQALoader(tmp_path)
    documents = loader.load()

    assert len(documents) == 1

    document = documents[0]
    assert document.id == "sample_0"
    assert document.title == record["question"]
    assert document.content == record["question"]
    assert document.source == "MedQA"
    assert document.metadata["answer"] == record["answer"]
    assert document.metadata["options"] == record["options"]


def test_medqa_loader_skips_blank_lines(tmp_path):
    dataset_file = tmp_path / "sample.jsonl"
    dataset_file.write_text(
        '\n{"question": "What is diabetes?", "answer": "A disease"}\n\n',
        encoding="utf-8",
    )

    documents = MedQALoader(tmp_path).load()

    assert len(documents) == 1
    assert documents[0].content == "What is diabetes?"


def test_medqa_loader_returns_empty_list_for_empty_directory(tmp_path):
    documents = MedQALoader(tmp_path).load()

    assert documents == []
