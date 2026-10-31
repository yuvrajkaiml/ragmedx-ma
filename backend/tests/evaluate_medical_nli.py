"""Evaluate the real NLI model on a labelled medical sentence-pair dataset."""

import json
from pathlib import Path

from app.rag.verification.contradiction_detector import (
    MedicalContradictionDetector,
)


LABELS = ("contradiction", "entailment", "neutral")


def load_dataset(path: Path) -> list[dict]:
    """Load and validate labelled sentence pairs."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError("The evaluation dataset must be a non-empty JSON list.")

    required_fields = {
        "id",
        "premise",
        "hypothesis",
        "expected_label",
    }

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be a JSON object.")

        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"Dataset item {index} is missing fields: {sorted(missing)}"
            )

        if not isinstance(item["premise"], str) or not item["premise"].strip():
            raise ValueError(f"Dataset item {index} has an empty premise.")

        if not isinstance(item["hypothesis"], str) or not item["hypothesis"].strip():
            raise ValueError(f"Dataset item {index} has an empty hypothesis.")

        if item["expected_label"] not in LABELS:
            raise ValueError(
                f"Dataset item {index} has an unsupported expected label."
            )

    return data


def calculate_metrics(expected: list[str], predicted: list[str]) -> dict:
    """Calculate accuracy and per-class precision, recall, and F1."""
    if not expected:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    results = {}

    for label in LABELS:
        tp = sum(
            actual == label and guess == label
            for actual, guess in zip(expected, predicted)
        )
        fp = sum(
            actual != label and guess == label
            for actual, guess in zip(expected, predicted)
        )
        fn = sum(
            actual == label and guess != label
            for actual, guess in zip(expected, predicted)
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(actual == label for actual in expected),
        }

    accuracy = sum(
        actual == guess
        for actual, guess in zip(expected, predicted)
    ) / len(expected)

    results["accuracy"] = accuracy
    return results


def print_confusion_matrix(
    expected: list[str],
    predicted: list[str],
) -> None:
    """Print a confusion matrix with actual labels as rows."""
    print("\nConfusion matrix (rows = actual, columns = predicted)")
    print(f"{'Actual / Predicted':<22}" + "".join(f"{label:>16}" for label in LABELS))

    for actual_label in LABELS:
        row = []
        for predicted_label in LABELS:
            count = sum(
                actual == actual_label and guess == predicted_label
                for actual, guess in zip(expected, predicted)
            )
            row.append(count)

        print(f"{actual_label:<22}" + "".join(f"{count:>16}" for count in row))


def main() -> None:
    dataset_path = Path(__file__).resolve().parent / "data" / "medical_nli_eval.json"
    dataset = load_dataset(dataset_path)

    print(f"Dataset: {dataset_path}")
    print(f"Sentence pairs: {len(dataset)}")
    print("Loading real NLI model on CPU...")

    detector = MedicalContradictionDetector(
        device=-1,
        contradiction_threshold=0.80,
    )

    expected_labels = []
    predicted_labels = []

    print("\nIndividual predictions")
    print("-" * 80)

    for item in dataset:
        predicted_label, score = detector.predict_pair(
            item["premise"],
            item["hypothesis"],
        )

        predicted_label = predicted_label.strip().lower()
        if predicted_label not in LABELS:
            raise ValueError(
                f"Unexpected model label for {item['id']}: {predicted_label!r}"
            )

        expected_label = item["expected_label"]
        expected_labels.append(expected_label)
        predicted_labels.append(predicted_label)

        status = "MATCH" if predicted_label == expected_label else "MISMATCH"

        print(
            f"{item['id']} | expected={expected_label:<13} "
            f"predicted={predicted_label:<13} "
            f"score={score:.4f} | {status}"
        )

    metrics = calculate_metrics(expected_labels, predicted_labels)

    print("\nEvaluation metrics")
    print("-" * 80)
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    for label in LABELS:
        result = metrics[label]
        print(
            f"{label:<14} "
            f"precision={result['precision']:.4f}  "
            f"recall={result['recall']:.4f}  "
            f"F1={result['f1']:.4f}  "
            f"support={result['support']}"
        )

    print_confusion_matrix(expected_labels, predicted_labels)

    print("\nNote: this is a small synthetic pilot dataset, not a clinical benchmark.")


if __name__ == "__main__":
    main()