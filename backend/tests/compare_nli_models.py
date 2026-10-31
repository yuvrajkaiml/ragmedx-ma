"""Compare two real NLI models on the same labelled sentence pairs.

The dataset is synthetic and intended for initial diagnostics only.
It is not a clinical benchmark.
"""

import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODELS = [
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-base",
]

LABELS = {"contradiction", "entailment", "neutral"}

# Additional controlled examples to complement the existing JSON dataset.
EXTRA_CASES = [
    {
        "id": "extra_001",
        "premise": "The patient has a fever.",
        "hypothesis": "The patient owns a blue bicycle.",
        "expected_label": "neutral",
    },
    {
        "id": "extra_002",
        "premise": "The patient has a fever.",
        "hypothesis": "The patient does not have a fever.",
        "expected_label": "contradiction",
    },
    {
        "id": "extra_003",
        "premise": "A man is eating pizza.",
        "hypothesis": "A man is eating food.",
        "expected_label": "entailment",
    },
    {
        "id": "extra_004",
        "premise": "The trial enrolled adults with type 2 diabetes.",
        "hypothesis": "The trial enrolled adults.",
        "expected_label": "entailment",
    },
    {
        "id": "extra_005",
        "premise": "The study measured blood pressure.",
        "hypothesis": "The study measured kidney function.",
        "expected_label": "neutral",
    },
    {
        "id": "extra_006",
        "premise": "Treatment X reduced pain scores compared with placebo.",
        "hypothesis": "Treatment X did not reduce pain scores compared with placebo.",
        "expected_label": "contradiction",
    },
]


def load_cases():
    data_path = Path(__file__).resolve().parent / "data" / "medical_nli_eval.json"

    with data_path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    cases = cases + EXTRA_CASES

    for case in cases:
        if case["expected_label"] not in LABELS:
            raise ValueError(
                f"Unsupported label in {case['id']}: "
                f"{case['expected_label']}"
            )

    return cases


def normalize_label(label):
    label = label.strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "contradiction": "contradiction",
        "contradictory": "contradiction",
        "entailment": "entailment",
        "entailed": "entailment",
        "neutral": "neutral",
    }

    if label not in aliases:
        raise ValueError(f"Unrecognized model label: {label!r}")

    return aliases[label]


def get_label_mapping(model):
    """Read and validate the model's label configuration."""
    mapping = {}

    for index, label in model.config.id2label.items():
        normalized = normalize_label(str(label))
        mapping[int(index)] = normalized

    if set(mapping.values()) != LABELS:
        raise ValueError(
            f"Expected the three NLI labels, got {mapping}"
        )

    return mapping


def evaluate_model(model_name, cases):
    print(f"\n{'=' * 90}")
    print(f"MODEL: {model_name}")
    print("Loading model on CPU...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    mapping = get_label_mapping(model)

    expected = []
    predicted = []
    scores = []

    with torch.no_grad():
        for case in cases:
            inputs = tokenizer(
                case["premise"],
                case["hypothesis"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            logits = model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

            predicted_index = int(torch.argmax(probabilities).item())
            predicted_label = mapping[predicted_index]
            confidence = float(probabilities[predicted_index].item())

            expected_label = case["expected_label"]

            expected.append(expected_label)
            predicted.append(predicted_label)
            scores.append(confidence)

            result = "MATCH" if predicted_label == expected_label else "MISMATCH"

            print(
                f"{case['id']:<14} "
                f"expected={expected_label:<13} "
                f"predicted={predicted_label:<13} "
                f"score={confidence:.4f} {result}"
            )

    accuracy = sum(
        actual == guess
        for actual, guess in zip(expected, predicted)
    ) / len(expected)

    print(f"\nAccuracy: {accuracy:.4f} ({sum(a == p for a, p in zip(expected, predicted))}/{len(expected)})")

    for label in sorted(LABELS):
        tp = sum(a == label and p == label for a, p in zip(expected, predicted))
        fp = sum(a != label and p == label for a, p in zip(expected, predicted))
        fn = sum(a == label and p != label for a, p in zip(expected, predicted))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        print(
            f"{label:<14} precision={precision:.3f} "
            f"recall={recall:.3f} F1={f1:.3f}"
        )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "predictions": list(zip(expected, predicted)),
    }


def main():
    cases = load_cases()
    print(f"Total synthetic sentence pairs: {len(cases)}")
    print("These results are diagnostic, not clinical validation.")

    results = []

    for model_name in MODELS:
        results.append(evaluate_model(model_name, cases))

    print(f"\n{'=' * 90}")
    print("MODEL COMPARISON")

    for result in results:
        print(
            f"{result['model']}: "
            f"accuracy={result['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()