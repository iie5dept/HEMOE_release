from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ("real", "fake")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tri-modal classifier JSONL predictions.")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    confusion = {label: {prediction: 0 for prediction in LABELS} for label in LABELS}
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            label = str(record["label"]).lower()
            prediction = str(record["prediction"]).lower()
            if label not in LABELS or prediction not in LABELS:
                raise ValueError(f"Invalid label at line {line_number}: {label!r}, {prediction!r}")
            confusion[label][prediction] += 1
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in LABELS)
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }
    payload = {
        "accuracy": correct / max(1, total),
        "macro_precision": sum(item["precision"] for item in per_class.values()) / len(LABELS),
        "macro_recall": sum(item["recall"] for item in per_class.values()) / len(LABELS),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(LABELS),
        "total": total,
        "confusion": confusion,
        "per_class": per_class,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
