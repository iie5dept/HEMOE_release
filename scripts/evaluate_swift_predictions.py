from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LABEL_ORDER = ("real", "fake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Swift infer JSONL predictions.")
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def normalize_label(text: str | None) -> str:
    value = (text or "").strip().lower()
    if re.search(r"\breal\b", value):
        return "real"
    if re.search(r"\bfake\b", value):
        return "fake"
    if "真" in value or "真实" in value:
        return "real"
    if "假" in value or "辟谣" in value or "谣言" in value:
        return "fake"
    return "fake"


def extract_text(record: dict) -> str:
    for key in ("response", "prediction", "predict", "output", "generated_text", "text"):
        value = record.get(key)
        if isinstance(value, str):
            return value

    choices = record.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

    messages = record.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "assistant" and isinstance(item.get("content"), str):
                return item["content"]

    return ""


def extract_video_key(record: dict[str, Any]) -> str | None:
    videos = record.get("videos")
    if isinstance(videos, list) and videos:
        first = videos[0]
        if isinstance(first, str):
            return Path(first).name
    return None


def extract_user_key(record: dict[str, Any]) -> str | None:
    messages = record.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str):
                return item["content"].strip()
    return None


def load_gold(dataset_jsonl: Path) -> tuple[list[dict[str, str]], list[str]]:
    records: list[dict[str, str]] = []
    labels: list[str] = []
    with dataset_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                {
                    "id": str(row.get("id", len(records))),
                    "video_key": extract_video_key(row) or "",
                    "user_key": extract_user_key(row) or "",
                }
            )
            assistant = ""
            for message in row.get("messages", []):
                if message.get("role") == "assistant":
                    assistant = str(message.get("content", ""))
            labels.append(normalize_label(assistant))
    return records, labels


def load_predictions(prediction_jsonl: Path, gold_records: list[dict[str, str]]) -> list[str]:
    by_id: dict[str, str] = {}
    by_video: dict[str, str] = {}
    by_user: dict[str, str] = {}
    ordered: list[str] = []
    raw_rows: list[dict[str, Any]] = []
    with prediction_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_rows.append(row)
            pred = normalize_label(extract_text(row))
            row_id = row.get("id")
            if row_id is not None:
                by_id[str(row_id)] = pred
            video_key = extract_video_key(row)
            if video_key:
                by_video[video_key] = pred
            user_key = extract_user_key(row)
            if user_key:
                by_user[user_key] = pred
            ordered.append(pred)

    preds: list[str] = []
    for index, gold_record in enumerate(gold_records):
        gold_id = gold_record["id"]
        video_key = gold_record["video_key"]
        user_key = gold_record["user_key"]
        if gold_id in by_id:
            preds.append(by_id[gold_id])
        elif video_key and video_key in by_video:
            preds.append(by_video[video_key])
        elif user_key and user_key in by_user:
            preds.append(by_user[user_key])
        elif index < len(ordered):
            # Last-resort fallback only when no stable key exists.
            preds.append(ordered[index])
        else:
            preds.append("fake")
    return preds


def load_direct_predictions_with_labels(prediction_jsonl: Path) -> tuple[list[str], list[str]] | None:
    gold: list[str] = []
    pred: list[str] = []
    saw_label = False

    with prediction_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "labels" not in row:
                return None
            saw_label = True
            gold.append(normalize_label(str(row.get("labels", ""))))
            pred.append(normalize_label(extract_text(row)))

    if not saw_label:
        return None
    return gold, pred


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(gold: list[str], pred: list[str]) -> dict[str, float]:
    total = len(gold)
    correct = sum(int(g == p) for g, p in zip(gold, pred))
    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for label in LABEL_ORDER:
        tp = sum(int(g == label and p == label) for g, p in zip(gold, pred))
        fp = sum(int(g != label and p == label) for g, p in zip(gold, pred))
        fn = sum(int(g == label and p != label) for g, p in zip(gold, pred))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(int(g == label) for g in gold)}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": safe_div(correct, total),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "total": total,
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    direct = load_direct_predictions_with_labels(args.prediction_jsonl)
    if direct is not None:
        gold_labels, pred_labels = direct
    else:
        if args.dataset_jsonl is None:
            raise ValueError(
                "--dataset-jsonl is required when prediction-jsonl does not contain top-level `labels`."
            )
        gold_records, gold_labels = load_gold(args.dataset_jsonl)
        pred_labels = load_predictions(args.prediction_jsonl, gold_records)
    metrics = compute_metrics(gold_labels, pred_labels)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
