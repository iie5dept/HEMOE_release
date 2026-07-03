from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


FAKE_NEWS_PROMPT = (
    "You are a video fake-news detection assistant. Determine whether the news claim conveyed by "
    "this short video and the auxiliary observations is real or fake.\n\n"
    "This task is short-video news veracity detection, not deepfake detection. Judge whether the "
    "news claim or event conveyed by the video, caption, and metadata is factually true as stated.\n\n"
    "A sample should be labeled fake if the conveyed news is verifiably false, debunked, misleading "
    "in context, old footage presented as a new event, mismatched with the claimed time, place, "
    "person, or event, or uses unrelated or repurposed footage to make a false news claim appear true.\n\n"
    "A sample should be labeled real if the conveyed news content is factually true and the video "
    "plus accompanying context do not misleadingly distort the event.\n\n"
    "Do not decide based only on uploader identity, political stance, hashtags, emotional tone, or "
    "visual authenticity. Use them only as auxiliary context.\n\n"
    "Reply with exactly one word: real or fake."
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build pure real/fake Swift SFT jsonl for FakeTT.")
    parser.add_argument("--annotation-path", type=Path, default=repo_root / "data" / "fakett" / "data.json")
    parser.add_argument("--split-dir", type=Path, default=repo_root / "external" / "ExMRD" / "data" / "FakeTT" / "vids")
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "swift" / "fakett")
    parser.add_argument("--video-extension", type=str, default=".mp4")
    parser.add_argument("--strict-video-check", action="store_true")
    return parser.parse_args()


def load_annotations(annotation_path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows[str(record["video_id"])] = record
    if not rows:
        raise ValueError(f"No records loaded from {annotation_path}.")
    return rows


def load_split_ids(split_path: Path) -> List[str]:
    ids: List[str] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample_id = line.strip()
            if sample_id:
                ids.append(sample_id)
    if not ids:
        raise ValueError(f"No sample ids found in split file: {split_path}")
    return ids


def format_publish_time(value) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = float(value)
        if timestamp > 1e12:
            timestamp /= 1000.0
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(value)


def normalize_label(record: dict) -> str:
    label = str(record.get("annotation") or "").strip().lower()
    if label not in {"real", "fake"}:
        raise ValueError(f"Unsupported FakeTT label: {label!r}")
    return label


def build_video_path(video_root: str, sample_id: str, video_extension: str) -> str:
    normalized_root = video_root.rstrip("/\\")
    normalized_root = normalized_root.replace("\\", "/")
    return f"{normalized_root}/{sample_id}{video_extension}"


def build_user_prompt(record: dict) -> str:
    parts: List[str] = ["<video>", FAKE_NEWS_PROMPT, "", "Auxiliary observations:"]

    event = str(record.get("event") or "").strip()
    if event:
        parts.append(f"- news claim to verify: {event}")

    description = str(record.get("description") or "").strip()
    if description:
        parts.append(f"- uploader caption: {description}")

    user_description = str(record.get("user_description") or "").strip()
    if user_description:
        parts.append(f"- uploader profile: {user_description}")

    publish_time = format_publish_time(record.get("publish_time"))
    if publish_time:
        parts.append(f"- publish time metadata: {publish_time}")

    return "\n".join(parts)


def build_jsonl_record(record: dict, video_path: str) -> dict:
    return {
        "id": str(record["video_id"]),
        "messages": [
            {"role": "user", "content": build_user_prompt(record)},
            {"role": "assistant", "content": normalize_label(record)},
        ],
        "videos": [video_path],
    }


def iter_split_records(
    sample_ids: Iterable[str],
    annotations: Dict[str, dict],
    video_root: str,
    video_extension: str,
    strict_video_check: bool,
) -> Iterable[dict]:
    missing_annotation_ids: List[str] = []
    missing_video_paths: List[str] = []

    for sample_id in sample_ids:
        record = annotations.get(sample_id)
        if record is None:
            missing_annotation_ids.append(sample_id)
            continue

        video_path = build_video_path(video_root, sample_id, video_extension)
        if strict_video_check and not Path(video_path).exists():
            missing_video_paths.append(video_path)
            continue

        yield build_jsonl_record(record, video_path)

    if missing_annotation_ids:
        preview = ", ".join(missing_annotation_ids[:5])
        raise ValueError(f"Missing annotations for {len(missing_annotation_ids)} ids. Examples: {preview}")
    if missing_video_paths:
        preview = ", ".join(missing_video_paths[:3])
        raise ValueError(f"Missing video files for {len(missing_video_paths)} samples. Examples: {preview}")


def dump_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotation_path)

    split_mapping = {
        "train": args.split_dir / "vid_time3_train.txt",
        "val": args.split_dir / "vid_time3_valid.txt",
        "test": args.split_dir / "vid_time3_test.txt",
    }

    for split_name, split_path in split_mapping.items():
        sample_ids = load_split_ids(split_path)
        output_path = args.output_dir / f"fakett_{split_name}.jsonl"
        rows = iter_split_records(
            sample_ids,
            annotations,
            args.video_root,
            args.video_extension,
            args.strict_video_check,
        )
        count = dump_jsonl(output_path, rows)
        print(json.dumps({"split": split_name, "output": str(output_path), "count": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
