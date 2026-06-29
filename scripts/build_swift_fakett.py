from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build FakeTT JSONL files for ms-swift SFT.")
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=repo_root / "data" / "fakett" / "data.json",
        help="Path to FakeTT data.json annotation file.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=repo_root / "data" / "fakett" / "video",
        help="Directory containing FakeTT mp4 files.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=repo_root / "external" / "ExMRD" / "data" / "FakeTT" / "vids",
        help="Directory containing ExMRD temporal split txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "swift" / "fakett",
        help="Directory to save fakett_train/val/test.jsonl.",
    )
    parser.add_argument(
        "--video-extension",
        type=str,
        default=".mp4",
        help="Video extension used to compose each sample path.",
    )
    parser.add_argument(
        "--strict-video-check",
        action="store_true",
        help="Raise an error if any referenced video file is missing.",
    )
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


def build_user_prompt(record: dict) -> str:
    parts: List[str] = [
        "Please judge whether this short news video is real or fake.",
    ]
    description = str(record.get("description") or "").strip()
    if description:
        parts.append(f"Description: {description}")
    event = str(record.get("event") or "").strip()
    if event:
        parts.append(f"Claimed event: {event}")
    user_description = str(record.get("user_description") or "").strip()
    if user_description:
        parts.append(f"Uploader profile: {user_description}")
    parts.append("Reply with exactly one word: real or fake.")
    return "\n".join(parts)


def build_jsonl_record(record: dict, video_path: Path) -> dict:
    label = str(record["annotation"]).strip().lower()
    if label not in {"real", "fake"}:
        raise ValueError(f"Unsupported label {label!r} for video_id={record['video_id']}.")
    return {
        "id": str(record["video_id"]),
        "messages": [
            {
                "role": "user",
                "content": build_user_prompt(record),
            },
            {
                "role": "assistant",
                "content": label,
            },
        ],
        "videos": [str(video_path.resolve(strict=False))],
    }


def iter_split_records(
    sample_ids: Iterable[str],
    annotations: Dict[str, dict],
    video_root: Path,
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

        video_path = video_root / f"{sample_id}{video_extension}"
        if strict_video_check and not video_path.exists():
            missing_video_paths.append(str(video_path))
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
        print(f"{split_name}: wrote {count} samples -> {output_path}")


if __name__ == "__main__":
    main()
