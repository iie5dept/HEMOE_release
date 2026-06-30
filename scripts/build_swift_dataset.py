from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


DATASET_SETTINGS = {
    "fakett": {
        "annotation_path": Path("data/fakett/data.json"),
        "split_dir": Path("external/ExMRD/data/FakeTT/vids"),
        "output_dir": Path("data/swift/fakett"),
        "video_extension": ".mp4",
    },
    "fakesv": {
        "annotation_path": Path("data/fakesv/data_complete.json"),
        "split_dir": Path("external/ExMRD/data/FakeSV/vids"),
        "output_dir": Path("data/swift/fakesv"),
        "video_extension": ".mp4",
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build Swift JSONL files for FakeTT/FakeSV.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SETTINGS.keys()), required=True)
    parser.add_argument("--annotation-path", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--video-extension", type=str, default=None)
    parser.add_argument("--strict-video-check", action="store_true")
    parser.add_argument(
        "--fakesv-refute-policy",
        choices=["drop", "map_to_fake"],
        default="drop",
        help="How to handle FakeSV '辟谣' labels.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root for resolving default relative paths.",
    )
    return parser.parse_args()


def resolve_path(base: Path, maybe_relative: Path | None) -> Path | None:
    if maybe_relative is None:
        return None
    return maybe_relative if maybe_relative.is_absolute() else (base / maybe_relative)


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


def normalize_label(dataset: str, record: dict, fakesv_refute_policy: str) -> str | None:
    label = str(record.get("annotation") or "").strip().lower()
    if dataset == "fakett":
        if label in {"real", "fake"}:
            return label
        raise ValueError(f"Unsupported FakeTT label: {label!r}")

    if dataset == "fakesv":
        mapping = {"真": "real", "假": "fake"}
        if label in mapping:
            return mapping[label]
        if label == "辟谣":
            if fakesv_refute_policy == "drop":
                return None
            if fakesv_refute_policy == "map_to_fake":
                return "fake"
        raise ValueError(f"Unsupported FakeSV label: {label!r}")

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_user_prompt(dataset: str, record: dict) -> str:
    parts: List[str] = ["Please judge whether this short news video is real or fake."]

    description = str(record.get("description") or "").strip()
    if description:
        parts.append(f"Description: {description}")

    event = str(record.get("event") or "").strip()
    if event:
        parts.append(f"Claimed event: {event}")

    user_description = str(record.get("user_description") or record.get("author_intro") or "").strip()
    if user_description:
        parts.append(f"Uploader profile: {user_description}")

    title = str(record.get("title") or "").strip()
    if title:
        parts.append(f"Title: {title}")

    ocr = str(record.get("ocr") or "").strip()
    if ocr:
        parts.append(f"OCR: {ocr}")

    parts.append("Reply with exactly one word: real or fake.")
    return "\n".join(parts)


def build_jsonl_record(dataset: str, record: dict, label: str, video_path: Path) -> dict:
    return {
        "id": str(record["video_id"]),
        "messages": [
            {"role": "user", "content": build_user_prompt(dataset, record)},
            {"role": "assistant", "content": label},
        ],
        "videos": [str(video_path.resolve(strict=False))],
    }


def iter_split_records(
    dataset: str,
    sample_ids: Iterable[str],
    annotations: Dict[str, dict],
    video_root: Path,
    video_extension: str,
    strict_video_check: bool,
    fakesv_refute_policy: str,
) -> Iterable[dict]:
    missing_annotation_ids: List[str] = []
    missing_video_paths: List[str] = []

    for sample_id in sample_ids:
        record = annotations.get(sample_id)
        if record is None:
            missing_annotation_ids.append(sample_id)
            continue

        label = normalize_label(dataset, record, fakesv_refute_policy)
        if label is None:
            continue

        video_path = video_root / f"{sample_id}{video_extension}"
        if strict_video_check and not video_path.exists():
            missing_video_paths.append(str(video_path))
            continue

        yield build_jsonl_record(dataset, record, label, video_path)

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
    defaults = DATASET_SETTINGS[args.dataset]
    repo_root = args.repo_root.resolve()

    annotation_path = resolve_path(repo_root, args.annotation_path or defaults["annotation_path"])
    split_dir = resolve_path(repo_root, args.split_dir or defaults["split_dir"])
    output_dir = resolve_path(repo_root, args.output_dir or defaults["output_dir"])
    video_root = args.video_root if args.video_root.is_absolute() else (repo_root / args.video_root)
    video_extension = args.video_extension or defaults["video_extension"]

    assert annotation_path is not None and split_dir is not None and output_dir is not None
    annotations = load_annotations(annotation_path)

    split_mapping = {
        "train": split_dir / "vid_time3_train.txt",
        "val": split_dir / "vid_time3_valid.txt",
        "test": split_dir / "vid_time3_test.txt",
    }

    for split_name, split_path in split_mapping.items():
        sample_ids = load_split_ids(split_path)
        output_path = output_dir / f"{args.dataset}_{split_name}.jsonl"
        rows = iter_split_records(
            args.dataset,
            sample_ids,
            annotations,
            video_root,
            video_extension,
            args.strict_video_check,
            args.fakesv_refute_policy,
        )
        count = dump_jsonl(output_path, rows)
        print(f"{split_name}: wrote {count} samples -> {output_path}")


if __name__ == "__main__":
    main()
