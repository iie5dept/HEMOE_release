from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


FAKE_NEWS_PROMPT_ZH = (
    "请判断这个短视频及其辅助信息所表达的新闻事件或主张是真实还是虚假。\n\n"
    "这是一项短视频新闻真实性判断任务，不是深度伪造检测任务。你需要判断视频及其相关文本上下文所表达的新闻内容是否属实。\n\n"
    "如果视频表达的新闻内容是错误的、已被辟谣的、存在断章取义、旧视频冒充新事件、与声称的时间地点人物事件不匹配，"
    "或用无关素材支撑错误新闻主张，则应标为 fake。\n\n"
    "如果视频表达的新闻内容真实，且视频与相关上下文没有误导性地歪曲事件，则应标为 real。\n\n"
    "不要仅根据发布者身份、立场、情绪化表达、标签热度或画面看起来是否逼真来判断，它们只能作为辅助信息。\n\n"
    "只回答一个词：real 或 fake。"
)


REAL_LABEL_VALUES = {"真", "真实", "real", "鐪?", "鐪熷疄"}
FAKE_LABEL_VALUES = {"假", "谣言", "假新闻", "fake", "鍋?", "璋ｈ█"}
DEBUNKED_LABEL_VALUES = {"辟谣", "辟谣视频", "debunked", "è¾Ÿè°£", "杈熻埃"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build Chinese Swift SFT jsonl for FakeSV.")
    parser.add_argument("--annotation-path", type=Path, default=repo_root / "data" / "fakesv" / "data_complete.json")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=repo_root / "external" / "ExMRD" / "data" / "FakeSV" / "vids",
    )
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "swift" / "fakesv")
    parser.add_argument("--video-extension", type=str, default=".mp4")
    parser.add_argument("--max-comments", type=int, default=3)
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


def normalize_label(record: dict) -> str | None:
    label = str(record.get("annotation") or "").strip().lower()
    if label in {value.lower() for value in REAL_LABEL_VALUES}:
        return "real"
    if label in {value.lower() for value in FAKE_LABEL_VALUES}:
        return "fake"
    if label in {value.lower() for value in DEBUNKED_LABEL_VALUES}:
        return None
    raise ValueError(f"Unsupported FakeSV label: {label!r}")


def build_video_path(video_root: str, sample_id: str, video_extension: str) -> str:
    normalized_root = video_root.rstrip("/\\")
    normalized_root = normalized_root.replace("\\", "/")
    return f"{normalized_root}/{sample_id}{video_extension}"


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "null":
        return ""
    return text


def format_comments(comments, max_comments: int) -> str:
    if not isinstance(comments, list) or max_comments <= 0:
        return ""
    cleaned: List[str] = []
    for item in comments:
        text = normalize_text(item)
        if not text:
            continue
        cleaned.append(text.replace("\n", " "))
        if len(cleaned) >= max_comments:
            break
    return " | ".join(cleaned)


def build_user_prompt(record: dict, max_comments: int) -> str:
    parts: List[str] = ["<video>", FAKE_NEWS_PROMPT_ZH, "", "辅助信息："]

    title = normalize_text(record.get("title"))
    if title:
        parts.append(f"- 待核验新闻主张: <claim>{title}</claim>")

    keywords = normalize_text(record.get("keywords"))
    if keywords:
        parts.append(f"- 事件关键词: {keywords}")

    author_intro = normalize_text(record.get("author_intro"))
    if author_intro:
        parts.append(f"- 发布者简介: {author_intro}")

    author_place = normalize_text(record.get("author_place"))
    if author_place:
        parts.append(f"- 发布者地区: {author_place}")

    publish_time = format_publish_time(record.get("publish_time_norm"))
    if publish_time:
        parts.append(f"- 发布时间: {publish_time}")

    comments = format_comments(record.get("comments"), max_comments=max_comments)
    if comments:
        parts.append(f"- 部分评论: {comments}")

    return "\n".join(parts)


def build_jsonl_record(record: dict, video_path: str, max_comments: int, label: str) -> dict:
    return {
        "id": str(record["video_id"]),
        "messages": [
            {"role": "user", "content": build_user_prompt(record, max_comments=max_comments)},
            {"role": "assistant", "content": label},
        ],
        "videos": [video_path],
    }


def iter_split_records(
    sample_ids: Iterable[str],
    annotations: Dict[str, dict],
    video_root: str,
    video_extension: str,
    strict_video_check: bool,
    max_comments: int,
) -> tuple[Iterable[dict], dict]:
    missing_annotation_ids: List[str] = []
    missing_video_paths: List[str] = []
    skipped_debunked = 0
    skipped_bad = 0

    def generator():
        nonlocal skipped_debunked, skipped_bad
        for sample_id in sample_ids:
            record = annotations.get(sample_id)
            if record is None:
                missing_annotation_ids.append(sample_id)
                continue

            try:
                label = normalize_label(record)
            except Exception:
                skipped_bad += 1
                continue

            if label is None:
                skipped_debunked += 1
                continue

            video_path = build_video_path(video_root, sample_id, video_extension)
            if strict_video_check and not Path(video_path).exists():
                missing_video_paths.append(video_path)
                continue

            yield build_jsonl_record(record, video_path, max_comments=max_comments, label=label)

    stats = {
        "missing_annotation_ids": missing_annotation_ids,
        "missing_video_paths": missing_video_paths,
        "skipped_debunked_ref": lambda: skipped_debunked,
        "skipped_bad_ref": lambda: skipped_bad,
    }
    return generator(), stats


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
        output_path = args.output_dir / f"fakesv_{split_name}.jsonl"
        rows, stats = iter_split_records(
            sample_ids=sample_ids,
            annotations=annotations,
            video_root=args.video_root,
            video_extension=args.video_extension,
            strict_video_check=args.strict_video_check,
            max_comments=args.max_comments,
        )
        count = dump_jsonl(output_path, rows)

        missing_annotation_ids = stats["missing_annotation_ids"]
        missing_video_paths = stats["missing_video_paths"]
        if missing_annotation_ids:
            preview = ", ".join(missing_annotation_ids[:5])
            raise ValueError(f"Missing annotations for {len(missing_annotation_ids)} ids. Examples: {preview}")
        if missing_video_paths:
            preview = ", ".join(missing_video_paths[:3])
            raise ValueError(f"Missing video files for {len(missing_video_paths)} samples. Examples: {preview}")

        print(
            json.dumps(
                {
                    "split": split_name,
                    "output": str(output_path),
                    "count": count,
                    "skipped_bad": stats["skipped_bad_ref"](),
                    "skipped_debunked": stats["skipped_debunked_ref"](),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
