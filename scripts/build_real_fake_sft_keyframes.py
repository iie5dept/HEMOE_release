from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


FAKE_NEWS_PROMPT = (
    "Determine whether the literal news claim or event stated in <claim> is factually real or fake "
    "based on these sampled key frames and the auxiliary observations.\n\n"
    "This task is short-video news veracity detection, not deepfake detection. Judge the truth of "
    "the stated claim itself, not whether the frames merely look edited, dramatic, or plausible.\n\n"
    "Label the sample fake if the claim is false, misleading in context, uses old footage presented "
    "as a new event, mismatches the claimed time, place, person, quote, or event, or uses unrelated "
    "or repurposed footage to support a false headline. Normal-looking frames can still support a "
    "fake claim if they do not actually verify the specific statement.\n\n"
    "Label the sample real if the claim itself is factually true as stated and the sampled frames "
    "plus accompanying context genuinely support it, even if the clip looks surprising, sensational, "
    "or visually unusual.\n\n"
    "Important: evaluate the exact wording of the claim. If the claim says that an image, video, or "
    "story is fake, edited, staged, or did not happen, then label the sample real when that "
    "debunking claim is true.\n\n"
    "Do not label a sample fake only because the claim sounds bizarre, miraculous, humorous, or "
    "politically extreme. Do not label a sample real only because the frames look authentic or "
    "because the speaker, topic, or location is real. Use uploader identity, political stance, "
    "hashtags, emotional tone, and publish time only as auxiliary evidence.\n\n"
    "Reply with exactly one word: real or fake."
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build keyframe-based Swift SFT jsonl for FakeTT.")
    parser.add_argument("--annotation-path", type=Path, default=repo_root / "data" / "fakett" / "data.json")
    parser.add_argument("--split-dir", type=Path, default=repo_root / "external" / "ExMRD" / "data" / "FakeTT" / "vids")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "swift_keyframes" / "fakett")
    parser.add_argument("--frame-dir", type=Path, default=repo_root / "data" / "swift_keyframes" / "fakett_frames")
    parser.add_argument("--video-extension", type=str, default=".mp4")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-fps", type=float, default=2.0)
    parser.add_argument("--diff-weight", type=float, default=0.7)
    parser.add_argument("--edge-weight", type=float, default=0.3)
    parser.add_argument("--short-side", type=int, default=96)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--strict-video-check", action="store_true")
    parser.add_argument("--skip-bad-videos", action="store_true")
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


def build_video_path(video_root: Path, sample_id: str, video_extension: str) -> Path:
    return video_root / f"{sample_id}{video_extension}"


def build_user_prompt(record: dict, num_images: int) -> str:
    image_prefix = "".join("<image>\n" for _ in range(num_images))
    parts: List[str] = [image_prefix.rstrip(), FAKE_NEWS_PROMPT, "", "Auxiliary observations:"]

    event = str(record.get("event") or "").strip()
    if event:
        parts.append(f"- news claim to verify: <claim>{event}</claim>")

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


def minmax_normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    value_min = float(array.min())
    value_max = float(array.max())
    if value_max - value_min < 1e-6:
        return np.zeros_like(array)
    return (array - value_min) / (value_max - value_min)


def resize_short_side(frame: np.ndarray, short_side: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    if min(height, width) == short_side:
        return frame
    if height <= width:
        new_height = short_side
        new_width = max(1, int(round(width * short_side / height)))
    else:
        new_width = short_side
        new_height = max(1, int(round(height * short_side / width)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def frame_to_gray_small(frame: np.ndarray, short_side: int) -> np.ndarray:
    import cv2

    small = resize_short_side(frame, short_side)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32)


def compute_edge_score(gray_frame: np.ndarray) -> float:
    gy, gx = np.gradient(gray_frame)
    magnitude = np.abs(gx) + np.abs(gy)
    return float(magnitude.mean())


def unique_sorted(values: Iterable[int]) -> List[int]:
    return sorted({int(v) for v in values})


def select_segment_indices(num_frames_total: int, num_segments: int) -> List[tuple[int, int]]:
    boundaries = np.linspace(0, num_frames_total, num_segments + 1, dtype=np.int32)
    ranges: List[tuple[int, int]] = []
    for i in range(num_segments):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        if end <= start:
            end = min(num_frames_total, start + 1)
        ranges.append((start, end))
    return ranges


def select_keyframe_indices(
    frames: Sequence[np.ndarray],
    diff_weight: float,
    edge_weight: float,
    num_frames: int,
    short_side: int,
) -> List[int]:
    total = len(frames)
    if total <= num_frames:
        return list(range(total))

    gray_frames = [frame_to_gray_small(frame, short_side) for frame in frames]
    diff_scores = [0.0] * total
    edge_scores = [compute_edge_score(gray) for gray in gray_frames]

    for idx in range(total):
        prev_idx = max(0, idx - 1)
        next_idx = min(total - 1, idx + 1)
        prev_diff = float(np.mean(np.abs(gray_frames[idx] - gray_frames[prev_idx]))) if idx != prev_idx else 0.0
        next_diff = float(np.mean(np.abs(gray_frames[next_idx] - gray_frames[idx]))) if idx != next_idx else 0.0
        diff_scores[idx] = max(prev_diff, next_diff)

    selected: List[int] = []
    for start, end in select_segment_indices(total, num_frames):
        candidate_indices = list(range(start, end))
        if not candidate_indices:
            continue
        segment_diff = minmax_normalize([diff_scores[i] for i in candidate_indices])
        segment_edge = minmax_normalize([edge_scores[i] for i in candidate_indices])
        segment_center = (start + end - 1) / 2.0

        best_index = candidate_indices[0]
        best_score = None
        for local_idx, frame_idx in enumerate(candidate_indices):
            score = diff_weight * float(segment_diff[local_idx]) + edge_weight * float(segment_edge[local_idx])
            center_penalty = abs(frame_idx - segment_center) * 1e-4
            if best_score is None or score > best_score + 1e-8:
                best_index = frame_idx
                best_score = score
            elif abs(score - best_score) <= 1e-8:
                current_distance = abs(best_index - segment_center)
                if abs(frame_idx - segment_center) < current_distance - 1e-8:
                    best_index = frame_idx
        selected.append(best_index)

    return unique_sorted(selected)[:num_frames]


def load_video_frames(video_path: Path, max_fps: float) -> List[np.ndarray]:
    try:
        return load_video_frames_cv2(video_path, max_fps)
    except Exception:
        return load_video_frames_decord(video_path, max_fps)


def load_video_frames_cv2(video_path: Path, max_fps: float) -> List[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video via cv2: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = max_fps if max_fps > 0 else 1.0
    step = max(1, int(round(fps / max_fps))) if max_fps > 0 else 1

    frames: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def load_video_frames_decord(video_path: Path, max_fps: float) -> List[np.ndarray]:
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    avg_fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0
    if not avg_fps or avg_fps <= 0:
        avg_fps = max_fps if max_fps > 0 else 1.0
    step = max(1, int(round(avg_fps / max_fps))) if max_fps > 0 else 1
    indices = list(range(0, len(vr), step))
    if not indices:
        indices = [0]
    batch = vr.get_batch(indices).asnumpy()
    return [frame for frame in batch]


def save_frames(frames: Sequence[np.ndarray], frame_indices: Sequence[int], target_dir: Path, jpeg_quality: int) -> List[str]:
    import cv2

    target_dir.mkdir(parents=True, exist_ok=True)
    image_paths: List[str] = []
    for order, frame_idx in enumerate(frame_indices):
        frame = frames[frame_idx]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = target_dir / f"frame_{order:02d}_{frame_idx:05d}.jpg"
        cv2.imwrite(str(image_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        image_paths.append(image_path.as_posix())
    return image_paths


def build_jsonl_record(record: dict, image_paths: Sequence[str]) -> dict:
    return {
        "id": str(record["video_id"]),
        "messages": [
            {"role": "user", "content": build_user_prompt(record, len(image_paths))},
            {"role": "assistant", "content": normalize_label(record)},
        ],
        "images": list(image_paths),
    }


def build_bad_video_record(split_name: str, sample_id: str, video_path: Path, error: Exception) -> dict:
    return {
        "split": split_name,
        "id": sample_id,
        "video_path": video_path.as_posix(),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def iter_split_records(
    split_name: str,
    sample_ids: Iterable[str],
    annotations: Dict[str, dict],
    video_root: Path,
    frame_dir: Path,
    video_extension: str,
    strict_video_check: bool,
    num_frames: int,
    max_fps: float,
    diff_weight: float,
    edge_weight: float,
    short_side: int,
    jpeg_quality: int,
    skip_bad_videos: bool,
) -> tuple[List[dict], List[dict]]:
    missing_annotation_ids: List[str] = []
    missing_video_paths: List[str] = []
    rows: List[dict] = []
    bad_videos: List[dict] = []

    for sample_id in sample_ids:
        record = annotations.get(sample_id)
        if record is None:
            missing_annotation_ids.append(sample_id)
            continue

        video_path = build_video_path(video_root, sample_id, video_extension)
        if strict_video_check and not video_path.exists():
            missing_video_paths.append(str(video_path))
            continue

        try:
            frames = load_video_frames(video_path, max_fps)
            selected_indices = select_keyframe_indices(frames, diff_weight, edge_weight, num_frames, short_side)
            if not selected_indices:
                raise RuntimeError(f"No keyframes selected from decoded frames: {video_path}")
            target_dir = frame_dir / split_name / sample_id
            image_paths = save_frames(frames, selected_indices, target_dir, jpeg_quality)
            rows.append(build_jsonl_record(record, image_paths))
        except Exception as error:
            if not skip_bad_videos:
                raise
            bad_videos.append(build_bad_video_record(split_name, sample_id, video_path, error))
            print(
                json.dumps(
                    {
                        "warning": "skip_bad_video",
                        "split": split_name,
                        "id": sample_id,
                        "video_path": video_path.as_posix(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                )
            )

    if missing_annotation_ids:
        preview = ", ".join(missing_annotation_ids[:5])
        raise ValueError(f"Missing annotations for {len(missing_annotation_ids)} ids. Examples: {preview}")
    if missing_video_paths:
        preview = ", ".join(missing_video_paths[:3])
        raise ValueError(f"Missing video files for {len(missing_video_paths)} samples. Examples: {preview}")
    return rows, bad_videos


def dump_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def dump_bad_video_report(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        bad_report_path = args.output_dir / f"fakett_{split_name}_bad_videos.jsonl"
        rows, bad_videos = iter_split_records(
            split_name,
            sample_ids,
            annotations,
            args.video_root,
            args.frame_dir,
            args.video_extension,
            args.strict_video_check,
            args.num_frames,
            args.max_fps,
            args.diff_weight,
            args.edge_weight,
            args.short_side,
            args.jpeg_quality,
            args.skip_bad_videos,
        )
        count = dump_jsonl(output_path, rows)
        dump_bad_video_report(bad_report_path, bad_videos)
        print(
            json.dumps(
                {
                    "split": split_name,
                    "output": str(output_path),
                    "count": count,
                    "frame_dir": str((args.frame_dir / split_name).as_posix()),
                    "bad_videos": len(bad_videos),
                    "bad_report": str(bad_report_path),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
