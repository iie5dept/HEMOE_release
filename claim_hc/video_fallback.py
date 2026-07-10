from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List

from PIL import Image

from swift.template.vision_utils import load_video_internvl


def _placeholder_frames(num_segments: int) -> List[Image.Image]:
    frame = Image.new("RGB", (224, 224), color=(0, 0, 0))
    return [frame.copy() for _ in range(max(1, num_segments))]


def _load_video_with_ffmpeg(video_path: str, num_segments: int) -> List[Image.Image]:
    with tempfile.TemporaryDirectory(prefix="videommd_ffmpeg_") as tmp_dir:
        output_pattern = str(Path(tmp_dir) / "frame_%03d.jpg")
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            video_path,
            "-vf",
            "thumbnail",
            "-frames:v",
            str(max(1, num_segments)),
            output_pattern,
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        frame_paths = sorted(Path(tmp_dir).glob("frame_*.jpg"))
        if result.returncode != 0 and not frame_paths:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(stderr or f"ffmpeg fallback failed for {video_path}")

        frames = [Image.open(path).convert("RGB") for path in frame_paths]
        if not frames:
            raise RuntimeError(f"ffmpeg fallback produced no frames for {video_path}")
        while len(frames) < max(1, num_segments):
            frames.append(frames[-1].copy())
        return frames


def safe_load_video_internvl(video_path: str, num_segments: int) -> List[Image.Image]:
    try:
        return load_video_internvl(video_path, num_segments=num_segments)
    except Exception as primary_error:
        try:
            frames = _load_video_with_ffmpeg(video_path, num_segments=num_segments)
            print(
                f"[videommd] ffmpeg fallback video loader used for {Path(video_path).name}: {primary_error}",
                flush=True,
            )
            return frames
        except Exception as fallback_error:
            print(
                f"[videommd] placeholder frames used for {Path(video_path).name}: "
                f"primary={primary_error}; fallback={fallback_error}",
                flush=True,
            )
            return _placeholder_frames(num_segments=num_segments)
