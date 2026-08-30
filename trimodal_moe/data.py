from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


LABEL_TO_ID = {"real": 0, "fake": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return records


def _feature_index(path: str | Path) -> dict[str, Path]:
    manifest_path = Path(path)
    index: dict[str, Path] = {}
    for record in read_jsonl(manifest_path):
        sample_id = str(record.get("video_id", record.get("id", ""))).strip()
        feature_path = Path(record["feature_path"])
        if not feature_path.is_absolute():
            feature_path = manifest_path.parent / feature_path
        if sample_id in index:
            raise ValueError(f"Duplicate feature record for {sample_id} in {manifest_path}")
        index[sample_id] = feature_path
    return index


def _message_content(record: dict[str, Any], role: str) -> str:
    for message in record.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", "")).strip()
    raise ValueError(f"Sample {record.get('id')} has no {role!r} message")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    user_content: str
    video_path: Path
    label: int
    visual_feature_path: Path
    text_feature_path: Path
    audio_feature_path: Path


class FourModalFeatureDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        visual_manifest: str | Path,
        text_manifest: str | Path,
        audio_manifest: str | Path,
    ) -> None:
        visual_index = _feature_index(visual_manifest)
        text_index = _feature_index(text_manifest)
        audio_index = _feature_index(audio_manifest)
        self.records: list[SampleRecord] = []
        missing: list[str] = []
        for record in read_jsonl(data_path):
            sample_id = str(record.get("id", "")).strip()
            if sample_id not in visual_index or sample_id not in text_index or sample_id not in audio_index:
                missing.append(sample_id)
                continue
            answer = _message_content(record, "assistant").lower()
            if answer not in LABEL_TO_ID:
                raise ValueError(f"Unsupported label {answer!r} for sample {sample_id}")
            videos = record.get("videos") or []
            if not videos:
                raise ValueError(f"Sample {sample_id} has no video path")
            self.records.append(
                SampleRecord(
                    sample_id=sample_id,
                    user_content=_message_content(record, "user"),
                    video_path=Path(videos[0]),
                    label=LABEL_TO_ID[answer],
                    visual_feature_path=visual_index[sample_id],
                    text_feature_path=text_index[sample_id],
                    audio_feature_path=audio_index[sample_id],
                )
            )
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(
                f"{len(missing)} samples are missing DINO, BERT, or Qwen2-Audio cache entries in "
                f"{data_path}; first IDs: {preview}"
            )
        if not self.records:
            raise ValueError(f"No samples loaded from {data_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from safetensors.torch import load_file

        record = self.records[index]
        visual = load_file(str(record.visual_feature_path))
        text = load_file(str(record.text_feature_path))
        audio = load_file(str(record.audio_feature_path))
        if "patch_tokens" not in visual:
            raise KeyError(f"DINO cache for {record.sample_id} has no patch_tokens")
        if "token_states" not in text:
            raise KeyError(f"BERT cache for {record.sample_id} has no token_states")
        if "audio_last_token" not in audio:
            raise KeyError(f"Qwen2-Audio cache for {record.sample_id} has no audio_last_token")
        return {
            "id": record.sample_id,
            "user_content": record.user_content,
            "video_path": str(record.video_path),
            "label": record.label,
            "visual_tokens": visual["patch_tokens"],
            "text_tokens": text["token_states"],
            "text_attention_mask": text.get(
                "attention_mask", torch.ones(text["token_states"].shape[0], dtype=torch.bool)
            ),
            "audio_states": audio["audio_last_token"],
        }


def _uniform_frame_indices(frame_count: int, num_segments: int) -> list[int]:
    if frame_count <= 0:
        return [0] * num_segments
    points = np.linspace(0, frame_count - 1, num=max(1, num_segments))
    return [int(round(value)) for value in points]


def _load_video_decord(path: str, num_segments: int) -> list[Image.Image]:
    from decord import VideoReader, cpu

    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    indices = _uniform_frame_indices(len(reader), num_segments)
    frames = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def _load_video_ffmpeg(path: str, num_segments: int) -> list[Image.Image]:
    with tempfile.TemporaryDirectory(prefix="trimodal_ffmpeg_") as directory:
        pattern = str(Path(directory) / "frame_%03d.jpg")
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            path,
            "-vf",
            "thumbnail",
            "-frames:v",
            str(num_segments),
            pattern,
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        frame_paths = sorted(Path(directory).glob("frame_*.jpg"))
        if result.returncode != 0 and not frame_paths:
            raise RuntimeError((result.stderr or "ffmpeg failed").strip())
        frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]
        if not frames:
            raise RuntimeError("ffmpeg produced no frames")
        while len(frames) < num_segments:
            frames.append(frames[-1].copy())
        return frames[:num_segments]


def load_video_frames(path: str, num_segments: int) -> list[Image.Image]:
    try:
        return _load_video_decord(path, num_segments)
    except Exception as primary_error:
        try:
            return _load_video_ffmpeg(path, num_segments)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Cannot decode {path}: decord={primary_error}; ffmpeg={fallback_error}"
            ) from fallback_error


def image_to_tensor(image: Image.Image, image_size: int) -> Tensor:
    resized = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def pad_feature_tokens(sequences: Iterable[Tensor]) -> tuple[Tensor, Tensor]:
    values = list(sequences)
    max_length = max(item.shape[0] for item in values)
    hidden_size = values[0].shape[-1]
    output = torch.zeros((len(values), max_length, hidden_size), dtype=values[0].dtype)
    mask = torch.zeros((len(values), max_length), dtype=torch.bool)
    for index, item in enumerate(values):
        length = item.shape[0]
        output[index, :length] = item
        mask[index, :length] = True
    return output, mask


class FourModalCollator:
    def __init__(
        self,
        tokenizer: Any,
        system_prompt: str,
        num_image_token: int,
        num_segments: int = 8,
        image_size: int = 448,
        max_length: int = 8192,
    ) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.num_image_token = num_image_token
        self.num_segments = num_segments
        self.image_size = image_size
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

    def _question(self, content: str) -> str:
        frame_prefix = "".join(f"Frame{index + 1}: <image>\n" for index in range(self.num_segments))
        content = content.replace("<video>", frame_prefix, 1)
        image_tokens = "<img>" + "<IMG_CONTEXT>" * self.num_image_token + "</img>"
        return content.replace("<image>", image_tokens)

    def _prompt(self, content: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._question(content)})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [self._prompt(sample["user_content"]) for sample in samples]
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        frames = [
            image_to_tensor(frame, self.image_size)
            for sample in samples
            for frame in load_video_frames(sample["video_path"], self.num_segments)
        ]
        visual_tokens, visual_mask = pad_feature_tokens(sample["visual_tokens"] for sample in samples)
        text_tokens, text_mask = pad_feature_tokens(sample["text_tokens"] for sample in samples)
        return {
            "ids": [sample["id"] for sample in samples],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"].bool(),
            "pixel_values": torch.stack(frames),
            "visual_tokens": visual_tokens,
            "visual_attention_mask": visual_mask,
            "text_tokens": text_tokens,
            "text_attention_mask": text_mask,
            "audio_states": torch.stack([sample["audio_states"] for sample in samples]),
            "labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        }


TriModalFeatureDataset = FourModalFeatureDataset
TriModalCollator = FourModalCollator
