from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASET_DEFAULTS = {
    "fakett": {
        "annotation": "data/fakett/data.json",
        "video_root": "/data2/573ops_ser/data/FakeTT/FakeTT/video",
    },
    "fakesv": {
        "annotation": "data/fakesv/data_complete.json",
        "video_root": "/data2/573ops_ser/data/FakeSV/video/videos",
    },
}


@dataclass
class CandidateFrame:
    frame_index: int
    timestamp_sec: float
    image: Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Select one text-aligned CLIP key frame for every FakeTT/FakeSV video."
    )
    parser.add_argument("--dataset", choices=("fakett", "fakesv", "all"), default="all")
    parser.add_argument("--model-path", type=Path, required=True, help="Local offline CLIP model directory.")
    parser.add_argument(
        "--jina-clip-code-path",
        type=Path,
        help="Local ModelScope snapshot of jinaai/jina-clip-implementation.",
    )
    parser.add_argument(
        "--jina-xlm-code-path",
        type=Path,
        help="Local ModelScope snapshot of jinaai/xlm-roberta-flash-implementation.",
    )
    parser.add_argument(
        "--jina-text-config-path",
        type=Path,
        help="Local directory containing jinaai/jina-embeddings-v3 config.json.",
    )
    parser.add_argument(
        "--offline-model-dir",
        type=Path,
        help="Generated offline bundle. Defaults to <model-path>-offline.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "data" / "key_frames",
        help="Each dataset is written to a separate subdirectory.",
    )
    parser.add_argument(
        "--fakett-annotation",
        type=Path,
        default=repo_root / DATASET_DEFAULTS["fakett"]["annotation"],
    )
    parser.add_argument(
        "--fakesv-annotation",
        type=Path,
        default=repo_root / DATASET_DEFAULTS["fakesv"]["annotation"],
    )
    parser.add_argument(
        "--fakett-video-root",
        type=Path,
        default=Path(DATASET_DEFAULTS["fakett"]["video_root"]),
    )
    parser.add_argument(
        "--fakesv-video-root",
        type=Path,
        default=Path(DATASET_DEFAULTS["fakesv"]["video_root"]),
    )
    parser.add_argument("--video-extensions", default=".mp4,.mov,.mkv,.webm")
    parser.add_argument("--candidate-frames", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Jina Matryoshka dimension.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="Use a small positive value for a smoke test.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.candidate_frames <= 0:
        parser.error("--candidate-frames must be positive")
    if args.image_batch_size <= 0:
        parser.error("--image-batch-size must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    return args


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {path}")
            records = payload
        else:
            records = [json.loads(line) for line in handle if line.strip()]

    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def list_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(normalize_text(item) for item in value if normalize_text(item))
    return normalize_text(value)


def build_retrieval_text(dataset: str, record: dict[str, Any]) -> tuple[str, list[str]]:
    if dataset == "fakett":
        fields = (("description", normalize_text(record.get("description"))),
                  ("event", normalize_text(record.get("event"))))
    elif dataset == "fakesv":
        fields = (("title", normalize_text(record.get("title"))),
                  ("keywords", list_text(record.get("keywords"))),
                  ("ocr", normalize_text(record.get("ocr"))))
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for name, text in fields:
        if text:
            return text, [name]
    raise ValueError("No usable description/title text")


def resolve_video(video_root: Path, sample_id: str, extensions: Sequence[str]) -> Path:
    direct_id = Path(sample_id)
    if direct_id.suffix:
        candidate = video_root / direct_id
        if candidate.is_file():
            return candidate
    for extension in extensions:
        candidate = video_root / f"{sample_id}{extension}"
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(video_root / f"{sample_id}{ext}") for ext in extensions)
    raise FileNotFoundError(f"Video not found. Tried: {searched}")


def uniform_indices(frame_count: int, candidate_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    count = min(frame_count, candidate_count)
    if count == 1:
        return [frame_count // 2]
    # Segment centers avoid over-selecting opening and closing black frames.
    indices = [int((slot + 0.5) * frame_count / count) for slot in range(count)]
    return sorted({min(frame_count - 1, index) for index in indices})


def sample_frames_cv2(video_path: Path, candidate_count: int, seed: int) -> list[CandidateFrame]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[CandidateFrame] = []

    try:
        for frame_index in uniform_indices(frame_count, candidate_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_index / fps if fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            frames.append(CandidateFrame(frame_index, max(0.0, timestamp), Image.fromarray(rgb)))
    finally:
        cap.release()

    expected = min(frame_count, candidate_count) if frame_count > 0 else candidate_count
    if frames and len(frames) >= max(1, math.ceil(expected * 0.5)):
        return frames

    # Some damaged containers report an invalid frame count or cannot seek. A bounded
    # reservoir keeps memory fixed while sequentially decoding such videos.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to reopen video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 0.0)
    rng = random.Random(seed)
    reservoir: list[tuple[int, Any]] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if len(reservoir) < candidate_count:
                reservoir.append((frame_index, frame.copy()))
            else:
                replacement = rng.randint(0, frame_index)
                if replacement < candidate_count:
                    reservoir[replacement] = (frame_index, frame.copy())
            frame_index += 1
    finally:
        cap.release()

    if not reservoir:
        raise RuntimeError(f"No frames could be decoded from: {video_path}")
    output = []
    for index, frame in sorted(reservoir, key=lambda item: item[0]):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = index / fps if fps > 0 else 0.0
        output.append(CandidateFrame(index, timestamp, Image.fromarray(rgb)))
    return output


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_dtype(name: str, device: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if not device.startswith("cuda"):
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def move_inputs(inputs: Any, device: str) -> dict[str, Any]:
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    return dict(inputs)


def as_embedding_tensor(value: Any, device: str):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def rewrite_external_auto_maps(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key == "auto_map" and isinstance(item, dict):
                output[key] = {
                    map_key: map_value.split("--", 1)[-1]
                    if isinstance(map_value, str) and "--" in map_value
                    else map_value
                    for map_key, map_value in item.items()
                }
            else:
                output[key] = rewrite_external_auto_maps(item)
        return output
    if isinstance(value, list):
        return [rewrite_external_auto_maps(item) for item in value]
    return value


def external_auto_map_repositories(config: dict[str, Any]) -> set[str]:
    repositories: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "auto_map" and isinstance(item, dict):
                    for target in item.values():
                        if isinstance(target, str) and "--" in target:
                            repositories.add(target.split("--", 1)[0])
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(config)
    return repositories


def link_model_entry(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        return
    except OSError:
        pass
    if source.is_file():
        try:
            os.link(source, target)
            return
        except OSError:
            shutil.copy2(source, target)
            return
    shutil.copytree(source, target)


def copy_python_implementation(source_dir: Path, target_dir: Path) -> int:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Jina implementation directory does not exist: {source_dir}")
    python_files = list(source_dir.glob("*.py"))
    if not python_files:
        raise FileNotFoundError(f"No Python implementation files found in: {source_dir}")
    for source in python_files:
        shutil.copy2(source, target_dir / source.name)
    return len(python_files)


def seed_transformers_dynamic_cache(source_dir: Path) -> int:
    from transformers.dynamic_module_utils import (
        HF_MODULES_CACHE,
        TRANSFORMERS_DYNAMIC_MODULE_NAME,
    )

    python_files = list(source_dir.rglob("*.py"))
    if not python_files:
        raise FileNotFoundError(f"No Python files available for dynamic module cache: {source_dir}")
    target_root = (
        Path(HF_MODULES_CACHE)
        / TRANSFORMERS_DYNAMIC_MODULE_NAME
        / source_dir.resolve().name
    )
    target_root.mkdir(parents=True, exist_ok=True)
    for parent in (Path(HF_MODULES_CACHE), target_root.parent, target_root):
        init_path = parent / "__init__.py"
        init_path.touch(exist_ok=True)
    for source in python_files:
        relative_path = source.relative_to(source_dir)
        target = target_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        current = target.parent
        while current != target_root:
            (current / "__init__.py").touch(exist_ok=True)
            current = current.parent
    importlib.invalidate_caches()
    return len(python_files)


def seed_jina_dynamic_caches(model_path: Path) -> None:
    config = read_json(model_path / "config.json")
    text_model_path = config.get("text_config", {}).get("hf_model_name_or_path")
    if not text_model_path:
        return
    text_code_dir = Path(text_model_path)
    required_xlm_files = {
        "configuration_xlm_roberta.py",
        "modeling_lora.py",
        "modeling_xlm_roberta.py",
        "mha.py",
    }
    missing = sorted(name for name in required_xlm_files if not (text_code_dir / name).is_file())
    if missing:
        raise FileNotFoundError(
            "The local xlm-roberta implementation snapshot is incomplete. Missing from "
            f"{text_code_dir}: {', '.join(missing)}. Re-download "
            "jinaai/xlm-roberta-flash-implementation from ModelScope."
        )
    main_count = seed_transformers_dynamic_cache(model_path)
    text_count = seed_transformers_dynamic_cache(text_code_dir)
    print(
        json.dumps(
            {
                "seeded_dynamic_cache": {
                    "clip_files": main_count,
                    "xlm_files": text_count,
                    "text_code_dir": str(text_code_dir.resolve()),
                }
            },
            ensure_ascii=False,
        )
    )


def prepare_jina_offline_bundle(
    model_path: Path,
    clip_code_path: Path | None,
    xlm_code_path: Path | None,
    text_config_path: Path | None,
    output_path: Path | None,
) -> Path:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")
    config = read_json(config_path)
    repositories = external_auto_map_repositories(config)
    if not repositories:
        return model_path

    required = {
        "--jina-clip-code-path": clip_code_path,
        "--jina-xlm-code-path": xlm_code_path,
        "--jina-text-config-path": text_config_path,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        repo_list = ", ".join(sorted(repositories))
        raise RuntimeError(
            "This Jina snapshot contains external Transformers auto_map references "
            f"({repo_list}). ModelScope does not place that code in the Hugging Face cache. "
            f"Provide the local dependencies with: {', '.join(missing)}."
        )

    text_config_file = text_config_path / "config.json"
    if not text_config_file.is_file():
        raise FileNotFoundError(f"Jina text config does not exist: {text_config_file}")

    bundle = output_path or model_path.with_name(f"{model_path.name}-offline")
    bundle.mkdir(parents=True, exist_ok=True)
    for source in model_path.iterdir():
        if source.suffix.lower() == ".json":
            try:
                payload = rewrite_external_auto_maps(read_json(source))
            except (json.JSONDecodeError, UnicodeDecodeError):
                shutil.copy2(source, bundle / source.name)
            else:
                write_json(bundle / source.name, payload)
            continue
        link_model_entry(source, bundle / source.name)
    clip_file_count = copy_python_implementation(clip_code_path, bundle)

    text_bundle = bundle / "jina-embeddings-v3-config"
    text_bundle.mkdir(parents=True, exist_ok=True)
    text_config = rewrite_external_auto_maps(read_json(text_config_file))
    write_json(text_bundle / "config.json", text_config)
    xlm_file_count = copy_python_implementation(xlm_code_path, text_bundle)

    config = rewrite_external_auto_maps(config)
    config.setdefault("text_config", {})["hf_model_name_or_path"] = str(text_bundle.resolve())
    write_json(bundle / "config.json", config)

    marker = {
        "source_model": str(model_path.resolve()),
        "clip_code_path": str(clip_code_path.resolve()),
        "xlm_code_path": str(xlm_code_path.resolve()),
        "text_config_path": str(text_config_path.resolve()),
        "clip_python_files": clip_file_count,
        "xlm_python_files": xlm_file_count,
    }
    write_json(bundle / "offline_bundle.json", marker)
    print(json.dumps({"offline_model_bundle": str(bundle.resolve()), **marker}, ensure_ascii=False))
    return bundle


class ClipEmbedder:
    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype_name: str,
        trust_remote_code: bool,
        embedding_dim: int,
    ) -> None:
        import torch
        from transformers import AutoModel

        self.device = choose_device(device)
        self.dtype = choose_dtype(dtype_name, self.device)
        self.embedding_dim = embedding_dim if embedding_dim > 0 else None
        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise NotADirectoryError(f"Local model directory does not exist: {self.model_path}")

        if (self.model_path / "offline_bundle.json").is_file():
            seed_jina_dynamic_caches(self.model_path)

        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": self.dtype,
        }
        self.model = AutoModel.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.to(self.device).eval()
        self.backend = "jina" if all(hasattr(self.model, name) for name in ("encode_text", "encode_image")) else "transformers"
        self.tokenizer = None
        self.image_processor = None
        self.text_max_length = None
        if self.backend == "transformers":
            from transformers import AutoImageProcessor, AutoTokenizer

            if not all(hasattr(self.model, name) for name in ("get_text_features", "get_image_features")):
                raise TypeError(
                    "Model exposes neither Jina encode_text/encode_image nor Transformers "
                    "get_text_features/get_image_features. Use a CLIP/SigLIP checkpoint."
                )
            tokenizer_files = (self.model_path / "tokenizer.json", self.model_path / "tokenizer.model")
            if not any(path.is_file() for path in tokenizer_files):
                expected = " or ".join(str(path) for path in tokenizer_files)
                raise FileNotFoundError(f"Local CLIP/SigLIP tokenizer is incomplete; expected {expected}.")
            if not (self.model_path / "preprocessor_config.json").is_file():
                raise FileNotFoundError(
                    f"Local image preprocessor config is missing: {self.model_path / 'preprocessor_config.json'}"
                )

            # Some Transformers versions bind SigLIP2's AutoProcessor to the
            # legacy SiglipTokenizer even though tokenizer_config.json declares
            # GemmaTokenizer. Loading both components explicitly avoids that
            # incompatibility and lets the fast tokenizer use tokenizer.json.
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                    use_fast=True,
                )
                self.image_processor = AutoImageProcessor.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception as error:
                raise RuntimeError(
                    "Failed to load the local CLIP/SigLIP tokenizer or image processor. "
                    "Verify tokenizer.json, tokenizer.model, tokenizer_config.json, and "
                    "preprocessor_config.json in the model directory."
                ) from error

            text_config = getattr(self.model.config, "text_config", None)
            max_length = getattr(text_config, "max_position_embeddings", None)
            if isinstance(max_length, int) and max_length > 0:
                self.text_max_length = max_length
        print(
            json.dumps(
                {
                    "model": str(self.model_path),
                    "backend": self.backend,
                    "device": self.device,
                    "dtype": str(self.dtype).replace("torch.", ""),
                },
                ensure_ascii=False,
            )
        )

    def encode_text(self, text: str):
        import torch

        with torch.inference_mode():
            if self.backend == "jina":
                kwargs = {"task": "retrieval.query"}
                if self.embedding_dim is not None:
                    kwargs["truncate_dim"] = self.embedding_dim
                try:
                    output = self.model.encode_text([text], **kwargs)
                except TypeError:
                    kwargs.pop("task", None)
                    output = self.model.encode_text([text], **kwargs)
            else:
                tokenizer_kwargs = {
                    "padding": "max_length",
                    "truncation": True,
                    "return_tensors": "pt",
                }
                if self.text_max_length is not None:
                    tokenizer_kwargs["max_length"] = self.text_max_length
                inputs = self.tokenizer(
                    [text],
                    **tokenizer_kwargs,
                )
                output = self.model.get_text_features(**move_inputs(inputs, self.device))
        return torch.nn.functional.normalize(as_embedding_tensor(output, self.device), dim=-1)

    def encode_images(self, images: Sequence[Any], batch_size: int):
        import torch

        batches = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = list(images[start : start + batch_size])
                if self.backend == "jina":
                    kwargs = {}
                    if self.embedding_dim is not None:
                        kwargs["truncate_dim"] = self.embedding_dim
                    output = self.model.encode_image(batch, **kwargs)
                else:
                    inputs = self.image_processor(images=batch, return_tensors="pt")
                    output = self.model.get_image_features(**move_inputs(inputs, self.device))
                batches.append(as_embedding_tensor(output, self.device))
        embeddings = torch.cat(batches, dim=0)
        return torch.nn.functional.normalize(embeddings, dim=-1)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def save_jpeg_atomic(image: Any, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.jpg")
    image.convert("RGB").save(temporary, format="JPEG", quality=quality, optimize=True)
    temporary.replace(path)


def rebuild_manifest(metadata_dir: Path, manifest_path: Path) -> int:
    records = []
    for path in metadata_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    records.sort(key=lambda item: str(item["video_id"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    return len(records)


def selected_datasets(name: str) -> list[str]:
    return ["fakett", "fakesv"] if name == "all" else [name]


def iter_records(records: Sequence[dict[str, Any]], max_samples: int) -> Iterable[dict[str, Any]]:
    if max_samples > 0:
        return records[:max_samples]
    return records


def run_dataset(dataset: str, args: argparse.Namespace, embedder: ClipEmbedder) -> dict[str, int]:
    import torch
    from tqdm import tqdm

    annotation_path = getattr(args, f"{dataset}_annotation")
    video_root = getattr(args, f"{dataset}_video_root")
    extensions = [item.strip() for item in args.video_extensions.split(",") if item.strip()]
    extensions = [item if item.startswith(".") else f".{item}" for item in extensions]
    records = load_json_records(annotation_path)

    dataset_dir = args.output_root / dataset
    image_dir = dataset_dir / "key_frames"
    metadata_dir = dataset_dir / "metadata"
    error_dir = dataset_dir / "errors"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    selected_records = list(iter_records(records, args.max_samples))
    for record in tqdm(selected_records, desc=dataset, unit="video"):
        sample_id = normalize_text(record.get("video_id"))
        if not sample_id:
            counts["failed"] += 1
            continue
        image_path = image_dir / f"{sample_id}.jpg"
        metadata_path = metadata_dir / f"{sample_id}.json"
        error_path = error_dir / f"{sample_id}.json"
        if image_path.is_file() and metadata_path.is_file() and not args.overwrite:
            counts["skipped"] += 1
            continue

        try:
            text, text_fields = build_retrieval_text(dataset, record)
            video_path = resolve_video(video_root, sample_id, extensions)
            frames = sample_frames_cv2(video_path, args.candidate_frames, args.seed)
            text_embedding = embedder.encode_text(text)
            image_embeddings = embedder.encode_images(
                [frame.image for frame in frames],
                args.image_batch_size,
            )
            similarities = (image_embeddings @ text_embedding.T).squeeze(-1)
            best_position = int(torch.argmax(similarities).item())
            best_frame = frames[best_position]
            best_score = float(similarities[best_position].item())

            save_jpeg_atomic(best_frame.image, image_path, args.jpeg_quality)
            metadata = {
                "dataset": dataset,
                "video_id": sample_id,
                "video_path": str(video_path),
                "retrieval_text": text,
                "text_fields": text_fields,
                "key_frame_path": str(image_path.resolve()),
                "frame_index": best_frame.frame_index,
                "timestamp_sec": round(best_frame.timestamp_sec, 6),
                "similarity": round(best_score, 8),
                "candidate_count": len(frames),
                "model_path": str(embedder.model_path),
                "backend": embedder.backend,
            }
            atomic_write_json(metadata_path, metadata)
            if error_path.exists():
                error_path.unlink()
            counts["processed"] += 1
        except Exception as error:
            counts["failed"] += 1
            atomic_write_json(
                error_path,
                {
                    "dataset": dataset,
                    "video_id": sample_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            tqdm.write(f"[{dataset}] failed {sample_id}: {type(error).__name__}: {error}")
            if args.fail_fast:
                raise

    counts["manifest"] = rebuild_manifest(metadata_dir, dataset_dir / "keyframes.jsonl")
    print(json.dumps({"dataset": dataset, **counts}, ensure_ascii=False))
    return counts


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_path = prepare_jina_offline_bundle(
        model_path=args.model_path.resolve(),
        clip_code_path=args.jina_clip_code_path,
        xlm_code_path=args.jina_xlm_code_path,
        text_config_path=args.jina_text_config_path,
        output_path=args.offline_model_dir,
    )
    embedder = ClipEmbedder(
        model_path=model_path,
        device=args.device,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        embedding_dim=args.embedding_dim,
    )
    total = {"processed": 0, "skipped": 0, "failed": 0}
    for dataset in selected_datasets(args.dataset):
        counts = run_dataset(dataset, args, embedder)
        for key in total:
            total[key] += counts[key]
    print(json.dumps({"total": total}, ensure_ascii=False))
    if total["failed"]:
        print("Some videos failed. See <output-root>/<dataset>/errors/*.json.", file=sys.stderr)


if __name__ == "__main__":
    main()
