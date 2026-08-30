from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MODEL_PATH = Path("/data2/573ops_ser/models/dinov3-vitl16-pretrain-lvd1689m")
DATASETS = ("fakett", "fakesv")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    initialized: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class WorkItem:
    record: dict[str, Any]
    sample_id: str
    image_path: Path
    feature_path: Path
    metadata_path: Path
    error_path: Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    keyframe_root = repo_root / "data" / "key_frames"
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen DINOv3 CLS/global and patch-token features from "
            "FakeTT/FakeSV key frames. Supports single-process and torchrun."
        )
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--fakett-manifest",
        type=Path,
        default=keyframe_root / "fakett" / "keyframes.jsonl",
    )
    parser.add_argument(
        "--fakesv-manifest",
        type=Path,
        default=keyframe_root / "fakesv" / "keyframes.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "data" / "dinov3_features",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--no-patch-tokens",
        action="store_true",
        help="Only save global vectors. The planned Transformer branch needs patch tokens.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    return args


def init_distributed() -> DistributedContext:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized = world_size > 1
    if initialized:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun multi-process extraction requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        initialized=initialized,
    )


def distributed_barrier(context: DistributedContext) -> None:
    if context.initialized:
        import torch.distributed as dist

        dist.barrier()


def distributed_sum(values: Sequence[int], context: DistributedContext) -> list[int]:
    if not context.initialized:
        return list(values)
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(values, dtype=torch.long, device=f"cuda:{context.local_rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [int(value) for value in tensor.cpu().tolist()]


def destroy_distributed(context: DistributedContext) -> None:
    if context.initialized:
        import torch.distributed as dist

        dist.destroy_process_group()


def selected_datasets(name: str) -> list[str]:
    return list(DATASETS) if name == "all" else [name]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Key-frame manifest does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def validate_records(records: Sequence[dict[str, Any]], dataset: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        sample_id = str(record.get("video_id") or "").strip()
        if not sample_id:
            raise ValueError(f"A {dataset} key-frame record has no video_id")
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate {dataset} video ids in manifest: {preview}")


def safe_stem(sample_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9._-]+", sample_id):
        return sample_id
    import hashlib

    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return f"sample-{digest}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_save_safetensors(path: Path, tensors: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(tensors, str(temporary))
    temporary.replace(path)


def choose_device(requested: str, context: DistributedContext) -> str:
    import torch

    if context.initialized:
        if requested == "cpu":
            raise ValueError("--device cpu cannot be used with NCCL torchrun extraction")
        return f"cuda:{context.local_rank}"
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        return "cuda:0"
    return requested


def choose_compute_dtype(name: str, device: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "float16":
        if device == "cpu":
            raise ValueError("float16 DINOv3 inference is not supported on CPU")
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def storage_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Unsupported DINOv3 {name}: {value!r}")


class DinoV3Encoder:
    def __init__(
        self,
        model_path: Path,
        device: str,
        compute_dtype_name: str,
        storage_dtype_name: str,
        trust_remote_code: bool,
        save_patch_tokens: bool,
    ) -> None:
        import torch

        try:
            import transformers
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError(
                "DINOv3 requires a recent Transformers installation (>=4.56)."
            ) from error

        version_match = re.match(r"^(\d+)\.(\d+)", transformers.__version__)
        if version_match is None:
            raise RuntimeError(
                f"Cannot parse Transformers version: {transformers.__version__!r}"
            )
        transformers_version = tuple(int(part) for part in version_match.groups())
        if transformers_version < (4, 56):
            raise RuntimeError(
                "Native DINOv3 is unavailable in the installed Transformers version: "
                f"{transformers.__version__}. Use transformers>=4.56 in a separate "
                "feature-extraction environment."
            )

        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise NotADirectoryError(f"Local DINOv3 model directory does not exist: {self.model_path}")
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"DINOv3 config is missing: {self.model_path / 'config.json'}")

        self.device = device
        self.compute_dtype = choose_compute_dtype(compute_dtype_name, device)
        self.storage_dtype = storage_dtype(storage_dtype_name)
        self.save_patch_tokens = save_patch_tokens
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
        }
        try:
            self.processor = AutoImageProcessor.from_pretrained(str(self.model_path), **load_kwargs)
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                dtype=self.compute_dtype,
                **load_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load the local DINOv3 checkpoint. Verify that the snapshot contains "
                "config.json, preprocessor_config.json, and model weights, and use "
                "transformers>=4.56 for native DINOv3 support."
            ) from error
        self.model.to(self.device).eval()

        config = self.model.config
        self.patch_size = pair(getattr(config, "patch_size", 16), "patch_size")
        self.hidden_size = int(getattr(config, "hidden_size", 0))
        configured_registers = getattr(config, "num_register_tokens", None)
        self.configured_registers = (
            int(configured_registers) if configured_registers is not None else None
        )
        if self.hidden_size <= 0:
            raise ValueError("DINOv3 config does not define a positive hidden_size")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    def describe(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "storage_dtype": str(self.storage_dtype).replace("torch.", ""),
            "hidden_size": self.hidden_size,
            "patch_size": list(self.patch_size),
            "configured_register_tokens": self.configured_registers,
            "save_patch_tokens": self.save_patch_tokens,
        }

    def encode(self, images: Sequence[Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        import torch

        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None or pixel_values.ndim != 4:
            raise RuntimeError("DINOv3 image processor did not return BCHW pixel_values")
        height, width = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        patch_height, patch_width = self.patch_size
        if height % patch_height or width % patch_width:
            raise ValueError(
                f"Processed image size {(height, width)} is not divisible by patch size {self.patch_size}"
            )
        expected_patch_count = (height // patch_height) * (width // patch_width)
        model_inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        autocast_enabled = self.device.startswith("cuda") and self.compute_dtype != torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if self.device.startswith("cuda") else "cpu",
            dtype=self.compute_dtype,
            enabled=autocast_enabled,
        ):
            outputs = self.model(**model_inputs, return_dict=True)

        sequence = getattr(outputs, "last_hidden_state", None)
        if sequence is None or sequence.ndim != 3:
            shape = None if sequence is None else tuple(sequence.shape)
            raise RuntimeError(f"Expected DINOv3 last_hidden_state [B,L,D], received {shape}")
        special_token_count = int(sequence.shape[1]) - expected_patch_count
        if special_token_count < 1:
            raise RuntimeError(
                f"Cannot split DINOv3 tokens: sequence={sequence.shape[1]}, patches={expected_patch_count}"
            )
        register_token_count = special_token_count - 1
        if (
            self.configured_registers is not None
            and self.configured_registers != register_token_count
        ):
            raise RuntimeError(
                "DINOv3 token layout disagrees with config: "
                f"inferred {register_token_count} register tokens, config says "
                f"{self.configured_registers}."
            )

        cls_tokens = sequence[:, 0]
        patch_tokens = sequence[:, special_token_count:]
        patch_mean = patch_tokens.float().mean(dim=1).to(sequence.dtype)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = cls_tokens
        global_features = torch.cat([pooled, patch_mean], dim=-1)

        feature_batches: list[dict[str, Any]] = []
        for index in range(sequence.shape[0]):
            tensors = {
                "cls_token": cls_tokens[index].detach().to("cpu", dtype=self.storage_dtype).contiguous(),
                "pooler_output": pooled[index].detach().to("cpu", dtype=self.storage_dtype).contiguous(),
                "patch_mean": patch_mean[index].detach().to("cpu", dtype=self.storage_dtype).contiguous(),
                "global_feature": global_features[index]
                .detach()
                .to("cpu", dtype=self.storage_dtype)
                .contiguous(),
            }
            if self.save_patch_tokens:
                tensors["patch_tokens"] = (
                    patch_tokens[index]
                    .detach()
                    .to("cpu", dtype=self.storage_dtype)
                    .contiguous()
                )
            if not all(torch.isfinite(tensor.float()).all().item() for tensor in tensors.values()):
                raise RuntimeError("DINOv3 produced NaN or Inf features")
            feature_batches.append(tensors)
        layout = {
            "image_height": height,
            "image_width": width,
            "patch_count": expected_patch_count,
            "special_token_count": special_token_count,
            "register_token_count": register_token_count,
        }
        return feature_batches, layout


def load_rgb_image(path: Path):
    from PIL import Image, ImageOps

    if not path.is_file():
        raise FileNotFoundError(f"Key-frame image does not exist: {path}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def batched(items: Sequence[WorkItem], batch_size: int) -> Iterable[Sequence[WorkItem]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def feature_shapes(tensors: dict[str, Any]) -> dict[str, list[int]]:
    return {name: list(tensor.shape) for name, tensor in tensors.items()}


def build_work_items(
    records: Sequence[dict[str, Any]],
    dataset_dir: Path,
    overwrite: bool,
) -> tuple[list[WorkItem], int]:
    feature_dir = dataset_dir / "features"
    metadata_dir = dataset_dir / "metadata"
    error_dir = dataset_dir / "errors"
    for directory in (feature_dir, metadata_dir, error_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pending: list[WorkItem] = []
    skipped = 0
    for record in records:
        sample_id = str(record["video_id"]).strip()
        stem = safe_stem(sample_id)
        feature_path = feature_dir / f"{stem}.safetensors"
        metadata_path = metadata_dir / f"{stem}.json"
        if feature_path.is_file() and metadata_path.is_file() and not overwrite:
            skipped += 1
            continue
        pending.append(
            WorkItem(
                record=record,
                sample_id=sample_id,
                image_path=Path(str(record.get("key_frame_path") or "")),
                feature_path=feature_path,
                metadata_path=metadata_path,
                error_path=error_dir / f"{stem}.json",
            )
        )
    return pending, skipped


def process_batch(
    items: Sequence[WorkItem],
    dataset: str,
    dataset_dir: Path,
    source_manifest: Path,
    encoder: DinoV3Encoder,
    fail_fast: bool,
) -> tuple[int, int]:
    valid_items: list[WorkItem] = []
    images: list[Any] = []
    failed = 0
    for item in items:
        try:
            images.append(load_rgb_image(item.image_path))
            valid_items.append(item)
        except Exception as error:
            failed += 1
            atomic_write_json(
                item.error_path,
                {
                    "dataset": dataset,
                    "video_id": item.sample_id,
                    "key_frame_path": str(item.image_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if fail_fast:
                raise
    if not valid_items:
        return 0, failed

    try:
        encoded, layout = encoder.encode(images)
    except Exception:
        # Encoder failures are normally model/configuration errors, not bad samples.
        # Stop immediately instead of writing the same misleading error for a whole shard.
        raise

    processed = 0
    for item, tensors in zip(valid_items, encoded):
        try:
            atomic_save_safetensors(item.feature_path, tensors)
            metadata = {
                "dataset": dataset,
                "video_id": item.sample_id,
                "key_frame_path": str(item.image_path),
                "source_manifest": str(source_manifest.resolve()),
                "feature_path": str(item.feature_path.resolve()),
                "feature_relpath": str(item.feature_path.relative_to(dataset_dir)),
                "model_path": str(encoder.model_path),
                "storage_dtype": str(encoder.storage_dtype).replace("torch.", ""),
                "tensor_shapes": feature_shapes(tensors),
                **layout,
            }
            for name in ("frame_index", "timestamp_sec", "similarity", "retrieval_text"):
                if name in item.record:
                    metadata[name] = item.record[name]
            atomic_write_json(item.metadata_path, metadata)
            if item.error_path.exists():
                item.error_path.unlink()
            processed += 1
        except Exception as error:
            failed += 1
            atomic_write_json(
                item.error_path,
                {
                    "dataset": dataset,
                    "video_id": item.sample_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if fail_fast:
                raise
    return processed, failed


def rebuild_manifest(dataset_dir: Path) -> int:
    metadata_dir = dataset_dir / "metadata"
    records: list[dict[str, Any]] = []
    for metadata_path in metadata_dir.glob("*.json"):
        with metadata_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        feature_path = dataset_dir / record["feature_relpath"]
        if feature_path.is_file():
            record["feature_path"] = str(feature_path.resolve())
            records.append(record)
    records.sort(key=lambda record: str(record["video_id"]))
    manifest_path = dataset_dir / "dinov3_features.jsonl"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    return len(records)


def run_dataset(
    dataset: str,
    args: argparse.Namespace,
    encoder: DinoV3Encoder,
    context: DistributedContext,
) -> dict[str, int]:
    from tqdm import tqdm

    source_manifest = getattr(args, f"{dataset}_manifest").resolve()
    records = load_jsonl(source_manifest)
    validate_records(records, dataset)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    dataset_dir = args.output_root.resolve() / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pending, skipped = build_work_items(records, dataset_dir, args.overwrite)
    # Freeze a shared cache snapshot before any worker starts writing. Without
    # this barrier a fast rank could change another rank's pending-list indices.
    distributed_barrier(context)
    rank_items = pending[context.rank :: context.world_size]

    processed = 0
    failed = 0
    batches = list(batched(rank_items, args.batch_size))
    progress = tqdm(
        batches,
        desc=f"{dataset}[rank{context.rank}]",
        unit="batch",
        disable=not context.is_main,
    )
    for batch in progress:
        batch_processed, batch_failed = process_batch(
            batch,
            dataset=dataset,
            dataset_dir=dataset_dir,
            source_manifest=source_manifest,
            encoder=encoder,
            fail_fast=args.fail_fast,
        )
        processed += batch_processed
        failed += batch_failed

    processed, failed = distributed_sum([processed, failed], context)
    # Every rank observes the same pre-sharding skip count, so do not all-reduce it.
    distributed_barrier(context)
    manifest_count = rebuild_manifest(dataset_dir) if context.is_main else 0
    distributed_barrier(context)
    result = {
        "source": len(records),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "manifest": manifest_count,
    }
    if context.is_main:
        print(json.dumps({"dataset": dataset, **result}, ensure_ascii=False))
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    context = init_distributed()
    try:
        device = choose_device(args.device, context)
        encoder = DinoV3Encoder(
            model_path=args.model_path,
            device=device,
            compute_dtype_name=args.compute_dtype,
            storage_dtype_name=args.storage_dtype,
            trust_remote_code=args.trust_remote_code,
            save_patch_tokens=not args.no_patch_tokens,
        )
        if context.is_main:
            print(json.dumps({"dinov3": encoder.describe(), "world_size": context.world_size}, ensure_ascii=False))

        totals = {"source": 0, "processed": 0, "skipped": 0, "failed": 0, "manifest": 0}
        for dataset in selected_datasets(args.dataset):
            result = run_dataset(dataset, args, encoder, context)
            if context.is_main:
                for name in totals:
                    totals[name] += result[name]
        if context.is_main:
            print(json.dumps({"total": totals}, ensure_ascii=False))
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()
