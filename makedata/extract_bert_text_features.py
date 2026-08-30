from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASETS = ("fakett", "fakesv")
SPLITS = ("train", "val", "test")
DEFAULT_MODELS = {
    "fakett": Path("/data2/573ops_ser/models/bert-base-uncased"),
    "fakesv": Path("/data2/573ops_ser/models/chinese-macbert-base"),
}


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
class TextRecord:
    dataset: str
    split: str
    sample_id: str
    text: str
    source_path: Path


@dataclass(frozen=True)
class WorkItem:
    record: TextRecord
    feature_path: Path
    metadata_path: Path
    error_path: Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Cache frozen English BERT and Chinese MacBERT token features from "
            "the exact user messages used by InternVL. Supports torchrun."
        )
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument(
        "--fakett-data-dir",
        type=Path,
        default=repo_root / "data" / "swift" / "fakett",
    )
    parser.add_argument(
        "--fakesv-data-dir",
        type=Path,
        default=repo_root / "data" / "swift" / "fakesv",
    )
    parser.add_argument("--fakett-model-path", type=Path, default=DEFAULT_MODELS["fakett"])
    parser.add_argument("--fakesv-model-path", type=Path, default=DEFAULT_MODELS["fakesv"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "data" / "bert_features",
    )
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--truncation-side",
        choices=("left", "right"),
        default="left",
        help="Left truncation preserves evidence appended after the task instruction.",
    )
    parser.add_argument("--device", default="auto")
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
    parser.add_argument("--keep-video-placeholder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    invalid_splits = sorted(set(requested_splits) - set(SPLITS))
    if invalid_splits:
        parser.error(f"Unsupported splits: {', '.join(invalid_splits)}")
    if not requested_splits:
        parser.error("--splits cannot be empty")
    args.splits = requested_splits
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
            raise RuntimeError("torchrun text extraction requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(rank, local_rank, world_size, initialized)


def barrier(context: DistributedContext) -> None:
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


def safe_stem(sample_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9._-]+", sample_id):
        return sample_id
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return f"sample-{digest}"


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def build_input_text(messages: Any, keep_video_placeholder: bool) -> str:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in {"system", "user"}:
            continue
        text = extract_message_text(message.get("content"))
        if text:
            parts.append(text)
    text = "\n".join(parts)
    if not keep_video_placeholder:
        text = re.sub(r"<video>", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        raise ValueError("No system/user text remains after removing media placeholders")
    return text


def load_split_records(
    dataset: str,
    data_dir: Path,
    splits: Sequence[str],
    keep_video_placeholder: bool,
) -> list[TextRecord]:
    records: list[TextRecord] = []
    seen: set[str] = set()
    for split in splits:
        path = data_dir / f"{dataset}_{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Swift dataset split does not exist: {path}")
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("id") or row.get("video_id") or "").strip()
                if not sample_id:
                    raise ValueError(f"Missing sample id at {path}:{line_number}")
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample id across {dataset} splits: {sample_id}")
                seen.add(sample_id)
                records.append(
                    TextRecord(
                        dataset=dataset,
                        split=split,
                        sample_id=sample_id,
                        text=build_input_text(row.get("messages"), keep_video_placeholder),
                        source_path=path.resolve(),
                    )
                )
    return records


def choose_device(requested: str, context: DistributedContext) -> str:
    import torch

    if context.initialized:
        if requested == "cpu":
            raise ValueError("--device cpu cannot be used with NCCL torchrun")
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
            raise ValueError("float16 BERT inference is not supported on CPU")
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def choose_storage_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


class BertEncoder:
    def __init__(
        self,
        model_path: Path,
        device: str,
        compute_dtype_name: str,
        storage_dtype_name: str,
        max_length: int,
        truncation_side: str,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise NotADirectoryError(f"Local BERT model directory does not exist: {self.model_path}")
        self.device = device
        self.compute_dtype = choose_compute_dtype(compute_dtype_name, device)
        self.storage_dtype = choose_storage_dtype(storage_dtype_name)
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            use_fast=True,
            **load_kwargs,
        )
        self.tokenizer.truncation_side = truncation_side
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            dtype=self.compute_dtype,
            **load_kwargs,
        ).to(self.device).eval()
        configured_max = int(getattr(self.model.config, "max_position_embeddings", max_length))
        self.max_length = min(max_length, configured_max)
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        if self.hidden_size <= 0:
            raise ValueError("BERT config does not define a positive hidden_size")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    def describe(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "storage_dtype": str(self.storage_dtype).replace("torch.", ""),
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
            "truncation_side": self.tokenizer.truncation_side,
        }

    def encode(self, texts: Sequence[str]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        import torch

        original_encodings = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        original_lengths = [len(ids) for ids in original_encodings["input_ids"]]
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special_tokens_mask = encoded.pop("special_tokens_mask")
        model_inputs = {name: value.to(self.device) for name, value in encoded.items()}
        autocast_enabled = self.device.startswith("cuda") and self.compute_dtype != torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if self.device.startswith("cuda") else "cpu",
            dtype=self.compute_dtype,
            enabled=autocast_enabled,
        ):
            outputs = self.model(**model_inputs, return_dict=True)
        states = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]
        results: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for index, original_length in enumerate(original_lengths):
            valid_length = int(attention_mask[index].sum().item())
            token_states = states[index, :valid_length]
            specials = special_tokens_mask[index, :valid_length].to(token_states.device).bool()
            content_mask = ~specials
            if not content_mask.any():
                content_mask = torch.ones_like(content_mask)
            content_mean = token_states[content_mask].float().mean(dim=0).to(token_states.dtype)
            cls_token = token_states[0]
            global_feature = torch.cat([cls_token, content_mean], dim=-1)
            tensors = {
                "token_states": token_states.detach().to("cpu", dtype=self.storage_dtype).contiguous(),
                "attention_mask": torch.ones(valid_length, dtype=torch.bool),
                "special_tokens_mask": specials.detach().to("cpu").contiguous(),
                "cls_token": cls_token.detach().to("cpu", dtype=self.storage_dtype).contiguous(),
                "content_mean": content_mean.detach().to("cpu", dtype=self.storage_dtype).contiguous(),
                "global_feature": global_feature.detach()
                .to("cpu", dtype=self.storage_dtype)
                .contiguous(),
            }
            if not all(torch.isfinite(tensor.float()).all().item() for tensor in tensors.values()):
                raise RuntimeError("BERT produced NaN or Inf features")
            info = {
                "original_token_count": original_length,
                "saved_token_count": valid_length,
                "truncated": original_length > self.max_length,
            }
            results.append((tensors, info))
        return results


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


def build_work_items(
    records: Sequence[TextRecord],
    dataset_dir: Path,
    overwrite: bool,
) -> tuple[list[WorkItem], int]:
    for name in ("features", "metadata", "errors"):
        (dataset_dir / name).mkdir(parents=True, exist_ok=True)
    pending: list[WorkItem] = []
    skipped = 0
    for record in records:
        stem = safe_stem(record.sample_id)
        feature_path = dataset_dir / "features" / f"{stem}.safetensors"
        metadata_path = dataset_dir / "metadata" / f"{stem}.json"
        if feature_path.is_file() and metadata_path.is_file() and not overwrite:
            skipped += 1
            continue
        pending.append(
            WorkItem(
                record=record,
                feature_path=feature_path,
                metadata_path=metadata_path,
                error_path=dataset_dir / "errors" / f"{stem}.json",
            )
        )
    return pending, skipped


def batched(items: Sequence[WorkItem], batch_size: int) -> Iterable[Sequence[WorkItem]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def process_batch(
    items: Sequence[WorkItem],
    dataset_dir: Path,
    encoder: BertEncoder,
    fail_fast: bool,
) -> tuple[int, int, int]:
    encoded = encoder.encode([item.record.text for item in items])
    processed = 0
    failed = 0
    truncated = 0
    for item, (tensors, info) in zip(items, encoded):
        try:
            atomic_save_safetensors(item.feature_path, tensors)
            text_hash = hashlib.sha256(item.record.text.encode("utf-8")).hexdigest()
            metadata = {
                "dataset": item.record.dataset,
                "split": item.record.split,
                "video_id": item.record.sample_id,
                "source_path": str(item.record.source_path),
                "feature_path": str(item.feature_path.resolve()),
                "feature_relpath": str(item.feature_path.relative_to(dataset_dir)),
                "model_path": str(encoder.model_path),
                "text_sha256": text_hash,
                "text_char_count": len(item.record.text),
                "storage_dtype": str(encoder.storage_dtype).replace("torch.", ""),
                "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
                **info,
            }
            atomic_write_json(item.metadata_path, metadata)
            if item.error_path.exists():
                item.error_path.unlink()
            processed += 1
            truncated += int(info["truncated"])
        except Exception as error:
            failed += 1
            atomic_write_json(
                item.error_path,
                {
                    "dataset": item.record.dataset,
                    "video_id": item.record.sample_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if fail_fast:
                raise
    return processed, failed, truncated


def rebuild_manifest(dataset_dir: Path) -> int:
    records: list[dict[str, Any]] = []
    for path in (dataset_dir / "metadata").glob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        feature_path = dataset_dir / record["feature_relpath"]
        if feature_path.is_file():
            record["feature_path"] = str(feature_path.resolve())
            records.append(record)
    records.sort(key=lambda record: (str(record["split"]), str(record["video_id"])))
    manifest_path = dataset_dir / "bert_features.jsonl"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    return len(records)


def run_dataset(
    dataset: str,
    args: argparse.Namespace,
    context: DistributedContext,
    device: str,
) -> dict[str, int]:
    from tqdm import tqdm

    data_dir = getattr(args, f"{dataset}_data_dir").resolve()
    records = load_split_records(
        dataset,
        data_dir,
        args.splits,
        keep_video_placeholder=args.keep_video_placeholder,
    )
    if args.max_samples > 0:
        records = records[: args.max_samples]
    model_path = getattr(args, f"{dataset}_model_path")
    encoder = BertEncoder(
        model_path=model_path,
        device=device,
        compute_dtype_name=args.compute_dtype,
        storage_dtype_name=args.storage_dtype,
        max_length=args.max_length,
        truncation_side=args.truncation_side,
        trust_remote_code=args.trust_remote_code,
    )
    if context.is_main:
        print(json.dumps({dataset: encoder.describe(), "world_size": context.world_size}, ensure_ascii=False))

    dataset_dir = args.output_root.resolve() / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pending, skipped = build_work_items(records, dataset_dir, args.overwrite)
    barrier(context)
    rank_items = pending[context.rank :: context.world_size]
    processed = failed = truncated = 0
    progress = tqdm(
        list(batched(rank_items, args.batch_size)),
        desc=f"{dataset}[rank{context.rank}]",
        unit="batch",
        disable=not context.is_main,
    )
    for batch in progress:
        values = process_batch(batch, dataset_dir, encoder, args.fail_fast)
        processed += values[0]
        failed += values[1]
        truncated += values[2]
    processed, failed, truncated = distributed_sum(
        [processed, failed, truncated], context
    )
    barrier(context)
    manifest_count = rebuild_manifest(dataset_dir) if context.is_main else 0
    barrier(context)
    result = {
        "source": len(records),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "truncated": truncated,
        "manifest": manifest_count,
    }
    if context.is_main:
        print(json.dumps({"dataset": dataset, **result}, ensure_ascii=False))

    del encoder
    gc.collect()
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    context = init_distributed()
    try:
        device = choose_device(args.device, context)
        totals = {
            "source": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "truncated": 0,
            "manifest": 0,
        }
        for dataset in selected_datasets(args.dataset):
            result = run_dataset(dataset, args, context, device)
            if context.is_main:
                for name in totals:
                    totals[name] += result[name]
        if context.is_main:
            print(json.dumps({"total": totals}, ensure_ascii=False))
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()
