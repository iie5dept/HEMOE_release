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
DEFAULT_MODEL_PATH = Path("/data2/573ops_ser/models/Qwen3-8B")
DEFAULT_SYSTEM_PROMPT = "Analyze the text evidence for news veracity detection."


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
            "Cache Qwen3 last-prompt-token states from the exact FakeTT/FakeSV "
            "text supplied to InternVL. Supports single-GPU and torchrun extraction."
        )
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "data" / "qwen3_text_features",
    )
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum chat-prompt length. Left truncation preserves appended post evidence.",
    )
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Keep Qwen3 thinking-mode prompt tokens. Disabled by default for deterministic features.",
    )
    parser.add_argument("--keep-video-placeholder", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
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
            raise RuntimeError("torchrun Qwen3 extraction requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=torch.device("cuda", local_rank),
            )
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(rank, local_rank, world_size, initialized)


def barrier(context: DistributedContext) -> None:
    if context.initialized:
        import torch.distributed as dist

        try:
            dist.barrier(device_ids=[context.local_rank])
        except TypeError:
            dist.barrier()


def distributed_sum(values: Sequence[int], context: DistributedContext) -> list[int]:
    if not context.initialized:
        return list(values)
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(values, dtype=torch.long, device=f"cuda:{context.local_rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [int(value) for value in tensor.cpu().tolist()]


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


def load_records(
    dataset: str,
    data_dir: Path,
    splits: Sequence[str],
    keep_video_placeholder: bool,
) -> list[TextRecord]:
    records: list[TextRecord] = []
    seen: set[str] = set()
    for split in splits:
        source_path = data_dir / f"{dataset}_{split}.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(f"Swift dataset split does not exist: {source_path}")
        with source_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("id") or row.get("video_id") or "").strip()
                if not sample_id:
                    raise ValueError(f"Missing sample ID at {source_path}:{line_number}")
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample ID across {dataset} splits: {sample_id}")
                seen.add(sample_id)
                records.append(
                    TextRecord(
                        dataset=dataset,
                        split=split,
                        sample_id=sample_id,
                        text=build_input_text(row.get("messages"), keep_video_placeholder),
                        source_path=source_path.resolve(),
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
            raise ValueError("float16 Qwen3 inference is not supported on CPU")
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


class Qwen3TextEncoder:
    def __init__(self, args: argparse.Namespace, device: str) -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.model_path = args.model_path.resolve()
        if not self.model_path.is_dir():
            raise NotADirectoryError(f"Local Qwen3 model directory does not exist: {self.model_path}")
        self.device = device
        self.compute_dtype = choose_compute_dtype(args.compute_dtype, device)
        self.storage_dtype = choose_storage_dtype(args.storage_dtype)
        self.system_prompt = args.system_prompt.strip()
        self.enable_thinking = bool(args.enable_thinking)
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": args.trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            use_fast=True,
            **load_kwargs,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Qwen3 tokenizer defines neither pad_token_id nor eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = args.truncation_side

        config = AutoConfig.from_pretrained(str(self.model_path), **load_kwargs)
        config._attn_implementation = args.attn_implementation
        config._attn_implementation_internal = args.attn_implementation
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            config=config,
            dtype=self.compute_dtype,
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=True,
            **load_kwargs,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        configured_max = int(getattr(config, "max_position_embeddings", args.max_length))
        self.max_length = min(args.max_length, configured_max)
        self.hidden_size = int(getattr(config, "hidden_size", 0))
        if self.hidden_size <= 0:
            raise ValueError("Qwen3 config does not define a positive hidden_size")
        if self.device.startswith("cuda"):
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
            "attention": getattr(self.model.config, "_attn_implementation", None),
            "enable_thinking": self.enable_thinking,
        }

    def build_prompt(self, text: str) -> str:
        conversation: list[dict[str, str]] = []
        if self.system_prompt:
            conversation.append({"role": "system", "content": self.system_prompt})
        conversation.append({"role": "user", "content": text})
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": self.enable_thinking,
        }
        try:
            return self.tokenizer.apply_chat_template(conversation, **kwargs)
        except TypeError as error:
            if self.enable_thinking:
                raise RuntimeError(
                    "The installed tokenizer does not support Qwen3 enable_thinking"
                ) from error
            kwargs.pop("enable_thinking")
            return self.tokenizer.apply_chat_template(conversation, **kwargs)

    def encode(self, texts: Sequence[str]) -> list[tuple[Any, dict[str, Any]]]:
        import torch

        prompts = [self.build_prompt(text) for text in texts]
        original = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        original_lengths = [len(ids) for ids in original["input_ids"]]
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_inputs = {name: value.to(self.device) for name, value in encoded.items()}
        attention_mask = model_inputs["attention_mask"].bool()
        if not attention_mask.any(dim=1).all():
            raise RuntimeError("Qwen3 tokenizer produced an empty prompt")
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                use_cache=False,
                return_dict=True,
            )
        states = outputs.last_hidden_state
        if states.ndim != 3 or states.shape[:2] != attention_mask.shape:
            raise RuntimeError(
                "Unexpected Qwen3 last_hidden_state shape: "
                f"states={tuple(states.shape)}, mask={tuple(attention_mask.shape)}"
            )
        reverse_offsets = attention_mask.flip(dims=(1,)).long().argmax(dim=1)
        last_indices = attention_mask.shape[1] - 1 - reverse_offsets
        batch_indices = torch.arange(states.shape[0], device=states.device)
        last_tokens = states[batch_indices, last_indices]
        if last_tokens.shape != (len(texts), self.hidden_size):
            raise RuntimeError(
                f"Unexpected Qwen3 last-token shape: {tuple(last_tokens.shape)}; "
                f"expected ({len(texts)}, {self.hidden_size})"
            )
        if not torch.isfinite(last_tokens.float()).all():
            raise RuntimeError("Qwen3 produced NaN or Inf features")

        saved_lengths = attention_mask.long().sum(dim=1).tolist()
        results = []
        for index, original_length in enumerate(original_lengths):
            state = last_tokens[index].detach().to("cpu", dtype=self.storage_dtype).contiguous()
            info = {
                "prompt": prompts[index],
                "original_token_count": int(original_length),
                "saved_token_count": int(saved_lengths[index]),
                "truncated": original_length > self.max_length,
            }
            results.append((state, info))
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


def save_encoded_item(
    item: WorkItem,
    dataset_dir: Path,
    encoder: Qwen3TextEncoder,
    state: Any,
    info: dict[str, Any],
) -> bool:
    atomic_save_safetensors(item.feature_path, {"text_last_token": state})
    text_hash = hashlib.sha256(item.record.text.encode("utf-8")).hexdigest()
    prompt_hash = hashlib.sha256(info["prompt"].encode("utf-8")).hexdigest()
    metadata = {
        "dataset": item.record.dataset,
        "split": item.record.split,
        "video_id": item.record.sample_id,
        "source_path": str(item.record.source_path),
        "feature_path": str(item.feature_path.resolve()),
        "feature_relpath": str(item.feature_path.relative_to(dataset_dir)),
        "model_path": str(encoder.model_path),
        "hidden_size": encoder.hidden_size,
        "storage_dtype": str(encoder.storage_dtype).replace("torch.", ""),
        "text_char_count": len(item.record.text),
        "text_sha256": text_hash,
        "prompt_sha256": prompt_hash,
        "original_token_count": info["original_token_count"],
        "saved_token_count": info["saved_token_count"],
        "truncated": info["truncated"],
    }
    atomic_write_json(item.metadata_path, metadata)
    if item.error_path.exists():
        item.error_path.unlink()
    return bool(info["truncated"])


def record_error(item: WorkItem, error: Exception) -> None:
    atomic_write_json(
        item.error_path,
        {
            "dataset": item.record.dataset,
            "video_id": item.record.sample_id,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def process_batch(
    items: Sequence[WorkItem],
    dataset_dir: Path,
    encoder: Qwen3TextEncoder,
    fail_fast: bool,
) -> tuple[int, int, int]:
    import torch

    try:
        encoded = encoder.encode([item.record.text for item in items])
    except Exception as batch_error:
        if len(items) == 1:
            record_error(items[0], batch_error)
            if fail_fast:
                raise
            return 0, 1, 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        processed = failed = truncated = 0
        for item in items:
            try:
                state, info = encoder.encode([item.record.text])[0]
                truncated += int(save_encoded_item(item, dataset_dir, encoder, state, info))
                processed += 1
            except Exception as error:
                failed += 1
                record_error(item, error)
                if fail_fast:
                    raise
        return processed, failed, truncated

    processed = truncated = 0
    for item, (state, info) in zip(items, encoded):
        truncated += int(save_encoded_item(item, dataset_dir, encoder, state, info))
        processed += 1
    return processed, 0, truncated


def rebuild_manifest(dataset_dir: Path) -> int:
    records: list[dict[str, Any]] = []
    for metadata_path in (dataset_dir / "metadata").glob("*.json"):
        with metadata_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        feature_path = dataset_dir / record["feature_relpath"]
        if feature_path.is_file():
            record["feature_path"] = str(feature_path.resolve())
            records.append(record)
    records.sort(key=lambda item: (str(item["split"]), str(item["video_id"])))
    manifest_path = dataset_dir / "qwen3_text_features.jsonl"
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
    encoder: Qwen3TextEncoder,
) -> dict[str, int]:
    from tqdm import tqdm

    data_dir = getattr(args, f"{dataset}_data_dir").resolve()
    records = load_records(dataset, data_dir, args.splits, args.keep_video_placeholder)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    dataset_dir = args.output_root.resolve() / dataset
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
    for items in progress:
        counts = process_batch(items, dataset_dir, encoder, args.fail_fast)
        processed += counts[0]
        failed += counts[1]
        truncated += counts[2]
    processed, failed, truncated = distributed_sum(
        [processed, failed, truncated], context
    )
    barrier(context)
    manifest_count = rebuild_manifest(dataset_dir) if context.is_main else 0
    barrier(context)
    return {
        "source": len(records),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "truncated": truncated,
        "manifest": manifest_count,
    }


def main() -> None:
    args = parse_args()
    import torch

    context = init_distributed()
    try:
        device = choose_device(args.device, context)
        encoder = Qwen3TextEncoder(args, device)
        if context.is_main:
            print(json.dumps({"qwen3": encoder.describe(), "world_size": context.world_size}, ensure_ascii=False))
        totals = {name: 0 for name in ("source", "processed", "skipped", "failed", "truncated")}
        for dataset in selected_datasets(args.dataset):
            result = run_dataset(dataset, args, context, encoder)
            if context.is_main:
                print(json.dumps({dataset: result}, ensure_ascii=False))
            for name in totals:
                totals[name] += result[name]
        if context.is_main:
            print(json.dumps({"total": totals}, ensure_ascii=False))
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        if context.initialized:
            import torch.distributed as dist

            dist.destroy_process_group()


if __name__ == "__main__":
    main()
