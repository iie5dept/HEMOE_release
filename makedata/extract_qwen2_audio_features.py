from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DATASETS = ("fakett", "fakesv")
SPLITS = ("train", "val", "test")
DEFAULT_MODEL_PATH = Path("/data2/573ops_ser/models/Qwen2-Audio-7B-Instruct")


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
class AudioRecord:
    dataset: str
    split: str
    sample_id: str
    video_path: Path
    text: str
    source_path: Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Cache Qwen2-Audio last-prompt-token states from FakeTT/FakeSV videos."
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--fakett-data-dir", type=Path, default=repo_root / "data" / "swift" / "fakett")
    parser.add_argument("--fakesv-data-dir", type=Path, default=repo_root / "data" / "swift" / "fakesv")
    parser.add_argument("--output-root", type=Path, default=repo_root / "data" / "qwen2_audio_features")
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--compute-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--storage-dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--fallback-audio-seconds", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    args.splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    invalid = sorted(set(args.splits) - set(SPLITS))
    if invalid:
        parser.error(f"Unsupported splits: {', '.join(invalid)}")
    if args.max_audio_seconds <= 0 or args.fallback_audio_seconds <= 0:
        parser.error("Audio durations must be positive")
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

        dist.barrier(device_ids=[context.local_rank])


def distributed_sum(values: Sequence[int], context: DistributedContext) -> list[int]:
    if not context.initialized:
        return list(values)
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(values, device=f"cuda:{context.local_rank}", dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [int(value) for value in tensor.cpu().tolist()]


def safe_stem(sample_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9._-]+", sample_id):
        return sample_id
    return "sample-" + hashlib.sha1(sample_id.encode("utf-8")).hexdigest()


def selected_datasets(dataset: str) -> list[str]:
    return list(DATASETS) if dataset == "all" else [dataset]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def message_content(record: dict[str, Any], role: str) -> str:
    for message in record.get("messages", []):
        if message.get("role") == role:
            content = str(message.get("content", ""))
            return content.replace("<video>", "").strip()
    raise ValueError(f"Sample {record.get('id')} has no {role} message")


def load_records(dataset: str, data_dir: Path, splits: list[str]) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    seen: set[str] = set()
    for split in splits:
        source_path = data_dir / f"{dataset}_{split}.jsonl"
        for row in read_jsonl(source_path):
            sample_id = str(row.get("id", "")).strip()
            videos = row.get("videos") or []
            if not sample_id or not videos:
                raise ValueError(f"Invalid sample in {source_path}: id={sample_id!r}, videos={videos!r}")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample ID {sample_id} across {data_dir}")
            seen.add(sample_id)
            records.append(
                AudioRecord(
                    dataset=dataset,
                    split=split,
                    sample_id=sample_id,
                    video_path=Path(videos[0]),
                    text=message_content(row, "user"),
                    source_path=source_path,
                )
            )
    return records


def decode_audio(
    video_path: Path,
    sampling_rate: int,
    max_seconds: float,
    fallback_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sampling_rate),
        "-t",
        str(max_seconds),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
        if result.returncode != 0 or audio.size == 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or "ffmpeg produced no audio samples")
        limit = int(round(sampling_rate * max_seconds))
        return audio[:limit], {
            "audio_fallback": False,
            "audio_samples": int(audio[:limit].size),
            "audio_seconds": float(audio[:limit].size / sampling_rate),
            "audio_truncated": audio.size >= limit,
            "decode_error": None,
        }
    except Exception as error:
        sample_count = max(1, int(round(sampling_rate * fallback_seconds)))
        return np.zeros(sample_count, dtype=np.float32), {
            "audio_fallback": True,
            "audio_samples": sample_count,
            "audio_seconds": float(fallback_seconds),
            "audio_truncated": False,
            "decode_error": str(error),
        }


class Qwen2AudioEncoder:
    @staticmethod
    def _replace_language_model_heads(model: Any) -> list[str]:
        """Expose decoder hidden states through logits on old and new Qwen2-Audio layouts."""
        import torch

        head_paths = [
            name
            for name, module in model.named_modules()
            if name and name.rsplit(".", 1)[-1] == "lm_head" and not isinstance(module, torch.nn.Identity)
        ]
        for head_path in head_paths:
            if "." in head_path:
                parent_path, attribute = head_path.rsplit(".", 1)
                parent = model.get_submodule(parent_path)
            else:
                parent, attribute = model, head_path
            setattr(parent, attribute, torch.nn.Identity())

        if not head_paths:
            raise RuntimeError(
                "Could not locate a Qwen2-Audio lm_head. The installed Transformers model layout "
                "is unsupported; print(model) and verify the local checkpoint architecture."
            )
        return head_paths

    def __init__(self, args: argparse.Namespace, device: str) -> None:
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen2AudioForConditionalGeneration

        dtype = getattr(torch, args.compute_dtype)
        self.storage_dtype = getattr(torch, args.storage_dtype)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            str(args.model_path), trust_remote_code=True, local_files_only=True
        )
        self.processor.tokenizer.padding_side = "left"
        config = AutoConfig.from_pretrained(
            str(args.model_path), trust_remote_code=True, local_files_only=True
        )
        for nested_config in (
            config,
            getattr(config, "audio_config", None),
            getattr(config, "text_config", None),
        ):
            if nested_config is not None:
                nested_config._attn_implementation = args.attn_implementation
                nested_config._attn_implementation_internal = args.attn_implementation
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            str(args.model_path),
            config=config,
            local_files_only=True,
            low_cpu_mem_usage=True,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
        ).to(device)
        # Qwen2-Audio moved lm_head between the outer model and its internal language model
        # across Transformers versions. Replace only heads that exist in the loaded module tree.
        self.replaced_lm_heads = self._replace_language_model_heads(self.model)
        self.model.eval()
        self.sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        self.hidden_size = int(self.model.config.text_config.hidden_size)
        self.compute_dtype = dtype
        self.autocast_enabled = self.device.startswith("cuda") and dtype != torch.float32
        self.attention_implementations = {
            "outer": getattr(self.model.config, "_attn_implementation", None),
            "audio": getattr(self.model.config.audio_config, "_attn_implementation", None),
            "text": getattr(self.model.config.text_config, "_attn_implementation", None),
        }

    def prompt(self, record: AudioRecord) -> str:
        conversation = [
            {
                "role": "system",
                "content": "Analyze the audio evidence and accompanying text for news veracity detection.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(record.video_path)},
                    {"type": "text", "text": record.text},
                ],
            },
        ]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    def encode(self, record: AudioRecord, audio: np.ndarray) -> Any:
        import torch

        prompt = self.prompt(record)
        inputs = self.processor(
            text=[prompt],
            audio=[audio],
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = {}
        for name, value in inputs.items():
            if torch.is_floating_point(value):
                value = value.to(device=self.device, dtype=self.compute_dtype)
            else:
                value = value.to(self.device)
            model_inputs[name] = value
        input_ids = model_inputs.get("input_ids")
        input_embeddings = self.model.get_input_embeddings()
        if input_ids is not None and input_embeddings is not None:
            minimum_id = int(input_ids.min().item())
            maximum_id = int(input_ids.max().item())
            vocabulary_size = int(input_embeddings.num_embeddings)
            if minimum_id < 0 or maximum_id >= vocabulary_size:
                raise RuntimeError(
                    "Qwen2-Audio tokenizer/model vocabulary mismatch: "
                    f"token range=[{minimum_id}, {maximum_id}], embeddings={vocabulary_size}. "
                    "Download processor and model files from the same checkpoint directory."
                )
            max_positions = int(self.model.config.text_config.max_position_embeddings)
            if input_ids.shape[1] > max_positions:
                raise RuntimeError(
                    f"Qwen2-Audio prompt length {input_ids.shape[1]} exceeds max positions {max_positions}"
                )
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        with torch.inference_mode(), torch.autocast(
            device_type, dtype=self.compute_dtype, enabled=self.autocast_enabled
        ):
            outputs = self.model(**model_inputs, use_cache=False, return_dict=True)
        last_token = outputs.logits[:, -1].squeeze(0)
        if last_token.shape != (self.hidden_size,):
            raise RuntimeError(
                "Unexpected Qwen2-Audio state shape after replacing "
                f"{self.replaced_lm_heads}: {tuple(last_token.shape)}; expected ({self.hidden_size},)"
            )
        if not torch.isfinite(last_token.float()).all():
            raise RuntimeError("Qwen2-Audio produced NaN or Inf")
        return last_token.detach().to("cpu", dtype=self.storage_dtype).contiguous(), prompt


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_safetensors(path: Path, tensors: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(tensors, str(temporary))
    temporary.replace(path)


def rebuild_manifest(dataset_dir: Path) -> int:
    records = []
    for metadata_path in (dataset_dir / "metadata").glob("*.json"):
        with metadata_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        feature_path = dataset_dir / record["feature_relpath"]
        if feature_path.is_file():
            record["feature_path"] = str(feature_path.resolve())
            records.append(record)
    records.sort(key=lambda item: (item["split"], item["video_id"]))
    path = dataset_dir / "qwen2_audio_features.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return len(records)


def run_dataset(
    dataset: str,
    args: argparse.Namespace,
    context: DistributedContext,
    encoder: Qwen2AudioEncoder,
) -> dict[str, int]:
    from tqdm import tqdm

    data_dir = getattr(args, f"{dataset}_data_dir").resolve()
    records = load_records(dataset, data_dir, args.splits)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    dataset_dir = args.output_root.resolve() / dataset
    for name in ("features", "metadata", "errors"):
        (dataset_dir / name).mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = 0
    for record in records:
        stem = safe_stem(record.sample_id)
        feature_path = dataset_dir / "features" / f"{stem}.safetensors"
        metadata_path = dataset_dir / "metadata" / f"{stem}.json"
        if feature_path.is_file() and metadata_path.is_file() and not args.overwrite:
            skipped += 1
        else:
            pending.append((record, feature_path, metadata_path, dataset_dir / "errors" / f"{stem}.json"))
    rank_items = pending[context.rank :: context.world_size]
    processed = failed = fallback = truncated = 0
    progress = tqdm(rank_items, desc=f"{dataset}[rank{context.rank}]", disable=not context.is_main)
    for record, feature_path, metadata_path, error_path in progress:
        try:
            audio, audio_info = decode_audio(
                record.video_path,
                encoder.sampling_rate,
                args.max_audio_seconds,
                args.fallback_audio_seconds,
            )
            state, prompt = encoder.encode(record, audio)
            atomic_safetensors(feature_path, {"audio_last_token": state})
            metadata = {
                "dataset": dataset,
                "split": record.split,
                "video_id": record.sample_id,
                "source_path": str(record.source_path),
                "video_path": str(record.video_path),
                "feature_relpath": str(feature_path.relative_to(dataset_dir)),
                "feature_path": str(feature_path.resolve()),
                "model_path": str(args.model_path.resolve()),
                "sampling_rate": encoder.sampling_rate,
                "hidden_size": encoder.hidden_size,
                "storage_dtype": args.storage_dtype,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                **audio_info,
            }
            atomic_json(metadata_path, metadata)
            if error_path.exists():
                error_path.unlink()
            processed += 1
            fallback += int(audio_info["audio_fallback"])
            truncated += int(audio_info["audio_truncated"])
        except Exception as error:
            failed += 1
            if failed <= 3:
                print(
                    f"[{dataset} rank{context.rank}] failed {record.sample_id}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            atomic_json(
                error_path,
                {"dataset": dataset, "video_id": record.sample_id, "error": str(error), "type": type(error).__name__},
            )
            if args.fail_fast:
                raise
    processed, failed, fallback, truncated = distributed_sum(
        [processed, failed, fallback, truncated], context
    )
    barrier(context)
    manifest = rebuild_manifest(dataset_dir) if context.is_main else 0
    barrier(context)
    return {
        "source": len(records),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "fallback": fallback,
        "truncated": truncated,
        "manifest": manifest,
    }


def main() -> None:
    import torch

    args = parse_args()
    context = init_distributed()
    device = f"cuda:{context.local_rank}" if torch.cuda.is_available() else "cpu"
    encoder = Qwen2AudioEncoder(args, device)
    if context.is_main:
        print(
            json.dumps(
                {
                    "model": str(args.model_path),
                    "hidden_size": encoder.hidden_size,
                    "sampling_rate": encoder.sampling_rate,
                    "attention": encoder.attention_implementations,
                    "replaced_lm_heads": encoder.replaced_lm_heads,
                    "world_size": context.world_size,
                },
                ensure_ascii=False,
            )
        )
    totals = {name: 0 for name in ("source", "processed", "skipped", "failed", "fallback", "truncated")}
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
    torch.cuda.empty_cache()
    if context.initialized:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
