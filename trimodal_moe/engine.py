from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler, SequentialSampler

from .data import FourModalCollator, FourModalFeatureDataset, ID_TO_LABEL
from .internvl import build_trimodal_model, infer_num_image_tokens, load_internvl_with_lora


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def torch_dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}")
    return mapping[name]


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}


def model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values": batch["pixel_values"],
        "visual_tokens": batch["visual_tokens"],
        "visual_attention_mask": batch["visual_attention_mask"],
        "text_states": batch["text_states"],
        "audio_states": batch["audio_states"],
        "labels": batch["labels"],
    }


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data across ranks without padding or duplicate samples."""

    def __init__(self, dataset: Any, rank: int, world_size: int) -> None:
        self.dataset_size = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.dataset_size:
            return 0
        return (self.dataset_size - self.rank + self.world_size - 1) // self.world_size


def build_model_and_tokenizer(config: dict[str, Any], checkpoint: str | Path | None = None):
    model_cfg = config["model"]
    fusion_config = dict(config.get("fusion", {}))
    checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
    if checkpoint_path is not None:
        adapter_config = checkpoint_path / "language_lora" / "adapter_config.json"
        classifier_path = checkpoint_path / "classifier.safetensors"
        missing_files = [path for path in (adapter_config, classifier_path) if not path.is_file()]
        if missing_files:
            missing_text = ", ".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                f"Incomplete four-modal checkpoint at {checkpoint_path}; missing: {missing_text}. "
                "A training process started before checkpoint-best support only writes "
                "checkpoint-epoch-N directories."
            )
    audio_manifest = config.get("data", {}).get("audio_manifest")
    if audio_manifest:
        with Path(audio_manifest).open("r", encoding="utf-8") as handle:
            first_record = next((json.loads(line) for line in handle if line.strip()), None)
        if first_record is None:
            raise ValueError(f"Empty Qwen2-Audio manifest: {audio_manifest}")
        cached_audio_dim = int(first_record["hidden_size"])
        configured_audio_dim = int(fusion_config.get("audio_dim", cached_audio_dim))
        if configured_audio_dim != cached_audio_dim:
            raise ValueError(
                f"audio_dim={configured_audio_dim} but Qwen2-Audio cache uses {cached_audio_dim}"
            )
        fusion_config["audio_dim"] = cached_audio_dim
    text_manifest = config.get("data", {}).get("text_manifest")
    if text_manifest:
        with Path(text_manifest).open("r", encoding="utf-8") as handle:
            first_record = next((json.loads(line) for line in handle if line.strip()), None)
        if first_record is None:
            raise ValueError(f"Empty Qwen3 text manifest: {text_manifest}")
        cached_text_dim = int(first_record["hidden_size"])
        configured_text_dim = int(fusion_config.get("text_dim", cached_text_dim))
        if configured_text_dim != cached_text_dim:
            raise ValueError(
                f"text_dim={configured_text_dim} but Qwen3 text cache uses {cached_text_dim}"
            )
        fusion_config["text_dim"] = cached_text_dim
    adapter_path = checkpoint_path / "language_lora" if checkpoint_path else None
    internvl, tokenizer = load_internvl_with_lora(
        model_path=model_cfg["path"],
        dtype=torch_dtype(model_cfg.get("dtype", "bfloat16")),
        lora_rank=int(model_cfg.get("lora_rank", 8)),
        lora_alpha=int(model_cfg.get("lora_alpha", 32)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        use_flash_attn=bool(model_cfg.get("use_flash_attn", True)),
        adapter_path=adapter_path,
    )
    model = build_trimodal_model(internvl, tokenizer, fusion_config)
    if checkpoint_path:
        classifier_path = checkpoint_path / "classifier.safetensors"
        missing, unexpected = model.classifier.load_state_dict(load_file(str(classifier_path)), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Classifier checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model, tokenizer


def build_loader(
    config: dict[str, Any],
    split: str,
    tokenizer: Any,
    model: nn.Module,
    rank: int,
    world_size: int,
    shuffle: bool,
) -> DataLoader:
    data_cfg = config["data"]
    data_path = data_cfg[f"{split}_path"]
    dataset = FourModalFeatureDataset(
        data_path=data_path,
        visual_manifest=data_cfg["visual_manifest"],
        text_manifest=data_cfg["text_manifest"],
        audio_manifest=data_cfg["audio_manifest"],
    )
    if world_size > 1:
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            if shuffle
            else DistributedEvalSampler(dataset, rank=rank, world_size=world_size)
        )
    else:
        sampler = None if shuffle else SequentialSampler(dataset)
    collator = FourModalCollator(
        tokenizer=tokenizer,
        system_prompt=str(data_cfg.get("system_prompt", "")),
        num_image_token=infer_num_image_tokens(model.internvl, int(data_cfg.get("image_size", 448))),
        num_segments=int(data_cfg.get("video_segments", 8)),
        image_size=int(data_cfg.get("image_size", 448)),
        max_length=int(data_cfg.get("max_length", 8192)),
    )
    training_cfg = config.get("training", {})
    batch_size_key = "batch_size" if split == "train" else "eval_batch_size"
    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get(batch_size_key, 1)),
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=int(training_cfg.get("num_workers", 2)),
        pin_memory=True,
        persistent_workers=int(training_cfg.get("num_workers", 2)) > 0,
        collate_fn=collator,
    )


def confusion_metrics(confusion: Tensor) -> dict[str, float]:
    confusion = confusion.float()
    total = confusion.sum().clamp_min(1)
    accuracy = confusion.diag().sum() / total
    precision = confusion.diag() / confusion.sum(dim=0).clamp_min(1)
    recall = confusion.diag() / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "accuracy": float(accuracy.item()),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype) -> dict[str, float]:
    evaluation_model = unwrap(model)
    evaluation_model.eval()
    confusion = torch.zeros((2, 2), device=device, dtype=torch.long)
    branch_correct = torch.zeros(2, device=device, dtype=torch.long)
    sample_count = torch.zeros(1, device=device, dtype=torch.long)
    router_sum = torch.zeros(4, device=device)
    loss_sum = torch.zeros(1, device=device)
    autocast_enabled = dtype != torch.float32
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            output = evaluation_model(**model_inputs(batch))
        predictions = output.logits.argmax(dim=-1)
        labels = batch["labels"]
        indices = labels * 2 + predictions
        confusion += torch.bincount(indices, minlength=4).reshape(2, 2)
        for index, logits in enumerate((output.logits, output.llm_logits)):
            branch_correct[index] += logits.argmax(dim=-1).eq(labels).sum()
        count = labels.numel()
        sample_count += count
        router_sum += output.router_weights.float().sum(dim=0)
        loss_sum += output.loss.detach().float() * count
    if dist.is_initialized():
        for tensor in (confusion, branch_correct, sample_count, router_sum, loss_sum):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    metrics = confusion_metrics(confusion)
    denominator = sample_count.clamp_min(1).float()
    metrics.update(
        {
            "loss": float((loss_sum / denominator).item()),
            "fusion_accuracy": float((branch_correct[0] / denominator).item()),
            "llm_accuracy": float((branch_correct[1] / denominator).item()),
            "router_llm": float((router_sum[0] / denominator).item()),
            "router_visual": float((router_sum[1] / denominator).item()),
            "router_text": float((router_sum[2] / denominator).item()),
            "router_audio": float((router_sum[3] / denominator).item()),
        }
    )
    return metrics


def save_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    checkpoint_name: str | None = None,
) -> Path:
    output_path = Path(output_dir) / (checkpoint_name or f"checkpoint-epoch-{epoch}")
    output_path.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap(model)
    raw_model.language_model.save_pretrained(output_path / "language_lora", safe_serialization=True)
    classifier_state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in raw_model.classifier.state_dict().items()
    }
    save_file(classifier_state, str(output_path / "classifier.safetensors"))
    with (output_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    state = {"epoch": epoch, "global_step": global_step}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, output_path / "trainer_state.pt")
    return output_path


def train(config: dict[str, Any], resume: str | Path | None = None) -> None:
    from transformers import get_cosine_schedule_with_warmup

    rank, local_rank, world_size, device = setup_distributed()
    training_cfg = config["training"]
    seed_everything(int(training_cfg.get("seed", 42)), rank)
    dtype = torch_dtype(config["model"].get("dtype", "bfloat16"))
    model, tokenizer = build_model_and_tokenizer(config, checkpoint=resume)
    model.to(device)
    train_loader = build_loader(config, "train", tokenizer, model, rank, world_size, shuffle=True)
    eval_split = str(training_cfg.get("evaluation_split", "val"))
    if eval_split not in {"val", "test"}:
        raise ValueError(f"evaluation_split must be 'val' or 'test', got {eval_split!r}")
    eval_loader = None
    if bool(training_cfg.get("evaluate", True)):
        eval_path_key = f"{eval_split}_path"
        if not config["data"].get(eval_path_key):
            raise ValueError(f"evaluate=true requires data.{eval_path_key}")
        eval_loader = build_loader(config, eval_split, tokenizer, model, rank, world_size, shuffle=False)

    classifier_parameters = list(model.classifier.parameters())
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    lora_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in classifier_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": float(training_cfg.get("lora_learning_rate", 2e-5))},
            {"params": classifier_parameters, "lr": float(training_cfg.get("head_learning_rate", 1e-4))},
        ],
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    epochs = int(training_cfg.get("epochs", 5))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = updates_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(training_cfg.get("warmup_ratio", 0.1))),
        num_training_steps=total_steps,
    )
    start_epoch = 1
    global_step = 0
    if resume:
        state_path = Path(resume) / "trainer_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state["global_step"])

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    if rank == 0:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(json.dumps({"trainable_parameters": trainable, "world_size": world_size, "steps": total_steps}))

    autocast_enabled = dtype != torch.float32
    log_steps = int(training_cfg.get("logging_steps", 10))
    best_metric_name = str(training_cfg.get("metric_for_best_model", "macro_f1"))
    greater_is_better = bool(training_cfg.get("greater_is_better", True))
    best_metric = -math.inf if greater_is_better else math.inf
    best_epoch = 0
    if resume:
        best_metadata_path = Path(training_cfg["output_dir"]) / "checkpoint-best" / "best_metric.json"
        if best_metadata_path.is_file():
            with best_metadata_path.open("r", encoding="utf-8") as handle:
                best_metadata = json.load(handle)
            if best_metadata.get("metric_name") == best_metric_name:
                best_metric = float(best_metadata["metric_value"])
                best_epoch = int(best_metadata["epoch"])
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        raw_model = unwrap(model)
        raw_model.internvl.vision_model.eval()
        if isinstance(getattr(raw_model.internvl, "mlp1", None), nn.Module):
            raw_model.internvl.mlp1.eval()
        running: dict[str, float] = {}
        running_batches = 0
        for micro_step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            should_step = (micro_step + 1) % accumulation == 0 or micro_step + 1 == len(train_loader)
            sync_context = nullcontext() if should_step or not isinstance(model, DistributedDataParallel) else model.no_sync()
            with sync_context:
                with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                    output = model(**model_inputs(batch))
                    loss = output.loss / accumulation
                loss.backward()
            for name, value in output.losses.items():
                running[name] = running.get(name, 0.0) + float(value.detach().float().item())
            labels = batch["labels"]
            for name, logits in (
                ("accuracy_fusion", output.logits),
                ("accuracy_llm", output.llm_logits),
            ):
                accuracy = logits.argmax(dim=-1).eq(labels).float().mean()
                running[name] = running.get(name, 0.0) + float(accuracy.item())
            for index, name in enumerate(("router_llm", "router_visual", "router_text", "router_audio")):
                mean_weight = output.router_weights[:, index].float().mean()
                running[name] = running.get(name, 0.0) + float(mean_weight.item())
            running_batches += 1
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if rank == 0 and global_step % log_steps == 0:
                payload = {
                    f"train/{key}": value / max(1, running_batches) for key, value in running.items()
                }
                payload.update({"epoch": epoch, "step": global_step})
                print(json.dumps(payload, ensure_ascii=False), flush=True)
                running.clear()
                running_batches = 0

        metrics = evaluate(model, eval_loader, device, dtype) if eval_loader is not None else {}
        is_best = False
        if metrics:
            if best_metric_name not in metrics:
                raise KeyError(
                    f"metric_for_best_model={best_metric_name!r} is unavailable; "
                    f"metrics={sorted(metrics)}"
                )
            candidate = float(metrics[best_metric_name])
            is_best = candidate > best_metric if greater_is_better else candidate < best_metric
            if is_best:
                best_metric = candidate
                best_epoch = epoch
        if rank == 0:
            if metrics:
                print(
                    json.dumps(
                        {f"{eval_split}/{key}": value for key, value in metrics.items()},
                        ensure_ascii=False,
                    )
                )
            checkpoint = save_checkpoint(
                model,
                training_cfg["output_dir"],
                config,
                epoch,
                global_step,
                optimizer,
                scheduler,
            )
            print(f"Saved {checkpoint}", flush=True)
            if is_best:
                best_checkpoint = save_checkpoint(
                    model,
                    training_cfg["output_dir"],
                    config,
                    epoch,
                    global_step,
                    optimizer,
                    scheduler,
                    checkpoint_name="checkpoint-best",
                )
                best_metadata = {
                    "selection_split": eval_split,
                    "metric_name": best_metric_name,
                    "metric_value": best_metric,
                    "epoch": epoch,
                    "global_step": global_step,
                    "source_checkpoint": str(checkpoint),
                }
                with (best_checkpoint / "best_metric.json").open("w", encoding="utf-8") as handle:
                    json.dump(best_metadata, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                print(
                    f"New best {eval_split}/{best_metric_name}={best_metric:.6f}; "
                    f"saved {best_checkpoint}",
                    flush=True,
                )
        if dist.is_initialized():
            dist.barrier()
    if rank == 0 and best_epoch:
        print(
            f"Best checkpoint: {Path(training_cfg['output_dir']) / 'checkpoint-best'} "
            f"({eval_split}/{best_metric_name}={best_metric:.6f}, epoch={best_epoch})",
            flush=True,
        )
    cleanup_distributed()


@torch.no_grad()
def predict(config: dict[str, Any], checkpoint: str | Path, output_path: str | Path) -> None:
    rank, _, world_size, device = setup_distributed()
    dtype = torch_dtype(config["model"].get("dtype", "bfloat16"))
    model, tokenizer = build_model_and_tokenizer(config, checkpoint=checkpoint)
    model.to(device).eval()
    split = str(config.get("inference", {}).get("split", "test"))
    loader = build_loader(config, split, tokenizer, model, rank, world_size, shuffle=False)
    local_records = []
    for batch in loader:
        ids = batch["ids"]
        batch = move_batch(batch, device)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            output = model(**model_inputs(batch))
        logits_sets = {
            "fusion": output.logits,
            "llm": output.llm_logits,
        }
        probabilities = {name: logits.float().softmax(dim=-1).cpu() for name, logits in logits_sets.items()}
        weights = output.router_weights.float().cpu()
        labels = batch["labels"].cpu()
        for index, sample_id in enumerate(ids):
            prediction = int(probabilities["fusion"][index].argmax().item())
            local_records.append(
                {
                    "id": sample_id,
                    "label": ID_TO_LABEL[int(labels[index].item())],
                    "prediction": ID_TO_LABEL[prediction],
                    "probabilities": {
                        name: {"real": float(value[index, 0]), "fake": float(value[index, 1])}
                        for name, value in probabilities.items()
                    },
                    "router_weights": {
                        "llm": float(weights[index, 0]),
                        "visual": float(weights[index, 1]),
                        "text": float(weights[index, 2]),
                        "audio": float(weights[index, 3]),
                    },
                }
            )
    if dist.is_initialized():
        gathered: list[list[dict[str, Any]] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_records)
        records = [record for group in gathered if group for record in group]
    else:
        records = local_records
    if rank == 0:
        unique = {record["id"]: record for record in records}
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for sample_id in sorted(unique):
                handle.write(json.dumps(unique[sample_id], ensure_ascii=False) + "\n")
        print(json.dumps({"predictions": len(unique), "output": str(output_path)}, ensure_ascii=False))
    cleanup_distributed()
