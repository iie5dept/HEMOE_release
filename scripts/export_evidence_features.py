from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BRANCH_FEATURE_NAMES = ("H_m", "H_v", "H_t", "H_a")
ROUTED_FEATURE_NAMES = ("Z_m", "Z_v", "Z_t", "Z_a")
FEATURE_NAMES = BRANCH_FEATURE_NAMES + ROUTED_FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export pre-expert modality representations for Linear CKA analysis."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--visual-manifest",
        type=Path,
        help="Override data.visual_manifest, for example with the original 90.6 key-frame cache.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of randomly sampled records; 0 exports the complete split in dataset order.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_samples < 0 or args.num_samples == 1:
        parser.error("--num-samples must be 0 (all) or greater than one for CKA")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end with .npz")
    return args


def _capture_hook(
    capture: dict[str, Tensor],
    name: str,
    selector: Callable[[Any], Tensor],
) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if name in capture:
            raise RuntimeError(f"Feature hook {name} ran more than once in one forward pass")
        value = selector(output)
        if value.ndim != 2:
            raise RuntimeError(f"Expected {name} to be [B, D], got {tuple(value.shape)}")
        capture[name] = value.detach().float().cpu().contiguous()

    return hook


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def main() -> None:
    args = parse_args()
    from trimodal_moe.engine import (
        build_loader,
        build_model_and_tokenizer,
        load_config,
        model_inputs,
        move_batch,
        torch_dtype,
    )

    config = load_config(args.config)
    if args.visual_manifest is not None:
        config["data"]["visual_manifest"] = str(args.visual_manifest.resolve())
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dtype = torch_dtype(config["model"].get("dtype", "bfloat16"))
    model, tokenizer = build_model_and_tokenizer(config, checkpoint=args.checkpoint)
    model.to(device).eval()

    base_loader = build_loader(
        config=config,
        split=args.split,
        tokenizer=tokenizer,
        model=model,
        rank=0,
        world_size=1,
        shuffle=False,
    )
    dataset = base_loader.dataset
    if args.num_samples > len(dataset):
        raise ValueError(
            f"Requested {args.num_samples} samples, but {args.split} contains only {len(dataset)}"
        )
    if args.num_samples == 0:
        selected_indices = list(range(len(dataset)))
        selection = "all"
    else:
        selected_indices = random.Random(args.seed).sample(range(len(dataset)), args.num_samples)
        selection = "random"
    selected_count = len(selected_indices)
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=base_loader.collate_fn,
    )

    capture: dict[str, Tensor] = {}
    classifier = model.classifier
    handles = [
        classifier.llm_branch.register_forward_hook(
            _capture_hook(capture, "H_m", lambda output: output[0])
        ),
        classifier.visual_branch.register_forward_hook(
            _capture_hook(capture, "H_v", lambda output: output)
        ),
        classifier.text_branch.register_forward_hook(
            _capture_hook(capture, "H_t", lambda output: output)
        ),
        classifier.audio_branch.register_forward_hook(
            _capture_hook(capture, "H_a", lambda output: output)
        ),
        classifier.fusion_norm.register_forward_hook(
            _capture_hook(capture, "Z_fused", lambda output: output)
        ),
    ]

    features: dict[str, list[Tensor]] = {name: [] for name in FEATURE_NAMES}
    sample_ids: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    router_weights: list[Tensor] = []
    autocast_context = (
        lambda: torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)
        if device.type == "cuda"
        else nullcontext()
    )

    try:
        print(
            f"Exporting {selected_count} samples from {args.split} "
            f"with batch_size={args.batch_size}",
            flush=True,
        )
        with torch.inference_mode():
            for step, batch in enumerate(loader, start=1):
                ids = [str(value) for value in batch["ids"]]
                batch = move_batch(batch, device)
                capture.clear()
                with autocast_context():
                    output = model(**model_inputs(batch))
                missing = [
                    name for name in (*BRANCH_FEATURE_NAMES, "Z_fused") if name not in capture
                ]
                if missing:
                    raise RuntimeError(f"Feature hooks did not capture: {', '.join(missing)}")
                fused_hidden = capture["Z_fused"]
                if fused_hidden.shape[1] % len(ROUTED_FEATURE_NAMES) != 0:
                    raise RuntimeError(
                        f"Cannot split final fusion feature {tuple(fused_hidden.shape)} into "
                        f"{len(ROUTED_FEATURE_NAMES)} routed contribution blocks"
                    )
                routed_blocks = fused_hidden.chunk(len(ROUTED_FEATURE_NAMES), dim=-1)
                for name, block in zip(ROUTED_FEATURE_NAMES, routed_blocks):
                    capture[name] = block.contiguous()
                batch_size = len(ids)
                for name in FEATURE_NAMES:
                    if capture[name].shape[0] != batch_size:
                        raise RuntimeError(
                            f"{name} batch is {capture[name].shape[0]}, expected {batch_size}"
                        )
                    features[name].append(capture[name])
                sample_ids.extend(ids)
                labels.extend(batch["labels"].detach().cpu().tolist())
                predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
                router_weights.append(output.router_weights.detach().float().cpu().contiguous())
                if step == 1 or step % 25 == 0 or step == len(loader):
                    print(f"Exported {min(step * args.batch_size, selected_count)}/{selected_count}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    arrays = {name: torch.cat(values).numpy() for name, values in features.items()}
    expected_shape = (selected_count, int(config["fusion"].get("hidden_dim", 256)))
    for name, value in arrays.items():
        if value.shape != expected_shape:
            raise RuntimeError(f"Expected {name} shape {expected_shape}, got {value.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **arrays,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        dataset_indices=np.asarray(selected_indices, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.int64),
        router_weights=torch.cat(router_weights).numpy(),
    )
    metadata = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "visual_manifest": str(Path(config["data"]["visual_manifest"]).resolve()),
        "split": args.split,
        "selection": selection,
        "seed": args.seed,
        "num_samples": selected_count,
        "sample_ids": sample_ids,
        "dataset_indices": selected_indices,
        "labels": labels,
        "predictions": predictions,
        "feature_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "feature_semantics": {
            "H_m": "InternVL multimodal branch after its trainable projector, before MED",
            "H_v": "DINOv3 visual branch after Transformer aggregation, before MED",
            "H_t": "Qwen3 text branch after its trainable projector, before MED",
            "H_a": "Qwen2-Audio branch after its trainable projector, before MED",
            "Z_m": "Multimodal contribution block after expert weighting, concatenation, and fusion LayerNorm",
            "Z_v": "Visual contribution block after expert weighting, concatenation, and fusion LayerNorm",
            "Z_t": "Textual contribution block after expert weighting, concatenation, and fusion LayerNorm",
            "Z_a": "Acoustic contribution block after expert weighting, concatenation, and fusion LayerNorm",
        },
    }
    metadata_path = _metadata_path(args.output)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "features": str(args.output.resolve()),
                "metadata": str(metadata_path.resolve()),
                "num_samples": selected_count,
                "sample_id_preview": sample_ids[:10],
                "feature_shapes": metadata["feature_shapes"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
