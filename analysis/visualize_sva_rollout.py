from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trimodal_moe.modules import FourModalMoEClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize summary-token attention rollout from the trained SVA branch."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument(
        "--evidence-archive",
        type=Path,
        help=(
            "NPZ from export_evidence_features.py. When available, automatic selection keeps "
            "only correctly predicted test samples before ranking by SigLIP similarity."
        ),
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        help=(
            "Explicit sample ID to export. Repeat this option to export several cases in one "
            "run; each case is written to its own numbered subdirectory."
        ),
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=1,
        help=(
            "Number of automatically ranked cases to export when --sample-id is omitted. "
            "Ranking starts at --selection-rank."
        ),
    )
    parser.add_argument(
        "--selection-rank",
        type=int,
        default=1,
        help="One-based SigLIP rank used for automatic selection after correctness filtering.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overlay-alpha", type=float, default=0.62)
    parser.add_argument("--export-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay-alpha must be in [0, 1]")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.export_size <= 0:
        parser.error("--export-size must be positive")
    if args.selection_rank <= 0:
        parser.error("--selection-rank must be positive")
    if args.num_cases <= 0:
        parser.error("--num-cases must be positive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("video_id", record.get("id", ""))).strip()


def resolve_feature_path(manifest_path: Path, record: dict[str, Any]) -> Path:
    path = Path(str(record.get("feature_path") or ""))
    if path.is_file():
        return path
    relative = record.get("feature_relpath")
    if relative:
        candidate = manifest_path.parent / str(relative)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Feature file not found for {record_id(record)}: {path}")


def select_record(
    manifest_records: list[dict[str, Any]],
    test_ids: set[str],
    sample_id: str | None,
    prediction_index: dict[str, dict[str, Any]] | None,
    selection_rank: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = [record for record in manifest_records if record_id(record) in test_ids]
    if not eligible:
        raise ValueError("The visual manifest has no records belonging to the configured test split")
    if sample_id is not None:
        for record in eligible:
            if record_id(record) == sample_id:
                return record, eligible
        raise KeyError(f"Sample {sample_id!r} is not available in both the test split and manifest")
    if prediction_index is not None:
        eligible = [
            record
            for record in eligible
            if prediction_index.get(record_id(record), {}).get("correct", False)
        ]
        if not eligible:
            raise ValueError("No correctly predicted test samples overlap the visual manifest")
    ranked = sorted(
        eligible,
        key=lambda record: float(record.get("similarity", float("-inf"))),
        reverse=True,
    )
    if selection_rank > len(ranked):
        raise ValueError(
            f"--selection-rank={selection_rank} exceeds {len(ranked)} eligible samples"
        )
    return ranked[selection_rank - 1], ranked


def load_prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    names = {0: "real", 1: "fake"}
    with np.load(path, allow_pickle=False) as archive:
        required = ("sample_ids", "labels", "predictions", "router_weights")
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(f"Evidence archive is missing: {', '.join(missing)}")
        sample_ids = archive["sample_ids"].astype(str)
        labels = archive["labels"].astype(np.int64)
        predictions = archive["predictions"].astype(np.int64)
        router_weights = archive["router_weights"].astype(np.float32)
    if not (len(sample_ids) == len(labels) == len(predictions) == len(router_weights)):
        raise ValueError("Evidence archive prediction arrays have inconsistent lengths")
    index: dict[str, dict[str, Any]] = {}
    for sample_id, label, prediction, weights in zip(
        sample_ids, labels, predictions, router_weights
    ):
        if sample_id in index:
            raise ValueError(f"Duplicate sample ID in evidence archive: {sample_id}")
        label_id = int(label)
        prediction_id = int(prediction)
        index[sample_id] = {
            "ground_truth": names.get(label_id, str(label_id)),
            "prediction": names.get(prediction_id, str(prediction_id)),
            "correct": label_id == prediction_id,
            "router_weights_MVTA": weights.tolist(),
        }
    return index


def build_classifier(config: dict[str, Any], checkpoint: Path) -> FourModalMoEClassifier:
    from safetensors.torch import load_file

    state_path = checkpoint / "classifier.safetensors"
    if not state_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {state_path}")
    state = load_file(str(state_path), device="cpu")
    llm_projection = state.get("llm_branch.projector.1.weight")
    if llm_projection is None or llm_projection.ndim != 2:
        raise KeyError("Cannot infer llm_dim from llm_branch.projector.1.weight")
    fusion = config["fusion"]
    classifier = FourModalMoEClassifier(
        llm_dim=int(llm_projection.shape[1]),
        visual_dim=int(fusion.get("visual_dim", 1024)),
        text_dim=int(fusion.get("text_dim", 4096)),
        audio_dim=int(fusion.get("audio_dim", 4096)),
        hidden_dim=int(fusion.get("hidden_dim", 256)),
        transformer_layers=int(fusion.get("transformer_layers", 2)),
        transformer_heads=int(fusion.get("transformer_heads", 8)),
        transformer_ff_dim=int(fusion.get("transformer_ff_dim", 1024)),
        expert_dim=int(fusion.get("expert_dim", 512)),
        router_dim=int(fusion.get("router_dim", 128)),
        dropout=float(fusion.get("dropout", 0.1)),
        modality_loss_weight=float(fusion.get("modality_loss_weight", 0.3)),
        llm_loss_weight=float(fusion.get("llm_loss_weight", 1.0)),
    )
    missing, unexpected = classifier.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Classifier checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return classifier


def capture_attention_rollout(
    visual_branch: nn.Module,
    patch_tokens: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if patch_tokens.ndim != 2:
        raise ValueError(
            f"SVA rollout requires the 90.6 single-key-frame cache [P, D], got {tuple(patch_tokens.shape)}"
        )
    attention_weights: list[Tensor] = []
    handles = []

    def force_weights(
        _module: nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return inputs, kwargs

    def save_weights(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        weights = output[1]
        if weights is None:
            raise RuntimeError("SVA self-attention did not return attention weights")
        attention_weights.append(weights.detach().float().cpu().contiguous())

    for layer in visual_branch.encoder.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(force_weights, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(save_weights))

    tokens = patch_tokens.float().unsqueeze(0).to(device)
    mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
    visual_branch = visual_branch.to(device).eval()
    try:
        with torch.inference_mode():
            visual_hidden = visual_branch(tokens, mask).detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    if len(attention_weights) != len(visual_branch.encoder.layers):
        raise RuntimeError(
            f"Captured {len(attention_weights)} attention tensors from "
            f"{len(visual_branch.encoder.layers)} SVA layers"
        )
    stacked = torch.cat(attention_weights, dim=0)
    sequence_length = stacked.shape[-1]
    rollout = torch.eye(sequence_length, dtype=torch.float32)
    for layer_attention in stacked:
        fused = layer_attention.mean(dim=0)
        fused = fused + torch.eye(sequence_length, dtype=fused.dtype)
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = fused @ rollout
    summary_to_patches = rollout[0, 1:]
    return visual_hidden, stacked, summary_to_patches


def processor_aligned_image(image_path: Path, model_path: Path) -> np.ndarray:
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"][0].float()
    mean = torch.tensor(getattr(processor, "image_mean", (0.485, 0.456, 0.406))).view(3, 1, 1)
    std = torch.tensor(getattr(processor, "image_std", (0.229, 0.224, 0.225))).view(3, 1, 1)
    restored = (pixel_values * std + mean).clamp(0, 1)
    return restored.permute(1, 2, 0).cpu().numpy()


def resize_rollout(values: Tensor, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    patch_count = int(values.numel())
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Cannot reshape {patch_count} patch scores into a square grid")
    grid = values.reshape(1, 1, grid_size, grid_size)
    height, width = image_shape
    resized = F.interpolate(grid, size=(height, width), mode="bicubic", align_corners=False)[0, 0]
    resized = resized.clamp_min(0)
    resized = (resized - resized.min()) / (resized.max() - resized.min()).clamp_min(1e-12)
    return values.reshape(grid_size, grid_size).numpy(), resized.numpy()


def save_visualizations(
    output_dir: Path,
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
    export_size: int,
    dpi: int,
) -> None:
    colorized = matplotlib.colormaps["magma"](heatmap)[..., :3]
    overlay = np.clip((1.0 - alpha) * image + alpha * colorized, 0.0, 1.0)
    for filename, panel in (
        ("selected_frame.png", image),
        ("sva_rollout_heatmap.png", colorized),
        ("sva_rollout_overlay.png", overlay),
    ):
        rendered = Image.fromarray((panel * 255).round().astype(np.uint8))
        rendered = rendered.resize((export_size, export_size), Image.Resampling.LANCZOS)
        rendered.save(output_dir / filename, dpi=(dpi, dpi))


def export_case(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    classifier: FourModalMoEClassifier,
    record: dict[str, Any],
    ranked_records: list[dict[str, Any]],
    prediction_index: dict[str, dict[str, Any]] | None,
    output_dir: Path,
    automatic_selection: bool,
    selection_rank: int | None,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    sample_id = record_id(record)
    feature_path = resolve_feature_path(args.visual_manifest, record)
    tensors = load_file(str(feature_path), device="cpu")
    if "patch_tokens" not in tensors:
        raise KeyError(f"DINO feature file has no patch_tokens: {feature_path}")

    visual_hidden, layer_attention, patch_rollout = capture_attention_rollout(
        classifier.visual_branch,
        tensors["patch_tokens"],
        torch.device(args.device),
    )
    image_path = Path(str(record.get("key_frame_path") or ""))
    model_path = Path(str(record.get("model_path") or ""))
    if not image_path.is_file():
        raise FileNotFoundError(f"Saved key frame not found: {image_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"DINO model path from manifest not found: {model_path}")
    image = processor_aligned_image(image_path, model_path)
    patch_grid, heatmap = resize_rollout(patch_rollout, image.shape[:2])

    output_dir.mkdir(parents=True, exist_ok=True)
    save_visualizations(
        output_dir,
        image,
        heatmap,
        args.overlay_alpha,
        args.export_size,
        args.dpi,
    )
    np.savez_compressed(
        output_dir / "sva_attention_rollout.npz",
        layer_attention=layer_attention.numpy(),
        summary_to_patch=patch_rollout.numpy(),
        patch_grid=patch_grid,
        upsampled_heatmap=heatmap,
        visual_hidden=visual_hidden.numpy(),
    )
    metadata = {
        "sample_id": sample_id,
        "automatic_selection": automatic_selection,
        "selection_rank": selection_rank,
        "selection_rule": (
            "SigLIP similarity rank among correctly predicted FakeTT test samples"
            if automatic_selection and prediction_index is not None
            else "SigLIP similarity rank among FakeTT test samples"
            if automatic_selection
            else "explicit sample_id"
        ),
        "similarity": record.get("similarity"),
        "frame_index": record.get("frame_index"),
        "timestamp_sec": record.get("timestamp_sec"),
        "retrieval_text": record.get("retrieval_text"),
        "key_frame_path": str(image_path.resolve()),
        "feature_path": str(feature_path.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "attention_shape": list(layer_attention.shape),
        "patch_grid_shape": list(patch_grid.shape),
        "visual_hidden_shape": list(visual_hidden.shape),
        "candidate_preview": [
            {
                "rank": rank,
                "sample_id": record_id(candidate),
                "similarity": candidate.get("similarity"),
                "retrieval_text": candidate.get("retrieval_text"),
            }
            for rank, candidate in enumerate(ranked_records[:10], start=1)
        ],
    }
    if prediction_index is not None and sample_id in prediction_index:
        metadata.update(prediction_index[sample_id])
    with (output_dir / "sva_attention_rollout.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, ensure_ascii=False))
    print(f"Saved SVA rollout visualization to {output_dir.resolve()}")
    return metadata


def main() -> None:
    args = parse_args()
    import yaml

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evidence_archive = args.evidence_archive
    if evidence_archive is None:
        candidate = args.checkpoint.parent / "evidence_cka" / "fakett_test_all_evidence.npz"
        if candidate.is_file():
            evidence_archive = candidate
    prediction_index = (
        load_prediction_index(evidence_archive) if evidence_archive is not None else None
    )
    test_path = Path(config["data"]["test_path"])
    test_records = read_jsonl(test_path)
    test_index = {record_id(record): record for record in test_records}
    manifest_records = read_jsonl(args.visual_manifest)
    classifier = build_classifier(config, args.checkpoint)
    selected: list[tuple[dict[str, Any], list[dict[str, Any]], int | None]] = []
    if args.sample_id:
        if len(set(args.sample_id)) != len(args.sample_id):
            raise ValueError("Duplicate --sample-id values are not allowed")
        for sample_id in args.sample_id:
            record, ranked_records = select_record(
                manifest_records,
                set(test_index),
                sample_id,
                prediction_index,
                args.selection_rank,
            )
            selected.append((record, ranked_records, None))
    else:
        for rank in range(args.selection_rank, args.selection_rank + args.num_cases):
            record, ranked_records = select_record(
                manifest_records,
                set(test_index),
                None,
                prediction_index,
                rank,
            )
            selected.append((record, ranked_records, rank))

    multiple_cases = len(selected) > 1
    exported: list[dict[str, Any]] = []
    for case_index, (record, ranked_records, rank) in enumerate(selected, start=1):
        sample_id = record_id(record)
        output_dir = (
            args.output_dir / f"case_{case_index:02d}_{sample_id}"
            if multiple_cases
            else args.output_dir
        )
        exported.append(
            export_case(
                args=args,
                config=config,
                classifier=classifier,
                record=record,
                ranked_records=ranked_records,
                prediction_index=prediction_index,
                output_dir=output_dir,
                automatic_selection=not args.sample_id,
                selection_rank=rank,
            )
        )

    if multiple_cases:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "cases.json").open("w", encoding="utf-8") as handle:
            json.dump(exported, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Saved {len(exported)} cases and index to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
