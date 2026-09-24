from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.visualize_sva_rollout import (
    build_classifier,
    capture_attention_rollout,
    load_prediction_index,
    read_jsonl,
    record_id,
    resize_rollout,
    resolve_feature_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export SVA original/overlay image pairs for correctly predicted fake test samples, "
            "ranked by descending SigLIP similarity."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--evidence-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overlay-alpha", type=float, default=0.68)
    parser.add_argument("--export-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Maximum number of ranked cases to export; 0 exports every eligible sample.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay-alpha must be in [0, 1]")
    if args.export_size <= 0:
        parser.error("--export-size must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.max_cases < 0:
        parser.error("--max-cases must be non-negative")
    return args


def load_processor(model_path: Path) -> Any:
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )


def processor_aligned_image(image_path: Path, processor: Any) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"][0].float()
    mean = torch.tensor(getattr(processor, "image_mean", (0.485, 0.456, 0.406))).view(3, 1, 1)
    std = torch.tensor(getattr(processor, "image_std", (0.229, 0.224, 0.225))).view(3, 1, 1)
    restored = (pixel_values * std + mean).clamp(0, 1)
    return restored.permute(1, 2, 0).cpu().numpy()


def save_image(path: Path, image: np.ndarray, size: int, dpi: int) -> None:
    rendered = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8))
    rendered = rendered.resize((size, size), Image.Resampling.LANCZOS)
    rendered.save(path, dpi=(dpi, dpi))


def save_pair(
    case_dir: Path,
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
    size: int,
    dpi: int,
) -> None:
    colorized = matplotlib.colormaps["magma"](heatmap)[..., :3]
    overlay = np.clip((1.0 - alpha) * image + alpha * colorized, 0.0, 1.0)
    case_dir.mkdir(parents=True, exist_ok=True)
    save_image(case_dir / "original.png", image, size, dpi)
    save_image(case_dir / "overlay.png", overlay, size, dpi)


def select_records(
    manifest_records: list[dict[str, Any]],
    test_ids: set[str],
    prediction_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in manifest_records:
        sample_id = record_id(record)
        prediction = prediction_index.get(sample_id)
        if sample_id not in test_ids or prediction is None:
            continue
        if not (
            prediction.get("correct", False)
            and prediction.get("ground_truth") == "fake"
            and prediction.get("prediction") == "fake"
        ):
            continue
        try:
            similarity = float(record.get("similarity"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(similarity):
            continue
        selected.append(record)
    selected.sort(key=lambda record: (-float(record["similarity"]), record_id(record)))
    return selected


def save_index(
    output_dir: Path,
    cases: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    with (output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "similarity_rank",
                "sample_id",
                "similarity",
                "retrieval_text",
                "frame_index",
                "timestamp_sec",
                "original_path",
                "overlay_path",
            ),
        )
        writer.writeheader()
        writer.writerows(cases)
    metadata = {
        "selection_rule": (
            "FakeTT test samples with ground_truth=fake and prediction=fake, ranked by "
            "descending SigLIP similarity."
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "visual_manifest": str(args.visual_manifest.resolve()),
        "evidence_archive": str(args.evidence_archive.resolve()),
        "overlay_alpha": args.overlay_alpha,
        "exported": len(cases),
        "failed": len(errors),
        "cases": cases,
        "errors": errors,
    }
    with (output_dir / "cases.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    import yaml
    from safetensors.torch import load_file

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    test_records = read_jsonl(Path(config["data"]["test_path"]))
    test_ids = {record_id(record) for record in test_records}
    prediction_index = load_prediction_index(args.evidence_archive)
    manifest_records = read_jsonl(args.visual_manifest)
    ranked_records = select_records(manifest_records, test_ids, prediction_index)
    if args.max_cases:
        ranked_records = ranked_records[: args.max_cases]
    if not ranked_records:
        raise ValueError("No correctly predicted fake test samples with valid SigLIP similarity")

    classifier = build_classifier(config, args.checkpoint)
    device = torch.device(args.device)
    processor_cache: dict[Path, Any] = {}
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Eligible correctly predicted fake samples: {len(ranked_records)}", flush=True)

    for similarity_rank, record in enumerate(ranked_records, start=1):
        sample_id = record_id(record)
        try:
            feature_path = resolve_feature_path(args.visual_manifest, record)
            tensors = load_file(str(feature_path), device="cpu")
            if "patch_tokens" not in tensors:
                raise KeyError(f"DINO feature file has no patch_tokens: {feature_path}")
            _, _, patch_rollout = capture_attention_rollout(
                classifier.visual_branch,
                tensors["patch_tokens"],
                device,
            )
            image_path = Path(str(record.get("key_frame_path") or ""))
            model_path = Path(str(record.get("model_path") or ""))
            if not image_path.is_file():
                raise FileNotFoundError(f"Saved key frame not found: {image_path}")
            if not model_path.is_dir():
                raise FileNotFoundError(f"DINO model path not found: {model_path}")
            if model_path not in processor_cache:
                processor_cache[model_path] = load_processor(model_path)
            processor = processor_cache[model_path]
            image = processor_aligned_image(image_path, processor)
            _, heatmap = resize_rollout(patch_rollout, image.shape[:2])

            case_id = len(cases) + 1
            case_dir = args.output_dir / str(case_id)
            save_pair(
                case_dir,
                image,
                heatmap,
                args.overlay_alpha,
                args.export_size,
                args.dpi,
            )
            cases.append(
                {
                    "case_id": case_id,
                    "similarity_rank": similarity_rank,
                    "sample_id": sample_id,
                    "similarity": float(record["similarity"]),
                    "retrieval_text": record.get("retrieval_text"),
                    "frame_index": record.get("frame_index"),
                    "timestamp_sec": record.get("timestamp_sec"),
                    "original_path": f"{case_id}/original.png",
                    "overlay_path": f"{case_id}/overlay.png",
                }
            )
            if case_id == 1 or case_id % 10 == 0 or similarity_rank == len(ranked_records):
                print(
                    f"Exported {case_id}/{len(ranked_records)} "
                    f"(similarity rank {similarity_rank}, sample {sample_id})",
                    flush=True,
                )
        except Exception as error:
            errors.append(
                {
                    "similarity_rank": similarity_rank,
                    "sample_id": sample_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(
                f"Failed similarity rank {similarity_rank}, sample {sample_id}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    save_index(args.output_dir, cases, errors, args)
    if not cases:
        raise RuntimeError("Every eligible sample failed during SVA export")
    print(
        json.dumps(
            {
                "eligible": len(ranked_records),
                "exported": len(cases),
                "failed": len(errors),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
