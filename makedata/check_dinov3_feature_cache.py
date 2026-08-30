from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_TENSORS = ("cls_token", "pooler_output", "patch_mean", "global_feature")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a DINOv3 feature-cache manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Validate all samples by default; use a positive value for a quick check.",
    )
    parser.add_argument("--require-patch-tokens", action="store_true")
    parser.add_argument("--error-output", type=Path)
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    return args


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"Feature manifest is empty: {path}")
    return records


def resolve_feature_path(manifest: Path, record: dict[str, Any]) -> Path:
    absolute = Path(str(record.get("feature_path") or ""))
    if absolute.is_file():
        return absolute
    relative = record.get("feature_relpath")
    if relative:
        candidate = manifest.parent / str(relative)
        if candidate.is_file():
            return candidate
    return absolute


def validate_record(
    manifest: Path,
    record: dict[str, Any],
    require_patch_tokens: bool,
) -> dict[str, Any] | None:
    import torch
    from safetensors.torch import load_file

    sample_id = str(record.get("video_id") or "<missing>")
    feature_path = resolve_feature_path(manifest, record)
    if not feature_path.is_file():
        return {"video_id": sample_id, "error": f"missing feature file: {feature_path}"}
    try:
        tensors = load_file(str(feature_path), device="cpu")
        required = list(REQUIRED_TENSORS)
        if require_patch_tokens:
            required.append("patch_tokens")
        missing = [name for name in required if name not in tensors]
        if missing:
            raise ValueError(f"missing tensors: {', '.join(missing)}")

        cls = tensors["cls_token"]
        patch_mean = tensors["patch_mean"]
        global_feature = tensors["global_feature"]
        if cls.ndim != 1 or patch_mean.shape != cls.shape:
            raise ValueError(
                f"invalid cls/patch_mean shapes: {tuple(cls.shape)}, {tuple(patch_mean.shape)}"
            )
        if global_feature.shape != (cls.shape[0] * 2,):
            raise ValueError(f"invalid global_feature shape: {tuple(global_feature.shape)}")
        if "patch_tokens" in tensors:
            patches = tensors["patch_tokens"]
            if patches.ndim != 2 or patches.shape[1] != cls.shape[0]:
                raise ValueError(f"invalid patch_tokens shape: {tuple(patches.shape)}")
            expected_count = int(record.get("patch_count", patches.shape[0]))
            if patches.shape[0] != expected_count:
                raise ValueError(
                    f"patch count mismatch: tensor={patches.shape[0]}, metadata={expected_count}"
                )
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor.float()).all().item():
                raise ValueError(f"{name} contains NaN or Inf")
            if math.prod(tensor.shape) == 0:
                raise ValueError(f"{name} is empty")
    except Exception as error:
        return {
            "video_id": sample_id,
            "feature_path": str(feature_path),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return None


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    selected = records[: args.max_samples] if args.max_samples > 0 else records
    errors = []
    for record in selected:
        error = validate_record(args.manifest, record, args.require_patch_tokens)
        if error is not None:
            errors.append(error)

    summary = {
        "manifest": str(args.manifest.resolve()),
        "records": len(records),
        "checked": len(selected),
        "valid": len(selected) - len(errors),
        "invalid": len(errors),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        preview = errors[:10]
        print(json.dumps({"error_preview": preview}, ensure_ascii=False, indent=2))
        if args.error_output is not None:
            args.error_output.parent.mkdir(parents=True, exist_ok=True)
            with args.error_output.open("w", encoding="utf-8") as handle:
                for error in errors:
                    handle.write(json.dumps(error, ensure_ascii=False) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
