from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = (
    "token_states",
    "attention_mask",
    "special_tokens_mask",
    "cls_token",
    "content_mean",
    "global_feature",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a BERT text-feature cache.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--error-output", type=Path)
    return parser.parse_args()


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


def validate(manifest: Path, record: dict[str, Any]) -> dict[str, Any] | None:
    import torch
    from safetensors.torch import load_file

    sample_id = str(record.get("video_id") or "<missing>")
    path = resolve_feature_path(manifest, record)
    if not path.is_file():
        return {"video_id": sample_id, "error": f"missing feature: {path}"}
    try:
        tensors = load_file(str(path), device="cpu")
        missing = [name for name in REQUIRED if name not in tensors]
        if missing:
            raise ValueError(f"missing tensors: {', '.join(missing)}")
        states = tensors["token_states"]
        cls = tensors["cls_token"]
        if states.ndim != 2 or cls.shape != (states.shape[1],):
            raise ValueError(f"invalid token/CLS shapes: {states.shape}, {cls.shape}")
        if tensors["attention_mask"].shape != (states.shape[0],):
            raise ValueError("attention_mask length does not match token_states")
        if tensors["global_feature"].shape != (states.shape[1] * 2,):
            raise ValueError("global_feature must concatenate CLS and content mean")
        for name, tensor in tensors.items():
            if tensor.numel() == 0 or not torch.isfinite(tensor.float()).all().item():
                raise ValueError(f"invalid values in {name}")
    except Exception as error:
        return {
            "video_id": sample_id,
            "feature_path": str(path),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return None


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    with args.manifest.open("r", encoding="utf-8-sig") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    selected = records[: args.max_samples] if args.max_samples > 0 else records
    errors = [error for record in selected if (error := validate(args.manifest, record))]
    summary = {
        "records": len(records),
        "checked": len(selected),
        "valid": len(selected) - len(errors),
        "invalid": len(errors),
        "truncated": sum(bool(record.get("truncated")) for record in records),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps({"error_preview": errors[:10]}, ensure_ascii=False, indent=2))
        if args.error_output:
            args.error_output.parent.mkdir(parents=True, exist_ok=True)
            with args.error_output.open("w", encoding="utf-8") as handle:
                for error in errors:
                    handle.write(json.dumps(error, ensure_ascii=False) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
