from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def main() -> None:
    import torch
    from safetensors.torch import load_file

    parser = argparse.ArgumentParser(description="Validate a Qwen3 text-feature manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    with args.manifest.open("r", encoding="utf-8-sig") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    selected = records[: args.max_samples] if args.max_samples > 0 else records
    counts = {
        "records": len(records),
        "checked": len(selected),
        "valid": 0,
        "invalid": 0,
        "truncated": sum(bool(record.get("truncated")) for record in records),
    }
    hidden_sizes: dict[str, int] = {}
    errors = []
    for line_number, record in enumerate(selected, start=1):
        try:
            feature_path = resolve_feature_path(args.manifest, record)
            if not feature_path.is_file():
                raise FileNotFoundError(feature_path)
            tensors = load_file(str(feature_path), device="cpu")
            state = tensors["text_last_token"]
            if state.ndim != 1:
                raise ValueError(f"expected [D], got {tuple(state.shape)}")
            if args.expected_hidden_size and state.shape[0] != args.expected_hidden_size:
                raise ValueError(
                    f"expected D={args.expected_hidden_size}, got {state.shape[0]}"
                )
            if not torch.isfinite(state.float()).all().item():
                raise ValueError("contains NaN or Inf")
            hidden_sizes[str(state.shape[0])] = hidden_sizes.get(str(state.shape[0]), 0) + 1
            counts["valid"] += 1
        except Exception as error:
            counts["invalid"] += 1
            if len(errors) < 20:
                errors.append(
                    {
                        "line": line_number,
                        "id": record.get("video_id"),
                        "error": str(error),
                    }
                )
    print(json.dumps({**counts, "hidden_sizes": hidden_sizes, "errors": errors}, ensure_ascii=False, indent=2))
    if counts["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
