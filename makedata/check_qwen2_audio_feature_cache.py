from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import torch
    from safetensors.torch import load_file

    parser = argparse.ArgumentParser(description="Validate a Qwen2-Audio feature manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-hidden-size", type=int, default=0)
    args = parser.parse_args()
    counts = {"records": 0, "valid": 0, "invalid": 0, "fallback": 0, "truncated": 0}
    hidden_sizes: dict[str, int] = {}
    errors = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            counts["records"] += 1
            record = json.loads(line)
            counts["fallback"] += int(bool(record.get("audio_fallback")))
            counts["truncated"] += int(bool(record.get("audio_truncated")))
            try:
                feature_path = Path(record["feature_path"])
                if not feature_path.is_absolute():
                    feature_path = args.manifest.parent / feature_path
                tensors = load_file(str(feature_path))
                state = tensors["audio_last_token"]
                if state.ndim != 1:
                    raise ValueError(f"expected [D], got {tuple(state.shape)}")
                if args.expected_hidden_size and state.shape[0] != args.expected_hidden_size:
                    raise ValueError(f"expected D={args.expected_hidden_size}, got {state.shape[0]}")
                if not torch.isfinite(state.float()).all():
                    raise ValueError("contains NaN or Inf")
                hidden_sizes[str(state.shape[0])] = hidden_sizes.get(str(state.shape[0]), 0) + 1
                counts["valid"] += 1
            except Exception as error:
                counts["invalid"] += 1
                if len(errors) < 20:
                    errors.append({"line": line_number, "id": record.get("video_id"), "error": str(error)})
    print(json.dumps({**counts, "hidden_sizes": hidden_sizes, "errors": errors}, ensure_ascii=False, indent=2))
    if counts["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
