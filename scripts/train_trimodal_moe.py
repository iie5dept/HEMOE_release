from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from trimodal_moe.engine import load_config, train
except ModuleNotFoundError as exc:
    if exc.name == "safetensors":
        raise SystemExit(
            "Missing dependency 'safetensors' in "
            f"{sys.executable}. Launch with the moedet interpreter, for example: "
            "/data2/573ops_ser/envs/moedet/bin/python -m torch.distributed.run ..."
        ) from exc
    raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cached-feature InternVL tri-modal MoE classifier.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
