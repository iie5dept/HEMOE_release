from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trimodal_moe.engine import load_config, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tri-modal MoE classifier inference.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict(load_config(args.config), args.checkpoint, args.output)


if __name__ == "__main__":
    main()
