from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from claim_hc.launcher import resolve_adapter_from_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve the best Swift/PEFT adapter path from a training output dir.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Swift training output directory, for example outputs/swift_internvl3_fakett.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = resolve_adapter_from_output_dir(args.output_dir)
    print(adapter_path)


if __name__ == "__main__":
    main()
