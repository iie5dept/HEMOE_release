from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Deploy repo-managed claim-HC forked files to the active Swift and InternVL locations."
    )
    parser.add_argument("--fork-dir", type=Path, default=repo_root / "forks" / "claim_hc")
    parser.add_argument("--swift-template-file", type=Path, required=True)
    parser.add_argument("--modeling-file", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=repo_root / "forks" / "claim_hc_backups")
    parser.add_argument("--tag", type=str, default="manual")
    return parser.parse_args()


def backup_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Fork file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    fork_template = args.fork_dir / "internvl.py"
    fork_modeling = args.fork_dir / "modeling_internvl_chat.py"

    backup_root = args.backup_dir / args.tag
    backup_template = backup_root / "internvl.py"
    backup_modeling = backup_root / "modeling_internvl_chat.py"

    backup_file(args.swift_template_file, backup_template)
    backup_file(args.modeling_file, backup_modeling)

    copy_file(fork_template, args.swift_template_file)
    copy_file(fork_modeling, args.modeling_file)

    print(
        json.dumps(
            {
                "ok": True,
                "fork_dir": str(args.fork_dir),
                "backup_dir": str(backup_root),
                "swift_template_file": str(args.swift_template_file),
                "modeling_file": str(args.modeling_file),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
