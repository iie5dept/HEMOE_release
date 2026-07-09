from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Snapshot the currently used Swift InternVL template and InternVL modeling file into the repo."
    )
    parser.add_argument("--swift-template-file", type=Path, required=True)
    parser.add_argument("--modeling-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "forks" / "claim_hc")
    parser.add_argument("--tag", type=str, default="current")
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    template_dst = output_dir / "internvl.py"
    modeling_dst = output_dir / "modeling_internvl_chat.py"
    meta_dst = output_dir / "snapshot_meta.json"

    copy_file(args.swift_template_file, template_dst)
    copy_file(args.modeling_file, modeling_dst)

    meta = {
        "tag": args.tag,
        "swift_template_file": str(args.swift_template_file),
        "modeling_file": str(args.modeling_file),
        "template_snapshot": str(template_dst),
        "modeling_snapshot": str(modeling_dst),
    }
    meta_dst.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "output_dir": str(output_dir), "meta": str(meta_dst)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
