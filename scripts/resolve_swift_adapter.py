from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve the best Swift/PEFT adapter path from a training output dir.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Swift training output directory, for example outputs/swift_internvl3_fakett.",
    )
    return parser.parse_args()


def is_adapter_dir(path: Path) -> bool:
    return any(
        (path / filename).is_file()
        for filename in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    )


def nested_adapter_dir(path: Path) -> Path | None:
    for adapter_cfg in sorted(path.glob("checkpoint-*/adapter_config.json")):
        return adapter_cfg.parent
    for adapter_cfg in sorted(path.rglob("adapter_config.json")):
        return adapter_cfg.parent
    return None


def resolve_best_adapter_path(output_dir: Path) -> Path:
    trainer_state = output_dir / "trainer_state.json"
    if trainer_state.is_file():
        with trainer_state.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        best_model_checkpoint = state.get("best_model_checkpoint")
        if best_model_checkpoint:
            best_path = Path(best_model_checkpoint)
            if is_adapter_dir(best_path):
                return best_path
            nested = nested_adapter_dir(best_path)
            if nested is not None:
                return nested

    checkpoint_dirs = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda item: int(item.name.split("-")[-1]) if item.name.split("-")[-1].isdigit() else -1,
    )
    for checkpoint_dir in reversed(checkpoint_dirs):
        if is_adapter_dir(checkpoint_dir):
            return checkpoint_dir
        nested = nested_adapter_dir(checkpoint_dir)
        if nested is not None:
            return nested

    if is_adapter_dir(output_dir):
        return output_dir

    nested = nested_adapter_dir(output_dir)
    if nested is not None:
        return nested

    raise FileNotFoundError(
        f"No adapter checkpoint found under {output_dir}. Expected adapter_config.json in checkpoint-*."
    )


def main() -> None:
    args = parse_args()
    adapter_path = resolve_best_adapter_path(args.output_dir.resolve())
    print(adapter_path)


if __name__ == "__main__":
    main()
