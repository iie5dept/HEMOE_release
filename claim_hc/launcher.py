from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _normalize_cli_value(value: Any) -> list[str]:
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_normalize_cli_value(item))
        return result
    return [str(value)]


def expand_config_argv(argv: list[str]) -> list[str]:
    if not argv:
        return []
    first = argv[0]
    if first.startswith("-") or not first.endswith((".yaml", ".yml")):
        return argv

    config_path = Path(first)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    env_mapping = config.pop("ENV", {}) or {}
    for key, value in env_mapping.items():
        os.environ[str(key)] = str(value)

    expanded: list[str] = []
    for key, value in config.items():
        expanded.append(f"--{key}")
        expanded.extend(_normalize_cli_value(value))
    expanded.extend(argv[1:])
    return expanded


def rewrite_sys_argv(argv: list[str]) -> None:
    sys.argv = [sys.argv[0]] + expand_config_argv(argv)


def maybe_relaunch_distributed() -> None:
    nproc_per_node = int(os.environ.get("NPROC_PER_NODE", "1"))
    if nproc_per_node <= 1:
        return
    if os.environ.get("LOCAL_RANK") is not None or os.environ.get("RANK") is not None:
        return

    launcher_argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        sys.argv[0],
        *sys.argv[1:],
    ]
    os.execvpe(sys.executable, launcher_argv, os.environ.copy())
