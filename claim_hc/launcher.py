from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

_ADAPTER_ENV = "VIDEOMMD_CLAIM_HC_ADAPTERS"


def _apply_project_config(config: dict[str, Any]) -> None:
    project_config = config.pop("VIDEOMMD", {}) or {}
    if not isinstance(project_config, dict):
        raise TypeError("VIDEOMMD must be a YAML mapping.")

    adapter_output_dir = project_config.pop("adapter_output_dir", None)
    if adapter_output_dir is not None:
        raise ValueError(
            "VIDEOMMD.adapter_output_dir has been removed. "
            "Specify the checkpoint manually with top-level `adapters` in YAML "
            "or pass `--adapters /path/to/checkpoint-xxx` on the command line."
        )

    if project_config:
        unknown = ", ".join(sorted(str(key) for key in project_config))
        raise ValueError(f"Unknown VIDEOMMD config keys: {unknown}")


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


def _remember_adapter_args(argv: list[str]) -> None:
    adapter_values: list[str] = []
    for index, argument in enumerate(argv):
        if argument != "--adapters":
            continue
        values: list[str] = []
        for value in argv[index + 1:]:
            if value.startswith("--"):
                break
            values.append(value)
        if values:
            adapter_values = values
    if adapter_values:
        os.environ[_ADAPTER_ENV] = ",".join(adapter_values)


def _has_cli_flag(argv: list[str], flag: str) -> bool:
    return any(argument == flag for argument in argv)


def expand_config_argv(argv: list[str]) -> list[str]:
    if not argv:
        return []
    first = argv[0]
    if first.startswith("-") or not first.endswith((".yaml", ".yml")):
        _remember_adapter_args(argv)
        return argv

    config_path = Path(first)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a YAML mapping: {config_path}")

    env_mapping = config.pop("ENV", {}) or {}
    if not isinstance(env_mapping, dict):
        raise TypeError("ENV must be a YAML mapping.")
    for key, value in env_mapping.items():
        os.environ[str(key)] = str(value)

    _apply_project_config(config)
    if _has_cli_flag(argv[1:], "--adapters"):
        config.pop("adapters", None)

    expanded: list[str] = []
    for key, value in config.items():
        expanded.append(f"--{key}")
        expanded.extend(_normalize_cli_value(value))
    expanded.extend(argv[1:])
    _remember_adapter_args(expanded)
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
