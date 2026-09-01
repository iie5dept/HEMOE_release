from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "model.lora_dropout": [0.0, 0.05, 0.1],
    "fusion.expert_dim": [256, 512, 768],
    "fusion.router_dim": [64, 128, 256],
    "fusion.dropout": [0.05, 0.1, 0.2],
    "fusion.modality_loss_weight": [0.0, 0.1, 0.3, 0.5],
    "training.lora_learning_rate": [1.0e-5, 2.0e-5, 4.0e-5],
    "training.head_learning_rate": [5.0e-5, 1.0e-4, 2.0e-4, 3.0e-4],
    "training.weight_decay": [0.0, 0.01, 0.05],
}


@dataclass
class TrialResult:
    trial: int
    status: str
    metric_name: str
    metric_value: float | None
    best_epoch: int | None
    duration_seconds: float
    config_path: str
    checkpoint_path: str | None
    log_path: str
    parameters: dict[str, Any]
    validation_metrics: dict[str, float]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential four-GPU hyperparameter search for the four-modal MoE classifier. "
            "Trials are selected on validation macro-F1; the best checkpoint is evaluated once on test."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--num-trials", type=int, default=12)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--search-space", type=Path, help="Optional YAML mapping dotted keys to value lists.")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--nproc-per-node", type=int, default=0)
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--keep-epoch-checkpoints", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_trials <= 0:
        parser.error("--num-trials must be positive")
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        parser.error("--gpus cannot be empty")
    args.gpu_ids = gpu_ids
    if args.nproc_per_node <= 0:
        args.nproc_per_node = len(gpu_ids)
    if args.nproc_per_node != len(gpu_ids):
        parser.error("--nproc-per-node must equal the number of IDs in --gpus")
    return args


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            raise KeyError(f"Cannot set {dotted_key!r}: {part!r} is not a mapping")
        target = nested
    if parts[-1] not in target:
        raise KeyError(f"Cannot tune missing config key {dotted_key!r}")
    target[parts[-1]] = value


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing config key {dotted_key!r}")
        value = value[part]
    return value


def load_search_space(path: Path | None, base_config: dict[str, Any]) -> dict[str, list[Any]]:
    space = read_yaml(path.resolve()) if path else deepcopy(DEFAULT_SEARCH_SPACE)
    if not space:
        raise ValueError("Search space cannot be empty")
    for key, values in space.items():
        get_nested(base_config, str(key))
        if not isinstance(values, list) or not values:
            raise ValueError(f"Search-space entry {key!r} must be a non-empty list")
    return {str(key): values for key, values in space.items()}


def parameter_signature(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def generate_parameter_sets(
    base_config: dict[str, Any],
    search_space: dict[str, list[Any]],
    num_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline = {key: get_nested(base_config, key) for key in search_space}
    parameter_sets = [baseline]
    seen = {parameter_signature(baseline)}
    rng = random.Random(seed)
    combinations = 1
    for values in search_space.values():
        combinations *= len(values)
    target = min(num_trials, combinations)
    attempts = 0
    max_attempts = max(1000, target * 100)
    while len(parameter_sets) < target and attempts < max_attempts:
        attempts += 1
        candidate = {key: rng.choice(values) for key, values in search_space.items()}
        signature = parameter_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        parameter_sets.append(candidate)
    if len(parameter_sets) != target:
        raise RuntimeError(f"Generated only {len(parameter_sets)} of {target} unique trials")
    return parameter_sets


def build_trial_config(
    base_config: dict[str, Any],
    parameters: dict[str, Any],
    trial_dir: Path,
    metric_name: str,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    for key, value in parameters.items():
        set_nested(config, key, value)
    training = config.setdefault("training", {})
    training["evaluate"] = True
    training["evaluation_split"] = "val"
    training["metric_for_best_model"] = metric_name
    training["greater_is_better"] = True
    training["output_dir"] = str((trial_dir / "checkpoints").resolve())
    config.setdefault("inference", {})["split"] = "test"
    return config


def stream_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def parse_validation_metrics(log_path: Path, split: str = "val") -> list[dict[str, float]]:
    entries: list[dict[str, float]] = []
    prefix = f"{split}/"
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            metrics = {
                str(key)[len(prefix) :]: float(value)
                for key, value in payload.items()
                if str(key).startswith(prefix) and isinstance(value, (int, float))
            }
            if metrics:
                entries.append(metrics)
    return entries


def best_validation_entry(
    entries: list[dict[str, float]], metric_name: str, metric_value: float
) -> dict[str, float]:
    candidates = [entry for entry in entries if metric_name in entry]
    if not candidates:
        return {}
    return min(candidates, key=lambda entry: abs(entry[metric_name] - metric_value))


def prune_epoch_checkpoints(checkpoint_root: Path) -> None:
    resolved_root = checkpoint_root.resolve()
    for path in checkpoint_root.glob("checkpoint-epoch-*"):
        resolved_path = path.resolve()
        if resolved_path.parent != resolved_root or not resolved_path.is_dir():
            raise RuntimeError(f"Refusing to prune unexpected checkpoint path: {resolved_path}")
        shutil.rmtree(resolved_path)


def read_existing_result(path: Path, parameters: dict[str, Any]) -> TrialResult | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if parameter_signature(payload.get("parameters", {})) != parameter_signature(parameters):
        raise RuntimeError(f"Existing trial parameters do not match generated parameters: {path}")
    checkpoint = payload.get("checkpoint_path")
    if payload.get("status") != "complete" or not checkpoint or not Path(checkpoint).is_dir():
        return None
    return TrialResult(**payload)


def run_trial(
    trial_index: int,
    base_config: dict[str, Any],
    parameters: dict[str, Any],
    output_root: Path,
    args: argparse.Namespace,
) -> TrialResult:
    trial_dir = output_root / "trials" / f"trial-{trial_index:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.yaml"
    log_path = trial_dir / "train.log"
    result_path = trial_dir / "result.json"
    config = build_trial_config(base_config, parameters, trial_dir, args.metric)
    write_yaml(config_path, config)
    if not args.rerun_completed:
        existing = read_existing_result(result_path, parameters)
        if existing is not None:
            print(f"[trial {trial_index:03d}] reusing {existing.checkpoint_path}", flush=True)
            return existing

    print(
        json.dumps(
            {"trial": trial_index, "parameters": parameters, "config": str(config_path)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        result = TrialResult(
            trial=trial_index,
            status="dry_run",
            metric_name=args.metric,
            metric_value=None,
            best_epoch=None,
            duration_seconds=0.0,
            config_path=str(config_path.resolve()),
            checkpoint_path=None,
            log_path=str(log_path.resolve()),
            parameters=parameters,
            validation_metrics={},
        )
        atomic_json(result_path, asdict(result))
        return result

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu_ids)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(REPO_ROOT / "scripts" / "train_trimodal_moe.py"),
        str(config_path.resolve()),
    ]
    started = time.monotonic()
    return_code = stream_command(command, log_path, env)
    duration = time.monotonic() - started
    checkpoint_root = Path(config["training"]["output_dir"])
    checkpoint_path = checkpoint_root / "checkpoint-best"
    metadata_path = checkpoint_path / "best_metric.json"
    try:
        if return_code != 0:
            raise RuntimeError(f"training exited with code {return_code}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Training completed without {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metric_name = str(metadata["metric_name"])
        metric_value = float(metadata["metric_value"])
        if metric_name != args.metric:
            raise ValueError(f"Expected metric {args.metric!r}, checkpoint reports {metric_name!r}")
        entries = parse_validation_metrics(log_path)
        result = TrialResult(
            trial=trial_index,
            status="complete",
            metric_name=metric_name,
            metric_value=metric_value,
            best_epoch=int(metadata["epoch"]),
            duration_seconds=duration,
            config_path=str(config_path.resolve()),
            checkpoint_path=str(checkpoint_path.resolve()),
            log_path=str(log_path.resolve()),
            parameters=parameters,
            validation_metrics=best_validation_entry(entries, metric_name, metric_value),
        )
        if not args.keep_epoch_checkpoints:
            prune_epoch_checkpoints(checkpoint_root)
    except Exception as error:
        result = TrialResult(
            trial=trial_index,
            status="failed",
            metric_name=args.metric,
            metric_value=None,
            best_epoch=None,
            duration_seconds=duration,
            config_path=str(config_path.resolve()),
            checkpoint_path=None,
            log_path=str(log_path.resolve()),
            parameters=parameters,
            validation_metrics={},
            error=f"{type(error).__name__}: {error}",
        )
    atomic_json(result_path, asdict(result))
    return result


def save_leaderboard(output_root: Path, results: list[TrialResult]) -> None:
    ordered = sorted(
        results,
        key=lambda item: (
            item.status != "complete",
            -(item.metric_value if item.metric_value is not None else float("-inf")),
            item.trial,
        ),
    )
    atomic_json(output_root / "leaderboard.json", [asdict(result) for result in ordered])
    parameter_keys = sorted({key for result in results for key in result.parameters})
    fieldnames = [
        "rank",
        "trial",
        "status",
        "metric_name",
        "metric_value",
        "best_epoch",
        "duration_seconds",
        "checkpoint_path",
        "error",
        *parameter_keys,
    ]
    with (output_root / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(ordered, start=1):
            row = {
                "rank": rank if result.status == "complete" else "",
                "trial": result.trial,
                "status": result.status,
                "metric_name": result.metric_name,
                "metric_value": result.metric_value,
                "best_epoch": result.best_epoch,
                "duration_seconds": round(result.duration_seconds, 3),
                "checkpoint_path": result.checkpoint_path,
                "error": result.error,
            }
            row.update(result.parameters)
            writer.writerow(row)


def evaluate_predictions(path: Path) -> dict[str, Any]:
    labels = ("real", "fake")
    confusion = {label: {prediction: 0 for prediction in labels} for label in labels}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            label = str(record["label"]).lower()
            prediction = str(record["prediction"]).lower()
            if label not in labels or prediction not in labels:
                raise ValueError(f"Invalid label at {path}:{line_number}")
            confusion[label][prediction] += 1
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in labels)
    per_class = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(confusion[label][other] for other in labels if other != label)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }
    return {
        "accuracy": correct / max(1, total),
        "macro_precision": sum(value["precision"] for value in per_class.values()) / len(labels),
        "macro_recall": sum(value["recall"] for value in per_class.values()) / len(labels),
        "macro_f1": sum(value["f1"] for value in per_class.values()) / len(labels),
        "total": total,
        "confusion": confusion,
        "per_class": per_class,
    }


def run_final_test(
    best_config: Path,
    best_checkpoint: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prediction_path = output_root / "best_test_predictions.jsonl"
    log_path = output_root / "best_test_inference.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu_ids)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(REPO_ROOT / "scripts" / "infer_trimodal_moe.py"),
        str(best_config),
        "--checkpoint",
        str(best_checkpoint),
        "--output",
        str(prediction_path),
    ]
    return_code = stream_command(command, log_path, env)
    if return_code != 0:
        raise RuntimeError(f"Best-checkpoint inference exited with code {return_code}")
    metrics = evaluate_predictions(prediction_path)
    atomic_json(output_root / "best_test_metrics.json", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    base_config = read_yaml(config_path)
    if "training" not in base_config or "fusion" not in base_config:
        raise ValueError("Config must contain training and fusion mappings")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else Path(str(base_config["training"]["output_dir"]) + "_tuning").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "base_config.yaml")
    search_space = load_search_space(args.search_space, base_config)
    atomic_json(output_root / "search_space.json", search_space)
    parameter_sets = generate_parameter_sets(
        base_config, search_space, args.num_trials, args.search_seed
    )
    atomic_json(output_root / "planned_trials.json", parameter_sets)

    results: list[TrialResult] = []
    for trial_index, parameters in enumerate(parameter_sets):
        result = run_trial(trial_index, base_config, parameters, output_root, args)
        results.append(result)
        save_leaderboard(output_root, results)
        print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
        if result.status == "failed" and args.fail_fast:
            raise RuntimeError(result.error)

    if args.dry_run:
        print(json.dumps({"dry_run": len(results), "output_root": str(output_root)}, ensure_ascii=False))
        return
    completed = [result for result in results if result.status == "complete" and result.metric_value is not None]
    if not completed:
        raise RuntimeError(f"No tuning trial completed successfully; inspect {output_root / 'leaderboard.json'}")
    best = max(completed, key=lambda result: (float(result.metric_value), -result.trial))
    best_config = output_root / "best_config.yaml"
    shutil.copy2(best.config_path, best_config)
    best_checkpoint = Path(str(best.checkpoint_path))
    (output_root / "best_checkpoint.txt").write_text(str(best_checkpoint) + "\n", encoding="utf-8")
    best_payload = asdict(best)
    best_payload["best_config"] = str(best_config)
    atomic_json(output_root / "best_result.json", best_payload)

    test_metrics = None
    if not args.skip_final_test:
        test_metrics = run_final_test(best_config, best_checkpoint, output_root, args)
    print(
        json.dumps(
            {
                "best_trial": best.trial,
                "validation_metric": best.metric_value,
                "best_config": str(best_config),
                "best_checkpoint": str(best_checkpoint),
                "test_metrics": test_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
