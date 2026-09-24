from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STREAM_NAMES = ("Multi", "Visual", "Text", "Audio")
STREAM_LABELS = ("M", "V", "T", "A")
CLASS_NAMES = {0: "real", 1: "fake"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select four correctly predicted samples with distinct modality-dominant routing "
            "patterns and render a per-sample router heatmap."
        )
    )
    parser.add_argument(
        "evidence_archive",
        type=Path,
        help="NPZ produced by scripts/export_evidence_features.py.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Allow incorrectly predicted samples during representative-case selection.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=("max-diversity", "modality-dominant"),
        default="max-diversity",
        help=(
            "max-diversity selects four routing distributions using greedy max-min "
            "Jensen-Shannon distance; modality-dominant requires one sample dominated by each stream."
        ),
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.evidence_archive.parent / "router_per_sample"
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("sample_ids", "labels", "predictions", "router_weights")
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"Evidence archive is missing: {', '.join(missing)}")
        arrays = {key: archive[key] for key in required}

    sample_ids = arrays["sample_ids"].astype(str)
    labels = arrays["labels"].astype(np.int64)
    predictions = arrays["predictions"].astype(np.int64)
    weights = arrays["router_weights"].astype(np.float64)
    sample_count = len(sample_ids)
    if labels.shape != (sample_count,) or predictions.shape != (sample_count,):
        raise ValueError("sample_ids, labels, and predictions must have matching lengths")
    if weights.shape != (sample_count, len(STREAM_NAMES)):
        raise ValueError(
            f"router_weights must have shape [N, {len(STREAM_NAMES)}], got {weights.shape}"
        )
    if not np.isfinite(weights).all():
        raise ValueError("router_weights contains NaN or infinite values")
    if (weights < -1e-7).any():
        raise ValueError("router_weights contains negative values")
    row_sums = weights.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3, rtol=1e-3):
        raise ValueError(
            "router_weights are expected to be Softmax probabilities whose rows sum to one"
        )
    arrays["sample_ids"] = sample_ids
    arrays["labels"] = labels
    arrays["predictions"] = predictions
    arrays["router_weights"] = weights
    return arrays


def sample_metadata(
    arrays: dict[str, np.ndarray],
    index: int,
) -> dict[str, object]:
    labels = arrays["labels"]
    predictions = arrays["predictions"]
    weights = arrays["router_weights"]
    dominant_stream = int(weights[index].argmax())
    other_streams = [stream for stream in range(len(STREAM_NAMES)) if stream != dominant_stream]
    runner_up = float(weights[index, other_streams].max())
    return {
        "archive_index": int(index),
        "sample_id": str(arrays["sample_ids"][index]),
        "ground_truth": CLASS_NAMES.get(int(labels[index]), str(int(labels[index]))),
        "prediction": CLASS_NAMES.get(
            int(predictions[index]), str(int(predictions[index]))
        ),
        "correct": bool(labels[index] == predictions[index]),
        "dominant_stream": STREAM_NAMES[dominant_stream],
        "dominance_margin": float(weights[index, dominant_stream] - runner_up),
        "weights": weights[index].tolist(),
    }


def select_modality_dominant_samples(
    arrays: dict[str, np.ndarray],
    correct_only: bool,
) -> list[dict[str, object]]:
    sample_ids = arrays["sample_ids"]
    labels = arrays["labels"]
    predictions = arrays["predictions"]
    weights = arrays["router_weights"]
    eligible = labels == predictions if correct_only else np.ones(len(sample_ids), dtype=bool)
    dominant_stream = weights.argmax(axis=1)
    selected: list[dict[str, object]] = []

    for target_stream, stream_name in enumerate(STREAM_NAMES):
        candidate_indices = np.flatnonzero(eligible & (dominant_stream == target_stream))
        if not len(candidate_indices):
            qualifier = "correctly predicted " if correct_only else ""
            raise ValueError(
                f"No {qualifier}test sample is dominated by {stream_name}. "
                "This indicates router collapse or insufficient stream specialization; "
                "do not manufacture a four-stream heatmap."
            )

        other_streams = [index for index in range(len(STREAM_NAMES)) if index != target_stream]
        ranked: list[tuple[float, float, str, int]] = []
        for index in candidate_indices:
            target_weight = float(weights[index, target_stream])
            runner_up = float(weights[index, other_streams].max())
            margin = target_weight - runner_up
            ranked.append((-margin, -target_weight, str(sample_ids[index]), int(index)))
        _, _, _, chosen_index = min(ranked)
        selected.append(sample_metadata(arrays, chosen_index))
    return selected


def pairwise_jensen_shannon_distance(weights: np.ndarray) -> np.ndarray:
    probabilities = np.clip(weights.astype(np.float64, copy=False), 1e-12, None)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    left = probabilities[:, None, :]
    right = probabilities[None, :, :]
    midpoint = 0.5 * (left + right)
    divergence = 0.5 * np.sum(left * np.log(left / midpoint), axis=-1)
    divergence += 0.5 * np.sum(right * np.log(right / midpoint), axis=-1)
    return np.sqrt(np.maximum(divergence, 0.0))


def select_max_diversity_samples(
    arrays: dict[str, np.ndarray],
    correct_only: bool,
    count: int = 4,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    labels = arrays["labels"]
    predictions = arrays["predictions"]
    eligible = labels == predictions if correct_only else np.ones(len(labels), dtype=bool)
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < count:
        raise ValueError(f"Need at least {count} eligible test samples, found {len(eligible_indices)}")

    eligible_weights = arrays["router_weights"][eligible_indices]
    distances = pairwise_jensen_shannon_distance(eligible_weights)
    np.fill_diagonal(distances, -np.inf)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    selected_local = [int(first), int(second)]

    while len(selected_local) < count:
        candidate_score = distances[:, selected_local].min(axis=1)
        candidate_score[selected_local] = -np.inf
        selected_local.append(int(np.argmax(candidate_score)))

    selected_indices = [int(eligible_indices[index]) for index in selected_local]
    selected = [sample_metadata(arrays, index) for index in selected_indices]
    selected_distances = pairwise_jensen_shannon_distance(
        arrays["router_weights"][selected_indices]
    )
    off_diagonal = selected_distances[np.triu_indices(count, k=1)]
    for row_index, row in enumerate(selected):
        distances_to_others = np.delete(selected_distances[row_index], row_index)
        row["nearest_selected_js_distance"] = float(distances_to_others.min())
    diagnostics = {
        "minimum_pairwise_js_distance": float(off_diagonal.min()),
        "mean_pairwise_js_distance": float(off_diagonal.mean()),
        "maximum_pairwise_js_distance": float(off_diagonal.max()),
    }
    return selected, diagnostics


def save_source_data(path: Path, selected: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "display_name",
                "sample_id",
                "ground_truth",
                "prediction",
                "correct",
                "dominant_stream",
                "dominance_margin",
                *STREAM_NAMES,
            )
        )
        for row_index, sample in enumerate(selected, start=1):
            writer.writerow(
                (
                    f"Sample {row_index}",
                    sample["sample_id"],
                    sample["ground_truth"],
                    sample["prediction"],
                    sample["correct"],
                    sample["dominant_stream"],
                    f"{float(sample['dominance_margin']):.8f}",
                    *(f"{float(value):.8f}" for value in sample["weights"]),
                )
            )


def text_color(cmap: matplotlib.colors.Colormap, normalized_value: float) -> str:
    red, green, blue, _ = cmap(float(np.clip(normalized_value, 0.0, 1.0)))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.52 else "white"


def save_heatmap(
    output_dir: Path,
    selected: list[dict[str, object]],
    dpi: int,
) -> None:
    matrix = np.asarray([sample["weights"] for sample in selected], dtype=np.float64)
    cmap = matplotlib.colormaps["YlGnBu"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(5.4, 4.1), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="equal")
    axis.set_xticks(range(len(STREAM_LABELS)), labels=STREAM_LABELS)
    axis.set_yticks(
        range(len(selected)), labels=[f"Sample {index}" for index in range(1, len(selected) + 1)]
    )
    axis.tick_params(
        axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=7
    )
    axis.tick_params(axis="y", length=0, pad=7)
    axis.xaxis.set_label_position("top")
    axis.set_xlabel("Evidence stream", labelpad=11)
    axis.set_ylabel("Test sample", labelpad=9)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color(cmap, value),
                fontsize=10,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.05)
    colorbar.set_label("Routing weight", rotation=90, labelpad=10)
    colorbar.outline.set_linewidth(0.6)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)

    stem = output_dir / "router_per_sample_heatmap"
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    arrays = load_archive(args.evidence_archive)
    correct_only = not args.include_incorrect
    if args.selection_strategy == "max-diversity":
        selected, diversity = select_max_diversity_samples(arrays, correct_only=correct_only)
        selection_rule = (
            "Among eligible test samples, initialize with the pair having maximum "
            "Jensen-Shannon distance, then greedily add each sample that maximizes its minimum "
            "distance to the selected set."
        )
    else:
        selected = select_modality_dominant_samples(arrays, correct_only=correct_only)
        diversity = None
        selection_rule = (
            "For each evidence stream, select the eligible sample dominated by that stream with "
            "the largest margin over its second-highest routing weight."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_source_data(args.output_dir / "router_per_sample_source_data.csv", selected)
    metadata = {
        "source_archive": str(args.evidence_archive.resolve()),
        "source_sample_count": int(len(arrays["sample_ids"])),
        "eligible_sample_count": int(
            len(arrays["sample_ids"])
            if args.include_incorrect
            else np.count_nonzero(arrays["labels"] == arrays["predictions"])
        ),
        "correct_only": not args.include_incorrect,
        "selection_strategy": args.selection_strategy,
        "selection_rule": selection_rule,
        "diversity_diagnostics": diversity,
        "stream_order": list(STREAM_NAMES),
        "stream_labels": list(STREAM_LABELS),
        "samples": selected,
    }
    with (args.output_dir / "router_per_sample_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    save_heatmap(args.output_dir, selected, args.dpi)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved per-sample router heatmap to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
