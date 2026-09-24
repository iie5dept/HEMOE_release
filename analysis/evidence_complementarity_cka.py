from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_GROUPS = {
    "branch": ("H_m", "H_v", "H_t", "H_a"),
    "routed": ("Z_m", "Z_v", "Z_t", "Z_a"),
}
DISPLAY_NAMES = ("M", "V", "T", "A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot a four-branch Linear CKA evidence-complementarity matrix."
    )
    parser.add_argument("features", type=Path, help="NPZ produced by export_evidence_features.py")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--representation",
        choices=tuple(FEATURE_GROUPS),
        default="branch",
        help=(
            "branch compares pre-MED evidence representations; routed compares the four blocks "
            "of the final fused feature passed to the linear classifier."
        ),
    )
    parser.add_argument(
        "--title",
        default="",
        help="Deprecated compatibility option; the figure is intentionally rendered without a title.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--value-font-size",
        type=float,
        default=14.0,
        help="Font size of the numeric CKA annotations inside heatmap cells.",
    )
    parser.add_argument(
        "--tick-font-size",
        type=float,
        default=14.0,
        help="Font size of the M/V/T/A labels on both heatmap axes.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use a space-efficient layout for a half-column side-by-side figure.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.features.parent / "evidence_complementarity"
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.value_font_size <= 0:
        parser.error("--value-font-size must be positive")
    if args.tick_font_size <= 0:
        parser.error("--tick-font-size must be positive")
    return args


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Compute biased Linear CKA after centering across samples."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Expected [N, D] arrays with matching N, got {x.shape} and {y.shape}")
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    self_x = np.linalg.norm(x.T @ x, ord="fro")
    self_y = np.linalg.norm(y.T @ y, ord="fro")
    denominator = self_x * self_y
    if denominator <= eps:
        raise ValueError("CKA is undefined because one representation has zero centered variance")
    return float(cross / denominator)


def cka_matrix(features: list[np.ndarray]) -> np.ndarray:
    matrix = np.empty((len(features), len(features)), dtype=np.float64)
    for row, x in enumerate(features):
        for column, y in enumerate(features):
            matrix[row, column] = linear_cka(x, y)
    return matrix


def save_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("branch", *DISPLAY_NAMES))
        for name, row in zip(DISPLAY_NAMES, matrix):
            writer.writerow((name, *(f"{value:.6f}" for value in row)))


def save_heatmap(
    output_dir: Path,
    matrix: np.ndarray,
    dpi: int,
    compact: bool,
    basename: str,
    value_font_size: float,
    tick_font_size: float,
) -> None:
    base_font_size = 10 if compact else 14
    axis_label_size = 11 if compact else 16
    colorbar_label_size = 10 if compact else 15
    colorbar_tick_size = 9 if compact else 13
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "normal",
            "font.size": base_font_size,
            "axes.labelsize": axis_label_size,
            "axes.labelweight": "normal",
            "xtick.labelsize": tick_font_size,
            "ytick.labelsize": tick_font_size,
        }
    )
    figure_size = (4.5, 3.6) if compact else (6.2, 5.4)
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="equal")
    axis.set_xticks(range(len(DISPLAY_NAMES)), labels=DISPLAY_NAMES)
    axis.set_yticks(range(len(DISPLAY_NAMES)), labels=DISPLAY_NAMES)
    axis.tick_params(length=0)
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontsize(tick_font_size)
        label.set_fontweight("normal")
    for row in range(len(DISPLAY_NAMES)):
        for column in range(len(DISPLAY_NAMES)):
            color = "white" if matrix[row, column] > 0.58 else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=value_font_size,
                fontweight="normal",
            )
    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.04 if compact else 0.046,
        pad=0.025 if compact else 0.04,
    )
    colorbar.set_label(
        "CKA" if compact else "Linear CKA",
        fontsize=colorbar_label_size,
        fontweight="normal",
        labelpad=5 if compact else 10,
    )
    colorbar.ax.tick_params(labelsize=colorbar_tick_size)
    if not compact:
        axis.set_xlabel("Evidence branch", fontsize=16, fontweight="normal", labelpad=10)
        axis.set_ylabel("Evidence branch", fontsize=16, fontweight="normal", labelpad=10)
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(
            output_dir / f"{basename}.{extension}",
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            **kwargs,
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.features.is_file():
        raise FileNotFoundError(f"Feature archive not found: {args.features}")
    feature_keys = FEATURE_GROUPS[args.representation]
    with np.load(args.features, allow_pickle=False) as archive:
        missing = [name for name in feature_keys if name not in archive]
        if missing:
            hint = (
                " Re-run scripts/export_evidence_features.py with the updated exporter."
                if args.representation == "routed"
                else ""
            )
            raise KeyError(f"Feature archive is missing: {', '.join(missing)}.{hint}")
        features = [archive[name] for name in feature_keys]
        sample_ids = archive["sample_ids"] if "sample_ids" in archive else None

    num_samples = int(features[0].shape[0])
    shapes = {name: list(value.shape) for name, value in zip(feature_keys, features)}
    if num_samples <= 1:
        raise ValueError(f"Linear CKA requires more than one sample, got {num_samples}")
    if any(value.shape[0] != num_samples for value in features):
        raise ValueError(f"Evidence matrices are not sample-aligned: {shapes}")
    if sample_ids is not None and len(sample_ids) != num_samples:
        raise ValueError(
            f"sample_ids contains {len(sample_ids)} records, but evidence contains {num_samples}"
        )

    matrix = cka_matrix(features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = (
        "evidence_complementarity_cka"
        if args.representation == "branch"
        else "routed_evidence_complementarity_cka"
    )
    save_csv(args.output_dir / f"{basename}.csv", matrix)
    save_heatmap(
        args.output_dir,
        matrix,
        args.dpi,
        args.compact,
        basename,
        args.value_font_size,
        args.tick_font_size,
    )
    summary = {
        "features": str(args.features.resolve()),
        "representation": args.representation,
        "num_samples": num_samples,
        "feature_shapes": shapes,
        "streams": list(DISPLAY_NAMES),
        "compact": args.compact,
        "value_font_size": args.value_font_size,
        "tick_font_size": args.tick_font_size,
        "linear_cka": matrix.tolist(),
    }
    with (args.output_dir / f"{basename}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"Saved CKA figure and values to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
