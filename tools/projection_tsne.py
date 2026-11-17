"""Build a t-SNE visualization from saved per-prompt projected Jacobians."""

from __future__ import annotations

import json
import logging
import math
import argparse
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import numpy as np

logger = logging.getLogger(__name__)


def create_projection_tsne(
    projected_dir: str | Path | Sequence[str | Path],
    *,
    output_dir: str | Path | None = None,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> dict[Path, tuple[Path, Path, Path, Path]]:
    """Create t-SNE outputs from one or more projected-feature directories.

    Each input may be a projected/ root containing configuration subdirectories
    or a configuration directory containing dataset feature files. Multiple
    directories are combined into one embedding and require a shared projection
    configuration.
    """
    input_dirs = (
        [Path(projected_dir)]
        if isinstance(projected_dir, (str, Path))
        else [Path(path) for path in projected_dir]
    )
    if not input_dirs:
        raise ValueError("At least one projected directory is required.")

    feature_dirs_by_input: list[list[Path]] = []
    for input_dir in input_dirs:
        if any(input_dir.glob("dataset_*.npz")):
            feature_dirs_by_input.append([input_dir])
        elif input_dir.is_dir():
            feature_dirs_by_input.append(
                [
                    child
                    for child in sorted(input_dir.iterdir())
                    if child.is_dir() and any(child.glob("dataset_*.npz"))
                ]
            )
        else:
            feature_dirs_by_input.append([])
    feature_dirs = [
        feature_dir
        for feature_dirs in feature_dirs_by_input
        for feature_dir in feature_dirs
    ]
    if not feature_dirs:
        raise FileNotFoundError(
            f"No per-dataset projected feature files found in {input_dirs} "
            "or their immediate subdirectories."
        )
    if len(input_dirs) == 1 and output_dir is None:
        return {
            feature_dir: _create_projection_tsne_for_directory(
                feature_dir,
                perplexity=perplexity,
                random_state=random_state,
            )
            for feature_dir in feature_dirs
        }
    if len(input_dirs) == 1 and len(feature_dirs) > 1:
        destination_root = (
            Path(output_dir) if output_dir is not None else feature_dirs[0].parent
        )
        return {
            destination_root / feature_dir.name: _create_projection_tsne_for_directory(
                feature_dir,
                output_dir=destination_root / feature_dir.name,
                perplexity=perplexity,
                random_state=random_state,
            )
            for feature_dir in feature_dirs
        }
    if len(input_dirs) > 1 and any(
        len(feature_dirs) != 1 for feature_dirs in feature_dirs_by_input
    ):
        raise ValueError(
            "When combining multiple inputs, pass one projected configuration "
            "directory per --projected-dir."
        )
    feature_dirs = list(dict.fromkeys(feature_dirs))
    if len(feature_dirs) > 1 or output_dir is not None:
        combined_output_dir = (
            Path(output_dir)
            if output_dir is not None
            else feature_dirs[0].parent / "combined"
        )
        feature_files = sorted(
            feature_file
            for feature_dir in feature_dirs
            for feature_file in feature_dir.glob("dataset_*.npz")
        )
        return {
            combined_output_dir: _create_projection_tsne_for_directory(
                feature_dirs[0],
                feature_files=feature_files,
                output_dir=combined_output_dir,
                perplexity=perplexity,
                random_state=random_state,
            )
        }
    return {
        feature_dir: _create_projection_tsne_for_directory(
            feature_dir,
            perplexity=perplexity,
            random_state=random_state,
        )
        for feature_dir in feature_dirs
    }


def plot_tsne_by_label(
    embedding_path: str | Path,
    label_key: str,
) -> Path:
    """Recolor a saved embedding by one scalar field in its sample labels."""
    embedding_path = Path(embedding_path)
    if not embedding_path.is_file():
        raise FileNotFoundError(f"t-SNE embedding not found: {embedding_path}")
    if not isinstance(label_key, str) or not label_key.strip():
        raise ValueError("label_key must be a non-empty string.")

    with np.load(embedding_path, allow_pickle=False) as data:
        embedding = np.asarray(data["embedding"], dtype=np.float32)
        if "sample_labels_json" not in data:
            raise ValueError(
                f"{embedding_path} has no saved sample labels; "
                "recreate t-SNE from projected features."
            )
        sample_labels_json = np.asarray(data["sample_labels_json"], dtype=np.str_)
        normalization = (
            str(data["feature_normalization"].item())
            if "feature_normalization" in data
            else "unknown"
        )

    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError(f"{embedding_path} must contain a two-dimensional embedding.")
    if sample_labels_json.shape != (embedding.shape[0],):
        raise ValueError(f"{embedding_path} has mismatched embedding and label counts.")

    values: list[str] = []
    for sample_index, label_json in enumerate(sample_labels_json):
        try:
            sample_label = json.loads(str(label_json))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid label JSON at embedding row {sample_index}."
            ) from exc
        if not isinstance(sample_label, dict) or label_key not in sample_label:
            raise ValueError(
                f"Embedding row {sample_index} has no label field {label_key!r}."
            )
        value = sample_label[label_key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Label {label_key!r} at embedding row {sample_index} "
                "must be a non-empty scalar string."
            )
        values.append(value)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = sorted(set(values))
    colors = plt.get_cmap("tab20", max(len(categories), 1))
    figure, axis = plt.subplots(figsize=(9, 7))
    for category_index, category in enumerate(categories):
        mask = np.asarray([value == category for value in values])
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=category,
            alpha=0.8,
            color=colors(category_index),
        )
    axis.set_title(f"t-SNE colored by {label_key} ({normalization})")
    axis.set_xlabel("t-SNE component 1")
    axis.set_ylabel("t-SNE component 2")
    axis.legend(title=label_key, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    output_directory = embedding_path.parent / (
        "normalized" if normalization == "l2_per_prompt" else "normal"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"{embedding_path.stem}_by_{quote(label_key, safe='._-')}.png"
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def create_block_projection_tsne(
    feature_path: str | Path,
    *,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> dict[tuple[int, int], tuple[Path, Path]]:
    """Create raw and normalized t-SNE embeddings for each saved layer block."""
    feature_path = Path(feature_path)
    if not feature_path.is_file():
        raise FileNotFoundError(f"Projected feature archive not found: {feature_path}")
    if (
        not isinstance(perplexity, (int, float))
        or isinstance(perplexity, bool)
        or not math.isfinite(perplexity)
        or perplexity <= 0
    ):
        raise ValueError("t-SNE perplexity must be a positive finite number.")
    if (
        not isinstance(random_state, int)
        or isinstance(random_state, bool)
        or random_state < 0
    ):
        raise ValueError("t-SNE random_state must be a non-negative integer.")

    with np.load(feature_path, allow_pickle=False) as data:
        if "block_features" not in data or "block_layer_ranges" not in data:
            raise ValueError(f"{feature_path} does not contain projected block features.")
        block_features = np.asarray(data["block_features"], dtype=np.float32)
        block_layer_ranges = np.asarray(data["block_layer_ranges"], dtype=np.int64)
        sample_ids = np.asarray(data["sample_ids"], dtype=np.str_)
        sample_labels_json = np.asarray(data["sample_labels_json"], dtype=np.str_)
        dataset_labels = np.asarray(data["dataset_labels"], dtype=np.str_)
        original_dim = int(data["original_dim"])
        projection_dim = int(data["projection_dim"])
        projection_seed = int(data["projection_seed"])
        projection_method = str(data["projection_method"].item())

    if (
        block_features.ndim != 4
        or block_features.shape[0] == 0
        or block_features.shape[2:] != (projection_dim, projection_dim)
    ):
        raise ValueError(
            f"{feature_path} has invalid block feature shape "
            f"{block_features.shape}."
        )
    if block_layer_ranges.shape != (block_features.shape[1], 2):
        raise ValueError(f"{feature_path} has mismatched block layer ranges.")
    if (
        sample_ids.shape != (block_features.shape[0],)
        or sample_labels_json.shape != (block_features.shape[0],)
        or dataset_labels.shape != (block_features.shape[0],)
    ):
        raise ValueError(f"{feature_path} has mismatched sample metadata.")
    if not np.isfinite(block_features).all():
        raise ValueError(f"{feature_path} contains non-finite block features.")
    for label_json in sample_labels_json:
        try:
            value = json.loads(str(label_json))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{feature_path} contains invalid sample label JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{feature_path} sample labels must be JSON objects.")

    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise ImportError(
            "t-SNE requires scikit-learn; install it in the active environment."
        ) from exc

    effective_perplexity = min(perplexity, float(block_features.shape[0] - 1))
    if block_features.shape[0] < 2:
        raise ValueError("t-SNE requires at least two samples.")
    effective_random_state = random_state % (2**32)
    output_paths: dict[tuple[int, int], tuple[Path, Path]] = {}
    for block_index, (start_layer, end_layer) in enumerate(block_layer_ranges):
        features = block_features[:, block_index].reshape(block_features.shape[0], -1)
        if all(np.array_equal(features[0], row) for row in features[1:]):
            raise ValueError(
                f"Block {start_layer}-{end_layer} has identical sample features."
            )
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized_features = np.divide(
            features,
            norms,
            out=np.zeros_like(features),
            where=norms > 0,
        )
        options = {
            "n_components": 2,
            "perplexity": effective_perplexity,
            "random_state": effective_random_state,
            "init": "random",
            "learning_rate": "auto",
        }
        block_dir = (
            feature_path.parent
            / "blocks"
            / f"layers_{start_layer}_{end_layer}"
        )
        block_dir.mkdir(parents=True, exist_ok=True)
        output_paths[(int(start_layer), int(end_layer))] = (
            block_dir / "tsne_embedding.npz",
            block_dir / "tsne_normalized_embedding.npz",
        )
        for embedding_path, values, normalization in (
            (
                output_paths[(int(start_layer), int(end_layer))][0],
                TSNE(**options).fit_transform(features),
                "none",
            ),
            (
                output_paths[(int(start_layer), int(end_layer))][1],
                TSNE(**options).fit_transform(normalized_features),
                "l2_per_prompt",
            ),
        ):
            np.savez_compressed(
                embedding_path,
                embedding=values,
                sample_ids=sample_ids,
                dataset_labels=dataset_labels,
                sample_labels_json=sample_labels_json,
                source_layers=np.asarray([start_layer, end_layer], dtype=np.int64),
                original_dim=np.asarray(original_dim, dtype=np.int64),
                projection_dim=np.asarray(projection_dim, dtype=np.int64),
                projection_seed=np.asarray(projection_seed, dtype=np.int64),
                projection_method=np.asarray(projection_method),
                requested_perplexity=np.asarray(perplexity, dtype=np.float64),
                effective_perplexity=np.asarray(effective_perplexity, dtype=np.float64),
                n_components=np.asarray(2, dtype=np.int64),
                random_state=np.asarray(effective_random_state, dtype=np.int64),
                initialization=np.asarray("random"),
                learning_rate=np.asarray("auto"),
                feature_normalization=np.asarray(normalization),
            )
    return output_paths


def _create_projection_tsne_for_directory(
    projected_dir: Path,
    *,
    feature_files: list[Path] | None = None,
    output_dir: Path | None = None,
    perplexity: float,
    random_state: int,
) -> tuple[Path, Path, Path, Path]:
    """Save t-SNE plots and embeddings for raw and per-prompt-normalized features."""
    if (
        not isinstance(perplexity, (int, float))
        or isinstance(perplexity, bool)
        or not math.isfinite(perplexity)
        or perplexity <= 0
    ):
        raise ValueError("t-SNE perplexity must be a positive finite number.")
    if (
        not isinstance(random_state, int)
        or isinstance(random_state, bool)
        or random_state < 0
    ):
        raise ValueError("t-SNE random_state must be a non-negative integer.")
    effective_random_state = random_state % (2**32)
    if effective_random_state != random_state:
        logger.info(
            "Reducing t-SNE random_state from %d to %d for scikit-learn.",
            random_state,
            effective_random_state,
        )

    feature_files = feature_files or sorted(projected_dir.glob("dataset_*.npz"))
    if not feature_files:
        raise FileNotFoundError(
            f"No per-dataset projected feature files found in {projected_dir}."
        )

    feature_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sample_label_parts: list[np.ndarray] = []
    expected_metadata: tuple[int, int, int, str, tuple[int, ...]] | None = None

    for feature_file in feature_files:
        with np.load(feature_file, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            sample_ids = np.asarray(data["sample_ids"], dtype=np.str_)
            labels = np.asarray(data["dataset_labels"], dtype=np.str_)
            # Older archives can still be plotted by dataset.
            sample_labels_json = (
                np.asarray(data["sample_labels_json"], dtype=np.str_)
                if "sample_labels_json" in data
                else np.asarray(["{}"] * features.shape[0], dtype=np.str_)
            )
            source_layers = tuple(int(layer) for layer in data["source_layers"])
            metadata = (
                int(data["original_dim"]),
                int(data["projection_dim"]),
                int(data["projection_seed"]),
                str(data["projection_method"].item()),
                source_layers,
            )

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(f"{feature_file} has no valid 2-D feature rows.")
        expected_width = len(source_layers) * metadata[1] ** 2
        if features.shape[1] != expected_width:
            raise ValueError(
                f"{feature_file} has feature width {features.shape[1]}, "
                f"expected {expected_width}."
            )
        if sample_ids.shape != (features.shape[0],):
            raise ValueError(f"{feature_file} has mismatched prompt identifiers.")
        if labels.shape != (features.shape[0],):
            raise ValueError(f"{feature_file} has mismatched dataset labels.")
        if sample_labels_json.shape != (features.shape[0],):
            raise ValueError(f"{feature_file} has mismatched sample labels.")
        for sample_label_json in sample_labels_json:
            try:
                sample_label = json.loads(str(sample_label_json))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{feature_file} contains invalid sample label JSON."
                ) from exc
            if not isinstance(sample_label, dict):
                raise ValueError(
                    f"{feature_file} sample labels must be JSON objects."
                )
        if not np.isfinite(features).all():
            raise ValueError(f"{feature_file} contains non-finite features.")
        if expected_metadata is None:
            expected_metadata = metadata
        elif metadata != expected_metadata:
            raise ValueError(
                f"Projection metadata in {feature_file} does not match the other "
                "dataset feature files."
            )

        feature_parts.append(features)
        id_parts.append(sample_ids)
        label_parts.append(labels)
        sample_label_parts.append(sample_labels_json)

    features = np.concatenate(feature_parts, axis=0)
    sample_ids = np.concatenate(id_parts)
    dataset_labels = np.concatenate(label_parts)
    sample_labels_json = np.concatenate(sample_label_parts)
    if features.shape[0] < 2:
        raise ValueError("t-SNE requires at least two successfully fitted prompts.")
    if all(np.array_equal(features[0], row) for row in features[1:]):
        raise ValueError("t-SNE requires at least two distinct prompt feature vectors.")

    effective_perplexity = min(perplexity, float(features.shape[0] - 1))
    if effective_perplexity != perplexity:
        logger.info(
            "Reducing t-SNE perplexity from %s to %s for %d prompts.",
            perplexity,
            effective_perplexity,
            features.shape[0],
        )

    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise ImportError(
            "t-SNE requires scikit-learn; install it in the active environment."
        ) from exc

    tsne_options = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "random_state": effective_random_state,
        "init": "random",
        "learning_rate": "auto",
    }
    embedding = TSNE(**tsne_options).fit_transform(features)
    feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = np.divide(
        features,
        feature_norms,
        out=np.zeros_like(features),
        where=feature_norms > 0,
    )
    normalized_embedding = TSNE(**tsne_options).fit_transform(normalized_features)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_dir = output_dir or projected_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / "tsne.png"
    embedding_path = result_dir / "tsne_embedding.npz"
    normalized_plot_path = result_dir / "tsne_normalized.png"
    normalized_embedding_path = result_dir / "tsne_normalized_embedding.npz"

    def save_plot(plot_embedding: np.ndarray, path: Path, title: str) -> None:
        figure, axis = plt.subplots(figsize=(9, 7))
        for label in sorted(set(dataset_labels.tolist())):
            mask = dataset_labels == label
            axis.scatter(
                plot_embedding[mask, 0],
                plot_embedding[mask, 1],
                label=label,
                alpha=0.8,
            )
        axis.set_title(title)
        axis.set_xlabel("t-SNE component 1")
        axis.set_ylabel("t-SNE component 2")
        axis.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.tight_layout()
        figure.savefig(path, dpi=200)
        plt.close(figure)

    save_plot(embedding, plot_path, "t-SNE of per-prompt projected Jacobians")
    save_plot(
        normalized_embedding,
        normalized_plot_path,
        "t-SNE of L2-normalized projected Jacobians",
    )

    if expected_metadata is None:
        raise RuntimeError("No projection metadata was loaded from the feature files.")
    original_dim, projection_dim, projection_seed, projection_method, source_layers = (
        expected_metadata
    )

    def save_embedding(path: Path, values: np.ndarray, normalization: str) -> None:
        np.savez_compressed(
            path,
            embedding=values,
            sample_ids=sample_ids,
            dataset_labels=dataset_labels,
            sample_labels_json=sample_labels_json,
            source_layers=np.asarray(source_layers, dtype=np.int64),
            original_dim=np.asarray(original_dim, dtype=np.int64),
            projection_dim=np.asarray(projection_dim, dtype=np.int64),
            projection_seed=np.asarray(projection_seed, dtype=np.int64),
            projection_method=np.asarray(projection_method),
            requested_perplexity=np.asarray(perplexity, dtype=np.float64),
            effective_perplexity=np.asarray(effective_perplexity, dtype=np.float64),
            n_components=np.asarray(2, dtype=np.int64),
            random_state=np.asarray(effective_random_state, dtype=np.int64),
            initialization=np.asarray("random"),
            learning_rate=np.asarray("auto"),
            feature_normalization=np.asarray(normalization),
        )

    save_embedding(embedding_path, embedding, "none")
    save_embedding(normalized_embedding_path, normalized_embedding, "l2_per_prompt")

    return plot_path, embedding_path, normalized_plot_path, normalized_embedding_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and recolor projected Jacobian t-SNE plots."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plot_label = commands.add_parser(
        "plot-label",
        help="Recolor an existing t-SNE embedding by a scalar sample label.",
    )
    plot_label.add_argument("--embedding", required=True)
    plot_label.add_argument("--label", required=True)

    create = commands.add_parser(
        "create",
        help="Create a t-SNE embedding from projected feature directories.",
    )
    create.add_argument(
        "--projected-dir",
        action="append",
        required=True,
        help="Projected-feature directory; repeat to combine datasets.",
    )
    create.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for combined results; defaults beside inputs.",
    )
    create.add_argument("--perplexity", type=float, default=30.0)
    create.add_argument("--random-state", type=int, default=42)

    create_blocks = commands.add_parser(
        "create-blocks",
        help="Create raw and normalized t-SNE embeddings for each saved block.",
    )
    create_blocks.add_argument("--features", required=True)
    create_blocks.add_argument("--perplexity", type=float, default=30.0)
    create_blocks.add_argument("--random-state", type=int, default=42)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plot-label":
        print(plot_tsne_by_label(args.embedding, args.label))
    elif args.command == "create":
        outputs = create_projection_tsne(
            args.projected_dir,
            output_dir=args.output_dir,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )
        for output_dir, paths in outputs.items():
            print(f"t-SNE outputs for {output_dir}:")
            for path in paths:
                print(f"  {path}")
    elif args.command == "create-blocks":
        outputs = create_block_projection_tsne(
            args.features,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )
        for (start_layer, end_layer), paths in outputs.items():
            print(f"Layers {start_layer}-{end_layer}:")
            for path in paths:
                print(f"  {path}")


if __name__ == "__main__":
    main()
