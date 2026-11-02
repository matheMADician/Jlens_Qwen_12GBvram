"""Build a t-SNE visualization from saved per-prompt projected Jacobians."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def create_projection_tsne(
    projected_dir: str | Path,
    *,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """Load per-dataset feature files and save a labeled t-SNE plot and embedding."""
    projected_dir = Path(projected_dir)
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

    feature_files = sorted(projected_dir.glob("dataset_*.npz"))
    if not feature_files:
        raise FileNotFoundError(
            f"No per-dataset projected feature files found in {projected_dir}."
        )

    feature_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    expected_metadata: tuple[int, int, int, str, tuple[int, ...]] | None = None

    for feature_file in feature_files:
        with np.load(feature_file, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            sample_ids = np.asarray(data["sample_ids"], dtype=np.str_)
            labels = np.asarray(data["dataset_labels"], dtype=np.str_)
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

    features = np.concatenate(feature_parts, axis=0)
    sample_ids = np.concatenate(id_parts)
    dataset_labels = np.concatenate(label_parts)
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

    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=effective_random_state,
        init="random",
        learning_rate="auto",
    ).fit_transform(features)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projected_dir.mkdir(parents=True, exist_ok=True)
    plot_path = projected_dir / "tsne.png"
    embedding_path = projected_dir / "tsne_embedding.npz"

    figure, axis = plt.subplots(figsize=(9, 7))
    for label in sorted(set(dataset_labels.tolist())):
        mask = dataset_labels == label
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=label,
            alpha=0.8,
        )
    axis.set_title("t-SNE of per-prompt projected Jacobians")
    axis.set_xlabel("t-SNE component 1")
    axis.set_ylabel("t-SNE component 2")
    axis.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    if expected_metadata is None:
        raise RuntimeError("No projection metadata was loaded from the feature files.")
    original_dim, projection_dim, projection_seed, projection_method, source_layers = (
        expected_metadata
    )
    np.savez_compressed(
        embedding_path,
        embedding=embedding,
        sample_ids=sample_ids,
        dataset_labels=dataset_labels,
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
    )

    return plot_path, embedding_path
