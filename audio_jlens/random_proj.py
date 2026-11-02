"""Seeded Gaussian projection helpers for Jacobian-lens matrices."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from torch import Tensor
import os
import tempfile

from jacobian_lens.jlens.lens import JacobianLens


PROJECTION_METHOD = "scaled_gaussian"


@dataclass(frozen=True)
class ProjectionMetadata:
    original_dim: int
    projection_dim: int
    projection_seed: int
    projection_method: str = PROJECTION_METHOD

    def as_dict(self) -> dict[str, int | str]:
        """Return metadata in a serialization-friendly form."""
        return {
            "original_dim": self.original_dim,
            "projection_dim": self.projection_dim,
            "projection_seed": self.projection_seed,
            "projection_method": self.projection_method,
        }


class GaussianRandomProjector:
    """Project Jacobians and cache/save prompt features for one configuration."""

    def __init__(
        self,
        d_model: int,
        projection_dim: int,
        seed: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer.")
        if (
            not isinstance(projection_dim, int)
            or isinstance(projection_dim, bool)
            or projection_dim <= 0
        ):
            raise ValueError("projection_dim must be a positive integer.")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("projection seed must be a non-negative integer.")
        if not dtype.is_floating_point:
            raise ValueError("projection dtype must be floating point.")

        self.d_model = d_model
        self.projection_dim = projection_dim
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        projection = torch.randn(
            d_model, projection_dim, generator=generator, dtype=torch.float32
        )
        self.matrix = (projection / math.sqrt(projection_dim)).to(
            device=self.device, dtype=self.dtype
        )
        self._features_by_dataset: dict[str, list[Tensor]] = {}
        self._sample_ids_by_dataset: dict[str, list[str]] = {}

    def project(self, jacobian: Tensor) -> Tensor:
        """Project one square Jacobian as ``P.T @ J @ P``."""
        if not torch.is_tensor(jacobian) or jacobian.ndim != 2:
            raise ValueError("A Jacobian must be a two-dimensional tensor.")
        if tuple(jacobian.shape) != (self.d_model, self.d_model):
            raise ValueError(
                f"Jacobian must have shape {(self.d_model, self.d_model)}, "
                f"got {tuple(jacobian.shape)}."
            )

        matrix = jacobian.detach().to(device=self.device, dtype=self.dtype)
        if not torch.isfinite(matrix).all():
            raise ValueError("A Jacobian contains non-finite values.")

        projected = self.matrix.T @ matrix @ self.matrix
        if not torch.isfinite(projected).all():
            raise ValueError("Projection produced non-finite values.")
        return projected

    def project_lens(self, lens: JacobianLens) -> JacobianLens:
        """Return a projected copy of a fitted lens."""
        if lens.d_model != self.d_model:
            raise ValueError(
                f"Lens d_model={lens.d_model} does not match projector "
                f"d_model={self.d_model}."
            )
        return JacobianLens(
            jacobians={
                layer: self.project(jacobian)
                for layer, jacobian in lens.jacobians.items()
            },
            n_prompts=lens.n_prompts,
            d_model=self.projection_dim,
        )

    @property
    def metadata(self) -> ProjectionMetadata:
        """Describe this projector's reproducible configuration."""
        return ProjectionMetadata(
            original_dim=self.d_model,
            projection_dim=self.projection_dim,
            projection_seed=self.seed,
        )

    @property
    def name(self) -> str:
        """Return a stable folder name for this dimension/seed configuration."""
        return f"dim_{self.projection_dim}_seed_{self.seed}"

    def cache_prompt(
        self,
        jacobians: dict[int, Tensor],
        source_layers: Sequence[int],
        *,
        sample_id: str,
        dataset_name: str,
        prompt_index: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Project and cache one prompt's concatenated layer features."""
        if not source_layers:
            raise ValueError("source_layers must not be empty.")
        missing_layers = set(source_layers) - set(jacobians)
        if missing_layers:
            raise ValueError(
                f"Prompt Jacobians are missing source layers: {sorted(missing_layers)}."
            )
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string.")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("dataset_name must be a non-empty string.")

        features = torch.cat(
            [self.project(jacobians[layer]).reshape(-1) for layer in source_layers]
        ).cpu()
        if cache_dir is not None:
            if (
                not isinstance(prompt_index, int)
                or isinstance(prompt_index, bool)
                or prompt_index < 0
            ):
                raise ValueError(
                    "prompt_index must be a non-negative integer for disk caching."
                )
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            destination = cache_path / f"prompt_{prompt_index:08d}.npz"
            with tempfile.NamedTemporaryFile(
                dir=cache_path, prefix=".feature-", suffix=".tmp", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                try:
                    np.savez_compressed(
                        temporary_file,
                        features=features.numpy(),
                        sample_id=np.asarray(sample_id),
                        dataset_name=np.asarray(dataset_name),
                    )
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                    os.replace(temporary_path, destination)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()
            return

        self._features_by_dataset.setdefault(dataset_name, []).append(features)
        self._sample_ids_by_dataset.setdefault(dataset_name, []).append(sample_id)

    def projected_directory(self, projected_root: str | Path) -> Path:
        """Return this projector's artifact directory under the shared root."""
        return Path(projected_root) / self.name

    def feature_cache_directory(self, projected_root: str | Path) -> Path:
        """Return the directory holding durable per-prompt feature shards."""
        return self.projected_directory(projected_root) / ".feature_cache"

    def prepare_feature_cache_for_resume(
        self,
        projected_root: str | Path,
        *,
        next_prompt_index: int,
        expected_count: int,
    ) -> None:
        """Discard uncheckpointed shards and verify the completed prompt cache."""
        cache_dir = self.feature_cache_directory(projected_root)
        if not cache_dir.exists():
            if expected_count:
                raise FileNotFoundError(
                    f"Projected prompt cache is missing for {self.name}; "
                    "use resume=False to refit and rebuild it."
                )
            return

        cached_paths = sorted(cache_dir.glob("prompt_*.npz"))
        for path in cached_paths:
            try:
                prompt_index = int(path.stem.removeprefix("prompt_"))
            except ValueError as exc:
                raise ValueError(f"Invalid projected cache shard name: {path}") from exc
            if prompt_index >= next_prompt_index:
                path.unlink()

        remaining = list(cache_dir.glob("prompt_*.npz"))
        if len(remaining) != expected_count:
            raise RuntimeError(
                f"Projected cache for {self.name} contains {len(remaining)} "
                f"checkpointed prompts; expected {expected_count}. Use resume=False "
                "to refit and rebuild projected features."
            )

    def save_lens(self, lens: JacobianLens, path: str | Path) -> Path:
        """Project and save a cumulative or combined lens."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.project_lens(lens).save(str(path))
        return path

    def save_projection_matrix(
        self,
        projected_root: str | Path,
        *,
        source_layers: Sequence[int],
    ) -> Path:
        """Save the reproducible projection matrix and its metadata."""
        output_dir = self.projected_directory(projected_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "projection_matrix.npz"
        np.savez_compressed(
            path,
            matrix=self.matrix.detach().cpu().numpy(),
            source_layers=np.asarray(source_layers, dtype=np.int64),
            original_dim=np.asarray(self.d_model, dtype=np.int64),
            projection_dim=np.asarray(self.projection_dim, dtype=np.int64),
            projection_seed=np.asarray(self.seed, dtype=np.int64),
            projection_method=np.asarray(PROJECTION_METHOD),
        )
        return path

    def save_features(
        self,
        projected_root: str | Path,
        *,
        source_layers: Sequence[int],
        dataset_counts: dict[str, int],
        use_disk_cache: bool = False,
    ) -> Path:
        """Save the projection matrix and cached per-dataset features as NPZ files."""
        output_dir = self.projected_directory(projected_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.save_projection_matrix(projected_root, source_layers=source_layers)
        common_metadata = {
            "source_layers": np.asarray(source_layers, dtype=np.int64),
            "original_dim": np.asarray(self.d_model, dtype=np.int64),
            "projection_dim": np.asarray(self.projection_dim, dtype=np.int64),
            "projection_seed": np.asarray(self.seed, dtype=np.int64),
            "projection_method": np.asarray(PROJECTION_METHOD),
        }
        cached: dict[str, list[tuple[int, Tensor, str]]] = {}
        if use_disk_cache:
            cache_dir = self.feature_cache_directory(projected_root)
            if not cache_dir.is_dir():
                raise FileNotFoundError(
                    f"Projected prompt cache is missing for {self.name}: {cache_dir}"
                )
            for path in sorted(cache_dir.glob("prompt_*.npz")):
                prompt_index = int(path.stem.removeprefix("prompt_"))
                with np.load(path, allow_pickle=False) as data:
                    features = torch.from_numpy(
                        np.asarray(data["features"], dtype=np.float32)
                    )
                    sample_id = str(data["sample_id"].item())
                    dataset_name = str(data["dataset_name"].item())
                cached.setdefault(dataset_name, []).append(
                    (prompt_index, features, sample_id)
                )
        else:
            for dataset_name, features in self._features_by_dataset.items():
                sample_ids = self._sample_ids_by_dataset[dataset_name]
                cached[dataset_name] = [
                    (index, feature, sample_id)
                    for index, (feature, sample_id) in enumerate(
                        zip(features, sample_ids)
                    )
                ]

        if set(cached) != set(dataset_counts):
            raise RuntimeError(
                "Projected feature datasets do not match successfully fitted datasets."
            )
        for dataset_name, prompt_features in cached.items():
            prompt_features.sort(key=lambda row: row[0])
            features = [row[1] for row in prompt_features]
            sample_ids = [row[2] for row in prompt_features]
            count = dataset_counts[dataset_name]
            if len(features) != count or len(sample_ids) != count:
                raise RuntimeError(
                    f"Projected cache for {dataset_name!r} has {len(features)} "
                    f"features and {len(sample_ids)} IDs; expected {count}."
                )
            filename = f"dataset_{quote(dataset_name, safe='._-')}.npz"
            with tempfile.NamedTemporaryFile(
                dir=output_dir, prefix=".dataset-", suffix=".tmp", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                try:
                    np.savez_compressed(
                        temporary_file,
                        features=torch.stack(features).numpy(),
                        sample_ids=np.asarray(sample_ids, dtype=np.str_),
                        dataset_labels=np.asarray([dataset_name] * count, dtype=np.str_),
                        dataset_name=np.asarray(dataset_name),
                        **common_metadata,
                    )
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                    os.replace(temporary_path, output_dir / filename)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()

        if not use_disk_cache:
            self.clear_cache()
        return output_dir

    def clear_disk_cache(self, projected_root: str | Path) -> None:
        """Remove durable per-prompt shards after final dataset files are saved."""
        cache_dir = self.feature_cache_directory(projected_root)
        if cache_dir.exists():
            for path in cache_dir.iterdir():
                if path.is_file():
                    path.unlink()
            cache_dir.rmdir()

    def clear_cache(self) -> None:
        """Release per-prompt features after their files have been saved."""
        self._features_by_dataset.clear()
        self._sample_ids_by_dataset.clear()


def delete_saved_prompt_features(projected_dir: str | Path) -> None:
    """Delete large per-prompt feature archives after t-SNE has succeeded."""
    for path in Path(projected_dir).glob("dataset_*.npz"):
        path.unlink()


def _projection_metadata_for_lens(path: str | Path) -> dict[str, int | str] | None:
    """Load projection metadata associated with a projected lens artifact."""
    lens_path = Path(path).resolve()
    projected_parent = next(
        (parent for parent in lens_path.parents if parent.name == "projected"),
        None,
    )
    if projected_parent is None:
        return None

    metadata_path = projected_parent / "projection_matrix.npz"
    if not metadata_path.is_file():
        metadata_path = next(
            (
                parent / "projection_matrix.npz"
                for parent in lens_path.parents
                if parent != projected_parent
                and parent.is_dir()
                and (parent / "projection_matrix.npz").is_file()
            ),
            None,
        )
    if metadata_path is None:
        raise FileNotFoundError(
            f"Projected lens metadata is missing under {projected_parent}"
        )
    with np.load(metadata_path, allow_pickle=False) as data:
        matrix = np.ascontiguousarray(data["matrix"])
        return {
            "original_dim": int(data["original_dim"]),
            "projection_dim": int(data["projection_dim"]),
            "projection_seed": int(data["projection_seed"]),
            "projection_method": str(data["projection_method"].item()),
            "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        }


def validate_projection_compatibility(
    lens_path1: str | Path,
    lens_path2: str | Path,
) -> None:
    """Reject comparisons between projected lenses made with different matrices."""
    metadata1 = _projection_metadata_for_lens(lens_path1)
    metadata2 = _projection_metadata_for_lens(lens_path2)
    if metadata1 is None and metadata2 is None:
        return
    if metadata1 is None or metadata2 is None:
        raise ValueError(
            "Cannot compare a projected JLens with a lens lacking matching "
            "projection metadata."
        )
    if metadata1 != metadata2:
        raise ValueError(
            "Cannot compare projected JLens artifacts created with incompatible "
            f"projection matrices: {metadata1!r} != {metadata2!r}."
        )
