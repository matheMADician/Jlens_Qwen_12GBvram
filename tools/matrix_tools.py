from collections.abc import Mapping
import os

import torch
from torch import Tensor


def SVD_decomp(M: Tensor) -> tuple[Tensor, Tensor, Tensor]:
	"""Return the reduced SVD factors ``(U, S, Vh)`` of a 2-D matrix."""
	if M.ndim != 2:
		raise ValueError("M must be a 2-D matrix.")
	return torch.linalg.svd(M, full_matrices=False)


def analyze_layer_svd(
	jacobians: Mapping[int, Tensor],
	*,
	top_k: int = 8,
	device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, list[int], list[str]]:
	"""Compute each layer's singular spectrum and adjacent top-k subspace similarity.

    Args:
		jacobians: Mapping from layer index to its square 2-D Jacobian matrix.
			All matrices must have the same shape.
		top_k: Number of leading left singular vectors used to define each
			layer's subspace. Must be between 1 and the matrix dimension.
		device: Device used for the per-layer SVD. The returned summaries are
			moved to CPU.

    Returns:
		A tuple ``(spectra, adjacent_cosines, layers, layer_pairs)``:

		- ``spectra``: Tensor of shape ``[n_layers, d_model]`` containing each
		  layer's singular values in descending order.
		- ``adjacent_cosines``: Tensor of shape
		  ``[n_layers - 1, top_k]``. Each row contains the principal cosines
		  between the top-k left-singular subspaces of a pair of neighboring
		  sorted layers. Values near 1 indicate aligned subspaces; values near
		  0 indicate orthogonal directions. The values are ordered from largest
		  to smallest and are invariant to singular-vector sign flips.
		- ``layers``: Sorted layer indices corresponding to rows of ``spectra``.
		- ``layer_pairs``: Labels for the compared layer pairs, in the same row
		  order as ``adjacent_cosines`` (for example, ``"2->3"``).

	Raises:
		ValueError: If fewer than two layers are provided, a Jacobian is not a
			square 2-D matrix, the Jacobians have inconsistent shapes, or
			``top_k`` is outside the valid range.
    """
	layers = sorted(jacobians)
	if len(layers) < 2:
		raise ValueError("At least two layer Jacobians are required.")
	if top_k < 1:
		raise ValueError("top_k must be positive.")

	first = jacobians[layers[0]]
	if first.ndim != 2 or first.shape[0] != first.shape[1]:
		raise ValueError("Each Jacobian must be a square 2-D matrix.")
	if top_k > first.shape[0]:
		raise ValueError(f"top_k must be <= the matrix dimension ({first.shape[0]}).")

	spectra: list[Tensor] = []
	adjacent_cosines: list[Tensor] = []
	layer_pairs: list[str] = []
	previous_basis: Tensor | None = None
	previous_layer: int | None = None

	for layer in layers:
		matrix = jacobians[layer]
		if matrix.shape != first.shape or matrix.ndim != 2:
			raise ValueError("All layer Jacobians must have the same 2-D shape.")

		left, singular_values, right = SVD_decomp(
			matrix.detach().to(device=device, dtype=torch.float32)
		)
		spectra.append(singular_values.cpu())
		current_basis = left[:, :top_k].detach().clone()

		if previous_basis is not None:
			overlap = previous_basis.T @ current_basis
			adjacent_cosines.append(torch.linalg.svdvals(overlap).cpu())
			layer_pairs.append(f"{previous_layer}->{layer}")

		del left, singular_values, right
		previous_basis = current_basis
		previous_layer = layer

	return torch.stack(spectra), torch.stack(adjacent_cosines), layers, layer_pairs


def draw_svd_heatmaps(
	jacobians: Mapping[int, Tensor],
	output_dir: str,
	*,
	name: str = "jlens",
	top_k: int = 8,
	device: torch.device | str = "cpu",
) -> tuple[str, str]:
	"""Save singular-spectrum and adjacent-layer subspace heatmaps."""
	spectra, similarities, layers, layer_pairs = analyze_layer_svd(
		jacobians, top_k=top_k, device=device
	)
	from tools.math_tools import draw_heatmap

	spectrum_path = os.path.join(output_dir, f"{name}_spectrum.png")
	similarity_path = os.path.join(output_dir, f"{name}_adjacent_subspace.png")

	log_spectra = spectra.clamp_min(torch.finfo(spectra.dtype).tiny).log10()
	draw_heatmap(
		data=log_spectra,
		png_path=spectrum_path,
		x_labels=list(range(log_spectra.shape[1])),
		y_labels=layers,
		title="Jacobian singular-value spectra",
		x_axis_label="Singular value index",
		y_axis_label="Layer",
		colorbar_label="log10(singular value)",
		cmap="magma",
		vmin=None,
		vmax=None,
	)
	draw_heatmap(
		data=similarities,
		png_path=similarity_path,
		x_labels=list(range(1, top_k + 1)),
		y_labels=layer_pairs,
		title=f"Adjacent-layer top-{top_k} left-singular subspaces",
		x_axis_label="Principal cosine rank",
		y_axis_label="Adjacent layer pair",
		colorbar_label="Principal cosine",
		cmap="viridis",
		vmin=0.0,
		vmax=1.0,
	)

	return spectrum_path, similarity_path
