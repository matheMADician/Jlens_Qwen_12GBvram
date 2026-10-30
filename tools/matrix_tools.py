import math, os
from collections.abc import Mapping
from dataclasses import dataclass, asdict

import torch
from torch import Tensor

from tools.math_tools import MathTools

"""
MatrixTools provides static methods for operations and metrics related to matrices,
including SVD decomposition and Jacobian analysis.
"""

@dataclass
class JacobianMetrics:
    sval: list[float]
    sval_threshold: float # [0, 1] relative threshold for strong singular values.
    strong_rank: int # Number of singular values above the threshold.
    frobenius_norm: float
    operator_norm: float
    effective_rank: float
    
@dataclass
class LayerSVDMetrics:
    sval: Tensor
    left_svec: Tensor
    right_svec: Tensor

class MatrixTools:

    @staticmethod
    def SVD_decomp(matrix: Tensor) -> LayerSVDMetrics:
        """Return the reduced SVD factors ``(U, S, Vh)`` of a 2-D matrix."""
        if matrix.ndim != 2:
            raise ValueError("matrix must be a 2-D tensor.")
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return LayerSVDMetrics(
            sval=S,
            left_svec=U,
            right_svec=Vh.T,
        )

    @staticmethod
    def _validate_jacobian(
        jacobian: Tensor,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Validate a finite square Jacobian and convert it to float32."""
        if not torch.is_tensor(jacobian) or jacobian.ndim != 2:
            raise ValueError("A Jacobian must be a two-dimensional tensor.")
        if jacobian.shape[0] != jacobian.shape[1]:
            raise ValueError(
                f"A Jacobian must be square, got shape {tuple(jacobian.shape)}."
            )
        if jacobian.shape[0] == 0:
            raise ValueError("A Jacobian must not be empty.")
        matrix = jacobian.detach().to(device=device, dtype=torch.float32)
        if not torch.isfinite(matrix).all():
            raise ValueError("A Jacobian contains non-finite values.")
        return matrix

    @staticmethod
    def _validate_sval_threshold(threshold: float) -> None:
        """Require a relative sval-value threshold in the interval (0, 1]."""
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError(
                f"sval_threshold must be in (0, 1], got {threshold!r}."
            )

    @staticmethod
    def get_jacobian_metrics(
        jacobian: Tensor,
        *,
        sval_threshold: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> JacobianMetrics:
        """Return the singular spectrum, strong rank, and norms of a Jacobian."""
        
        # Validate inputs
        MatrixTools._validate_sval_threshold(sval_threshold)
        matrix = MatrixTools._validate_jacobian(jacobian, device=device)
        # Sval and related
        sval = torch.linalg.svdvals(matrix)
        largest_sval = sval[0]
        sval_sum = sval.sum()
        # Frobenius norm
        # 👇 This is just a smart way to calculate the Frobenius norm of the matrix.
        frobenius_norm = torch.linalg.vector_norm(sval) 
		# Large singular value count(strong rank)
        if largest_sval > 0:
            strong_rank = int(
                (
                    sval
                    >= sval_threshold * largest_sval
                ).sum().item()
            )
        else:
            strong_rank = 0
		# Effective rank
			# This is the entropy of the normalized singular values
   			# If all are equal, it's equal to the number of singular values.
			# If one dominates, it approaches 1.
        if sval_sum > 0:
            probabilities = sval / sval_sum
            effective_rank = float(
                torch.exp(
                    -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                ).item()
            )
        else:
            effective_rank = 0.0

        return JacobianMetrics(
            sval=sval.cpu().tolist(),
            strong_rank=strong_rank,
            sval_threshold=sval_threshold,
            frobenius_norm=float(frobenius_norm.item()),
            operator_norm=float(largest_sval.item()),
            effective_rank=effective_rank,
        )

    @staticmethod
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
              between the top-k left-singular subspaces of adjacent layers.
            - ``layers``: Sorted layer indices corresponding to rows of ``spectra``.
            - ``layer_pairs``: Labels for the compared layer pairs (e.g. ``"2->3"``).
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
            raise ValueError(
                f"top_k must be <= the matrix dimension ({first.shape[0]})."
            )

        spectra: list[Tensor] = []
        adjacent_cosines: list[Tensor] = []
        layer_pairs: list[str] = []
        previous_basis: Tensor | None = None
        previous_layer: int | None = None

        for layer in layers:
            matrix = jacobians[layer]
            if matrix.shape != first.shape or matrix.ndim != 2:
                raise ValueError("All layer Jacobians must have the same 2-D shape.")

            svd_metrics = MatrixTools.SVD_decomp(
                MatrixTools._validate_jacobian(matrix, device=device)
            )
            spectra.append(torch.tensor(svd_metrics.sval))
            current_basis = svd_metrics.left_svec[:, :top_k].detach().clone()

            if previous_basis is not None:
                overlap = previous_basis.T @ current_basis
                adjacent_cosines.append(torch.linalg.svdvals(overlap).cpu())
                layer_pairs.append(f"{previous_layer}->{layer}")

            previous_basis = current_basis
            previous_layer = layer

        return torch.stack(spectra), torch.stack(adjacent_cosines), layers, layer_pairs

    @staticmethod
    def draw_svd_heatmaps(
        jacobians: Mapping[int, Tensor],
        output_dir: str,
        *,
        name: str = "jlens",
        top_k: int = 8,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor, list[int], list[str]]:
        """Save singular-spectrum and adjacent-layer subspace heatmaps."""
        spectra, similarities, layers, layer_pairs = MatrixTools.analyze_layer_svd(
            jacobians, top_k=top_k, device=device
        )

        spectrum_path = os.path.join(output_dir, f"{name}_spectrum.png")
        similarity_path = os.path.join(output_dir, f"{name}_adjacent_subspace.png")
        log_spectra = spectra.clamp_min(torch.finfo(spectra.dtype).tiny).log10()
        MathTools.draw_heatmap(
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
        MathTools.draw_heatmap(
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

        return spectra, similarities, layers, layer_pairs
