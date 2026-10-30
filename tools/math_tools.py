from collections.abc import Mapping, Sequence, Callable
import os

import torch
from torch import Tensor, empty, stack

"""
MathTools provides 
    1. math methods that doesn't involve matrix operations.
    2. Drawing methods
"""

class MathTools:
    
    @staticmethod
    def cos_similarity(v1: Tensor, v2: Tensor) -> float:
        """Compute pairwise cosine similarity between two Tensors.

        Args:
            v1, v2 (Tensor): Hidden layers for comparison. Dimensions must match.

        Returns:
            _type_: _description_
        """
        if v1.shape != v2.shape:
            raise ValueError(f"Shape of vectors don't match, v1{v1.shape}, v2{v2.shape}")

        v1_flat = v1.reshape(-1)
        v2_flat = v2.reshape(-1)
        v1_norm = v1_flat.norm()
        v2_norm = v2_flat.norm()
        if v1_norm == 0 or v2_norm == 0:
            raise ValueError("Cosine similarity is undefined for a zero vector")
        return (v1_flat @ v2_flat / (v1_norm * v2_norm)).item()

    @staticmethod
    def cka_similarity(
        l1: dict[int, Tensor],
        l2: dict[int, Tensor],
        d1: Sequence[int],
        d2: Sequence[int],
        device: str | torch.device | None = None,
    ) -> Tensor:
        """
        Compute pairwise linear CKA between two sets of representations.

        The first dimension of every representation must index the same examples.
        Remaining dimensions are treated as features, so both vectors and matrices
        are accepted. The returned tensor is indexed as ``[d1_layer, d2_layer]``.
        Computation defaults to CUDA when available, otherwise CPU; the returned
        heatmap is on CPU.

        Args:
            l1, l2: Layer-to-representation mappings.
            d1, d2: Layer indices to compare from ``l1`` and ``l2``.
            device: Device for the CKA calculation. Defaults to CUDA when
                available, otherwise CPU.
        """
        compute_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        def prepare(representations: dict[int, Tensor], layers: Sequence[int]):
            prepared = {}
            covariance_norms = {}
            for layer in layers:
                values = representations[layer]
                if values.ndim == 0:
                    raise ValueError(
                        "CKA representations must have a sample dimension"
                    )
                values = values.reshape(values.shape[0], -1).to(
                    device=compute_device, dtype=torch.float32
                )
                values = values - values.mean(dim=0, keepdim=True)
                prepared[layer] = values
                covariance_norm = torch.linalg.matrix_norm(values.T @ values)
                if covariance_norm == 0:
                    raise ValueError(
                        "CKA is undefined for a constant representation"
                    )
                covariance_norms[layer] = covariance_norm
            return prepared, covariance_norms

        if not d1 or not d2:
            return torch.empty((len(d1), len(d2)))

        prepared1, norms1 = prepare(l1, d1)
        if l1 is l2 and d1 == d2:
            prepared2, norms2 = prepared1, norms1
        else:
            prepared2, norms2 = prepare(l2, d2)

        def linear_cka(layer_1: int, layer_2: int) -> Tensor:
            x = prepared1[layer_1]
            y = prepared2[layer_2]
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    "CKA representations must contain the same number of samples"
                )

            denominator = norms1[layer_1] * norms2[layer_2]
            return torch.linalg.matrix_norm(x.T @ y).square() / denominator

        values = [
            [linear_cka(layer_1, layer_2) for layer_2 in d2]
            for layer_1 in d1
        ]
        return torch.stack([torch.stack(row) for row in values]).detach().cpu()
    
    @staticmethod
    def sim_for_sets(
        l1: dict[int, Tensor],
        l2: dict[int, Tensor],
        d1: Sequence[int],
        d2: Sequence[int],
        func: Callable[[Tensor, Tensor], float],
    ) -> Tensor:
        """
        Compute pairwise similarity between two sets of representations.
        
        The first dimension of every representation must index the same examples.
        Remaining dimensions are treated as features, so both vectors and matrices
        are accepted. The returned tensor is indexed as ``[d1_layer, d2_layer]``.

        Args:
            l1, l2: Layer-to-representation mappings.
            d1, d2: Layer indices to compare from ``l1`` and ``l2``.
        """
        if not d1 or not d2:
            return empty((len(d1), len(d2)), dtype=Tensor().dtype)

        values = [
            [
                func(l1[layer_1], l2[layer_2])
                for layer_2 in d2
            ]
            for layer_1 in d1
        ]
        return stack([Tensor(row) for row in values])

    @staticmethod
    def draw_heatmap(
        data: Tensor,
        png_path: str,
        labels: Sequence[int | str] | None = None,
        x_labels: Sequence[int | str] | None = None,
        y_labels: Sequence[int | str] | None = None,
        title: str = "Layer cosine similarity",
        x_axis_label: str = "Layer",
        y_axis_label: str = "Layer",
        colorbar_label: str = "Cosine similarity",
        cmap: str = "coolwarm",
        vmin: float | None = -1.0,
        vmax: float | None = 1.0,
    ) -> None:
        """Save a two-dimensional tensor as a heatmap image.

        Args:
            data: The heatmap data. It must have exactly two dimensions.
            png_path: Path to store the heatmap image.
            labels: Optional labels for both axes when they are identical.
            x_labels: Optional labels for the horizontal axis.
            y_labels: Optional labels for the vertical axis.
            title: Figure title.
            x_axis_label: Horizontal axis label.
            y_axis_label: Vertical axis label.
            colorbar_label: Label for the color scale.
            cmap: Matplotlib colormap name.
            vmin, vmax: Optional color scale limits.
        """
        if data.ndim != 2:
            raise ValueError(f"Heatmap data must be 2-D, got shape {tuple(data.shape)}")
        if labels is not None:
            if x_labels is not None or y_labels is not None:
                raise ValueError("Use labels or x_labels/y_labels, not both")
            x_labels = labels
            y_labels = labels
        if x_labels is not None and len(x_labels) != data.shape[1]:
            raise ValueError("x_labels must match the number of heatmap columns")
        if y_labels is not None and len(y_labels) != data.shape[0]:
            raise ValueError("y_labels must match the number of heatmap rows")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(
            data.detach().cpu().numpy(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel(x_axis_label)
        axis.set_ylabel(y_axis_label)
        if x_labels is not None:
            axis.set_xticks(range(len(x_labels)), labels=[str(label) for label in x_labels])
        if y_labels is not None:
            axis.set_yticks(range(len(y_labels)), labels=[str(label) for label in y_labels])
        figure.colorbar(image, ax=axis, label=colorbar_label)
        figure.tight_layout()
        figure.savefig(png_path, dpi=200)
        plt.close(figure)
        
    @staticmethod
    def draw_line_graph(
        data: Mapping[str, Sequence[float] | Tensor],
        png_path: str,
        x_values: Sequence[float] | Tensor | None = None,
        title: str = "Line graph",
        x_axis_label: str = "X",
        y_axis_label: str = "Value",
    ) -> None:
        """Save one or more named series as a line graph.

        Args:
            data: Mapping of legend labels to one-dimensional y-value sequences.
            png_path: Path where the plot image will be saved.
            x_values: Optional shared x-values. Defaults to integer positions.
            title: Figure title.
            x_axis_label: Horizontal axis label.
            y_axis_label: Vertical axis label.
        """
        if not data:
            raise ValueError("Line graph data must contain at least one series.")

        series: dict[str, Tensor] = {}
        expected_length: int | None = None
        for label, values in data.items():
            if not isinstance(label, str) or not label:
                raise ValueError("Every line graph series must have a non-empty label.")
            try:
                tensor = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Line graph series {label!r} must contain numeric values."
                ) from exc
            if tensor.ndim != 1 or tensor.numel() == 0:
                raise ValueError(
                    f"Line graph series {label!r} must be a non-empty 1-D sequence."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Line graph series {label!r} contains non-finite values."
                )
            if expected_length is None:
                expected_length = tensor.numel()
            elif tensor.numel() != expected_length:
                raise ValueError("All line graph series must have the same length.")
            series[label] = tensor

        assert expected_length is not None
        if x_values is None:
            x_tensor = torch.arange(expected_length, dtype=torch.float32)
        else:
            try:
                x_tensor = torch.as_tensor(x_values, dtype=torch.float32).detach().cpu()
            except (TypeError, ValueError) as exc:
                raise ValueError("Line graph x_values must be numeric.") from exc
            if x_tensor.ndim != 1 or x_tensor.numel() != expected_length:
                raise ValueError(
                    "x_values must be a 1-D sequence matching each series length."
                )
            if not torch.isfinite(x_tensor).all():
                raise ValueError("Line graph x_values contain non-finite values.")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        figure, axis = plt.subplots(figsize=(8, 6))
        x = x_tensor.numpy()
        for label, values in series.items():
            axis.plot(x, values.numpy(), marker="o", label=label)
        axis.set_title(title)
        axis.set_xlabel(x_axis_label)
        axis.set_ylabel(y_axis_label)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(png_path, dpi=200)
        plt.close(figure)
