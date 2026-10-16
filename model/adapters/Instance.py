import os
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn, Tensor

from jlens.protocol import LensModel


ABLATION_MODES = ("ablation", "jablation", "injection")


class AblationSpec:
    def __init__(
        self,
        mode: str,
        layers: list[int],
        token_pos: list[int],
        directions: dict[int, Tensor] | None = None,
        beta: float = 1.0,
        directions2: dict[int, Tensor] | None = None,
    ):
        if mode not in ABLATION_MODES:
            raise ValueError(
                f"Unknown ablation mode {mode!r}; expected one of {ABLATION_MODES}"
            )
        if mode != "ablation" and directions is None:
            raise ValueError(f"mode={mode!r} requires directions.")
        if mode == "injection" and directions2 is None:
            raise ValueError("mode='injection' requires directions2.")
        for name, table in (("directions", directions), ("directions2", directions2)):
            if table is not None:
                missing = sorted(set(layers) - set(table))
                if missing:
                    raise ValueError(f"{name} is missing layers {missing}.")
        self.mode = mode
        self.layers = list(layers)
        self.token_pos = list(token_pos)
        self.directions = directions
        self.beta = beta
        self.directions2 = directions2


class Instance(LensModel):
    """Shared interface and utilities for model-specific JLens adapters."""

    def __init__(self, MODEL_ID: str, do_4bit: bool = False):
        self.data_root: str | None = None
        self.hf_model: nn.Module | None = None
        self.processor: Any | None = None
        self.layers: Sequence[nn.Module] = []
        self.n_layers = 0
        self.d_model = 0
        self.tokenizer: Any | None = None
        self._initialize_model(MODEL_ID, do_4bit)

    def _initialize_model(self, _model_id: str, _do_4bit: bool) -> None:
        raise NotImplementedError("Subclasses must initialize their model adapter.")

    def get_model(self) -> tuple[nn.Module, Any]:
        if self.hf_model is None or self.processor is None:
            raise RuntimeError("The model adapter has not been initialized.")
        return self.hf_model, self.processor

    def load_data(self, data_root: str) -> None:
        """Only sets the root directory of the data for the model. Doesn't actually parse or load the data."""
        if not data_root:
            raise ValueError("data_root must not be empty")
        resolved_root = os.path.abspath(os.path.expanduser(data_root))
        if not os.path.isdir(resolved_root):
            raise FileNotFoundError(f"Data root does not exist: {resolved_root}")
        self.data_root = resolved_root

    def clear_model_input_cache(self) -> None:
        raise NotImplementedError("Subclasses must clear their cached model inputs.")

    def encode(self, text: str, *, max_length: int= 3000) -> Tensor:
        """Encodes the input text into a tensor suitable for the model.
        For multi-model purposes, text should be the raw jsonl line.
        """
        raise NotImplementedError("Subclasses must implement prompt encoding.")

    def forward(self, input_ids: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the model forward pass.")

    def unembed(self, residual: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the output readout.")

    def get_unembedding_direction(self, token_id: int) -> Tensor:
        raise NotImplementedError("Subclasses must implement token directions for ablation.")

    def ablation_forward(
        self,
        input_ids: Tensor,
        ablation_spec: AblationSpec | None = None,
    ) -> Tensor:
        if not self.layers:
            raise RuntimeError("The model adapter has no decoder layers.")
        final_layer = self.n_layers - 1
        captured: dict[str, Tensor] = {}
        handles = []

        def capture_final_residual(_module, _inputs, output):
            captured["residual"] = (
                output if torch.is_tensor(output) else output[0]
            ).detach()

        try:
            if ablation_spec is not None:
                for layer in ablation_spec.layers:
                    if not 0 <= layer < self.n_layers:
                        raise ValueError(
                            f"layer {layer} out of range for a {self.n_layers}-layer model"
                        )
                    handles.append(
                        self.layers[layer].register_forward_hook(
                            self._make_ablation_hook(layer, ablation_spec)
                        )
                    )
            handles.append(
                self.layers[final_layer].register_forward_hook(capture_final_residual)
            )
            with torch.no_grad():
                self.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        if "residual" not in captured:
            raise RuntimeError("The final decoder residual was not captured.")
        return captured["residual"]

    @staticmethod
    def _make_ablation_hook(layer: int, spec: AblationSpec):
        def hook(_module, _inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            positions = [position % hidden.shape[1] for position in spec.token_pos]
            selected = hidden[:, positions, :].float()

            if spec.mode == "ablation":
                updated = torch.zeros_like(selected)
            else:
                if spec.directions is None:
                    raise RuntimeError("Ablation directions are missing.")
                direction = spec.directions[layer].to(
                    selected.device, dtype=torch.float32
                )
                coefficient = (selected @ direction).unsqueeze(-1)
                if spec.mode == "jablation":
                    updated = selected - spec.beta * coefficient * direction
                else:
                    if spec.directions2 is None:
                        raise RuntimeError("Injection directions are missing.")
                    direction2 = spec.directions2[layer].to(
                        selected.device, dtype=torch.float32
                    )
                    updated = selected - coefficient * direction + coefficient * direction2

            patched = hidden.clone()
            patched[:, positions, :] = updated.to(hidden.dtype)
            return patched if torch.is_tensor(output) else (patched, *output[1:])

        return hook
