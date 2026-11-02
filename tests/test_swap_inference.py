"""Tests for Model.run_swap_inference on a fake model. No weights or GPU are needed.

transformers and librosa are stubbed when missing, but jacobian-lens must be
importable as both ``jlens`` and ``jacobian_lens.jlens``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

pytest.importorskip("jlens")
pytest.importorskip("jacobian_lens.jlens.lens")

from audio_jlens.jlens import Jlens
from model.adapters.Instance import Instance
from model.model import Model
from tools.matrix_tools import MatrixTools

D_MODEL = 8
VOCAB = 16
SEQ_LEN = 6
N_LAYERS = 4
SOURCE, TARGET = 3, 5


class FakeLayer(nn.Module):
    """Scales each position independently, so an untouched position stays predictable."""

    def __init__(self, scale: float, returns_tuple: bool):
        super().__init__()
        self.scale = scale
        self.returns_tuple = returns_tuple

    def forward(self, hidden):
        output = hidden * self.scale
        return (output, "extra") if self.returns_tuple else output


class FakeInstance(Instance):
    def __init__(self, *, returns_tuple: bool = False, dtype: torch.dtype = torch.float32):
        self.returns_tuple = returns_tuple
        self.dtype = dtype
        super().__init__("fake")

    def _initialize_model(self, _model_id: str, _do_4bit: bool) -> None:
        generator = torch.Generator().manual_seed(0)
        self.layer_modules = nn.ModuleList(
            FakeLayer(1.0 + i / 10, self.returns_tuple) for i in range(N_LAYERS)
        )
        self.layers = list(self.layer_modules)
        self.n_layers = N_LAYERS
        self.d_model = D_MODEL
        self.embedded = torch.randn(1, SEQ_LEN, D_MODEL, generator=generator)
        self.unembedding = torch.randn(VOCAB, D_MODEL, generator=generator)

    def encode(self, text: str, *, max_length: int = 3000) -> torch.Tensor:
        return torch.arange(SEQ_LEN).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Same capture pattern as the real adapters.
        captured = {}

        def capture(_module, _args, output):
            captured["residual"] = output if torch.is_tensor(output) else output[0]

        handle = self.layers[-1].register_forward_hook(capture)
        try:
            hidden = self.embedded.to(self.dtype)
            for layer in self.layers:
                output = layer(hidden)
                hidden = output if torch.is_tensor(output) else output[0]
        finally:
            handle.remove()
        return captured["residual"]

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.float() @ self.unembedding.T

    def get_unembedding_direction(self, token_id: int) -> torch.Tensor:
        return self.unembedding[token_id]


def _model(instance: FakeInstance | None = None, *, with_lens: bool = True) -> Model:
    logger = logging.getLogger("test")
    model = Model(logger, "Qwen/Qwen2-Audio-7B", "Qwen/Qwen2-Audio-7B")
    model._model = instance or FakeInstance()
    if with_lens:
        generator = torch.Generator().manual_seed(1)
        lens = Jlens(logger, model._model)
        lens.lens = SimpleNamespace(
            jacobians={
                layer: torch.randn(D_MODEL, D_MODEL, generator=generator)
                for layer in range(N_LAYERS - 1)
            },
            source_layers=list(range(N_LAYERS - 1)),
            d_model=D_MODEL,
        )
        model.lens = lens
    return model


def _run(model: Model, tmp_path, **overrides):
    arguments = {
        "jsonl_line": "{}",
        "data_root": str(tmp_path),
        "source_token": SOURCE,
        "target_token": TARGET,
        "layers": [1, 2],
    }
    return model.run_swap_inference(**{**arguments, **overrides})


def _expected_final(instance: FakeInstance, directions, positions, alpha=1.0, clamp=True):
    """Recompute the intervention by hand, one layer at a time."""
    clean = instance.embedded.clone()
    hidden = instance.embedded.clone()
    for index, layer in enumerate(instance.layers):
        output = layer(hidden)
        hidden = output if torch.is_tensor(output) else output[0]
        clean = clean * layer.scale
        if index in directions:
            source, target = directions[index]
            hidden = hidden.clone()
            hidden[:, positions, :] = MatrixTools.swap_residual(
                hidden[:, positions, :],
                source,
                target,
                alpha=alpha,
                reference=clean[:, positions, :] if clamp else None,
            )
    return hidden


def _jlens_directions(model: Model, layers):
    source = model._token_directions(SOURCE, layers)
    target = model._token_directions(TARGET, layers)
    return {layer: (source[layer], target[layer]) for layer in layers}


def test_swap_matches_a_manual_layer_by_layer_computation(tmp_path):
    instance = FakeInstance()
    model = _model(instance)
    result = _run(model, tmp_path, alpha=2.0)

    final = _expected_final(
        instance, _jlens_directions(model, [1, 2]), list(range(SEQ_LEN)), alpha=2.0
    )
    assert torch.allclose(result.swapped_logits, instance.unembed(final[0][[-1]]))
    assert not torch.allclose(result.swapped_logits, result.baseline_logits)
    assert result.baseline_logits.shape == result.swapped_logits.shape == (1, VOCAB)
    assert result.input_ids.shape == (1, SEQ_LEN)


def test_unclamped_swap_uses_each_layers_current_values(tmp_path):
    instance = FakeInstance()
    model = _model(instance)
    result = _run(model, tmp_path, clamp_to_clean=False)
    clamped = _run(model, tmp_path)

    final = _expected_final(
        instance, _jlens_directions(model, [1, 2]), list(range(SEQ_LEN)), clamp=False
    )
    assert torch.allclose(result.swapped_logits, instance.unembed(final[0][[-1]]))
    assert not torch.allclose(result.swapped_logits, clamped.swapped_logits)


def test_clamped_swap_survives_consecutive_layers_with_a_shared_direction(tmp_path):
    instance = FakeInstance()
    model = _model(instance, with_lens=False)
    # The logit directions are shared across layers, as residual connections would carry them.
    clamped = _run(model, tmp_path, direction_source="logit", layers=[1, 2])
    single = _run(model, tmp_path, direction_source="logit", layers=[2])
    undone = _run(
        model, tmp_path, direction_source="logit", layers=[1, 2], clamp_to_clean=False
    )

    assert torch.allclose(clamped.swapped_logits, single.swapped_logits, atol=1e-4)
    gap = lambda logits: logits[0, TARGET] - logits[0, SOURCE]
    # Without clamping, the second layer swaps the coordinates back.
    assert torch.sign(gap(undone.swapped_logits)) == torch.sign(gap(undone.baseline_logits))
    assert torch.sign(gap(clamped.swapped_logits)) != torch.sign(gap(clamped.baseline_logits))


def test_capture_forward_returns_clean_outputs_and_removes_hooks():
    instance = FakeInstance()
    residual, captured = instance.capture_forward(torch.zeros(1, SEQ_LEN), [0, 2])
    assert set(captured) == {0, 2}
    assert torch.allclose(captured[0], instance.embedded * 1.0)
    assert torch.allclose(captured[2], instance.embedded * 1.0 * 1.1 * 1.2)
    assert torch.allclose(residual, instance.embedded * 1.0 * 1.1 * 1.2 * 1.3)
    assert all(not layer._forward_hooks for layer in instance.layers)


def test_only_selected_positions_change(tmp_path):
    result = _run(
        _model(), tmp_path, token_pos=[1, -1], read_positions=list(range(SEQ_LEN))
    )
    changed = (result.swapped_logits != result.baseline_logits).any(dim=-1)
    assert changed.tolist() == [False, True, False, False, False, True]


def test_final_layer_swap_reaches_the_returned_residual(tmp_path):
    instance = FakeInstance()
    model = _model(instance)
    result = _run(model, tmp_path, layers=[N_LAYERS - 1], direction_source="logit")
    assert not torch.allclose(result.swapped_logits, result.baseline_logits)


def test_tuple_layer_outputs_are_supported(tmp_path):
    result = _run(_model(FakeInstance(returns_tuple=True)), tmp_path)
    assert not torch.allclose(result.swapped_logits, result.baseline_logits)


def test_half_precision_residual_is_restored_after_the_swap(tmp_path):
    instance = FakeInstance(dtype=torch.float16)
    seen = []
    instance.layers[-1].register_forward_hook(
        lambda _module, inputs, _output: seen.append(inputs[0].dtype)
    )
    result = _run(_model(instance), tmp_path)
    assert set(seen) == {torch.float16}
    assert not torch.allclose(result.swapped_logits, result.baseline_logits)


def test_hooks_are_removed_after_success_and_failure(tmp_path):
    instance = FakeInstance()
    model = _model(instance)
    _run(model, tmp_path)
    assert all(not layer._forward_hooks for layer in instance.layers)

    instance.forward = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        instance.swap_forward(
            torch.zeros(1, SEQ_LEN),
            {1: (torch.ones(D_MODEL), torch.ones(D_MODEL))},
            [0],
        )
    assert all(not layer._forward_hooks for layer in instance.layers)


def test_jlens_direction_is_the_normalized_lens_row(tmp_path):
    instance = FakeInstance()
    model = _model(instance)
    source, target = _jlens_directions(model, [1])[1]
    expected = instance.unembedding[SOURCE] @ model.lens.lens.jacobians[1]
    assert torch.allclose(source, expected / expected.norm())

    result = _run(model, tmp_path, layers=[1])
    assert result.direction_cosines[1] == pytest.approx((source @ target).item())


def test_logit_control_needs_no_lens_and_uses_the_unembedding(tmp_path):
    instance = FakeInstance()
    model = _model(instance, with_lens=False)
    result = _run(model, tmp_path, direction_source="logit")

    source = instance.unembedding[SOURCE] / instance.unembedding[SOURCE].norm()
    target = instance.unembedding[TARGET] / instance.unembedding[TARGET].norm()
    final = _expected_final(
        instance, {1: (source, target), 2: (source, target)}, list(range(SEQ_LEN))
    )
    assert torch.allclose(result.swapped_logits, instance.unembed(final[0][[-1]]))


def test_random_control_is_seeded_and_varies_by_layer(tmp_path):
    model = _model(with_lens=False)
    first = _run(model, tmp_path, direction_source="random", random_seed=1)
    again = _run(model, tmp_path, direction_source="random", random_seed=1)
    other = _run(model, tmp_path, direction_source="random", random_seed=2)

    assert torch.equal(first.swapped_logits, again.swapped_logits)
    assert not torch.equal(first.swapped_logits, other.swapped_logits)
    assert first.direction_cosines[1] != first.direction_cosines[2]


def test_jlens_source_requires_a_loaded_lens(tmp_path):
    with pytest.raises(RuntimeError, match="needs a Jlens"):
        _run(_model(with_lens=False), tmp_path)


def test_projected_lens_is_rejected(tmp_path):
    model = _model()
    model.lens.lens.d_model = 2
    with pytest.raises(ValueError, match="Projected lenses"):
        _run(model, tmp_path)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"direction_source": "other"}, "direction_source"),
        ({"target_token": SOURCE}, "must differ"),
        ({"layers": []}, "layers must not be empty"),
        ({"layers": [1, 1]}, "layers must be unique"),
        ({"layers": [N_LAYERS]}, "out of range"),
        ({"token_pos": []}, "token_pos must not be empty"),
        ({"token_pos": [SEQ_LEN]}, "out of range"),
        ({"token_pos": [SEQ_LEN - 1, -1]}, "token_pos must be unique"),
        ({"read_positions": []}, "read_positions must not be empty"),
        ({"read_positions": [-SEQ_LEN - 1]}, "out of range"),
    ],
)
def test_invalid_arguments_raise(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _run(_model(), tmp_path, **overrides)
