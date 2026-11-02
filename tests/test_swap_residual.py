"""Tests for MatrixTools.swap_residual. Run from the repo root: python -m pytest tests"""

import pytest
import torch

from tools.matrix_tools import MatrixTools


def _random_setup(d_model: int = 64, *, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    residual = torch.randn(3, 5, d_model, generator=generator, dtype=torch.float64)
    source = torch.randn(d_model, generator=generator, dtype=torch.float64)
    target = torch.randn(d_model, generator=generator, dtype=torch.float64)
    return residual, source / source.norm(), target / target.norm()


def _coordinates(residual, source, target):
    directions = torch.stack([source, target], dim=1)
    return residual @ torch.linalg.pinv(directions).T


def test_orthogonal_directions_swap_values():
    residual = torch.tensor([3.0, 4.0])
    patched = MatrixTools.swap_residual(
        residual, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    )
    assert torch.allclose(patched, torch.tensor([4.0, 3.0]))


def test_correlated_directions_match_worked_example():
    residual = torch.tensor([3.0, 4.0], dtype=torch.float64)
    source = torch.tensor([1.0, 0.0], dtype=torch.float64)
    target = torch.tensor([0.8, 0.6], dtype=torch.float64)
    patched = MatrixTools.swap_residual(residual, source, target)
    assert torch.allclose(patched, torch.tensor([4.8, -1.4], dtype=torch.float64))


def test_coordinates_are_exchanged_and_shape_is_preserved():
    residual, source, target = _random_setup()
    patched = MatrixTools.swap_residual(residual, source, target)
    assert patched.shape == residual.shape
    assert torch.allclose(
        _coordinates(patched, source, target),
        _coordinates(residual, source, target).flip(-1),
    )


def test_orthogonal_component_is_unchanged():
    residual, source, target = _random_setup()
    patched = MatrixTools.swap_residual(residual, source, target)
    basis, _ = torch.linalg.qr(torch.stack([source, target], dim=1))
    assert torch.allclose(
        (patched - residual) @ basis @ basis.T, patched - residual
    )


def test_swapping_twice_restores_the_residual():
    residual, source, target = _random_setup()
    once = MatrixTools.swap_residual(residual, source, target)
    twice = MatrixTools.swap_residual(once, source, target)
    assert torch.allclose(twice, residual)


def test_alpha_scales_swapped_coordinates():
    residual, source, target = _random_setup()
    patched = MatrixTools.swap_residual(residual, source, target, alpha=2.0)
    assert torch.allclose(
        _coordinates(patched, source, target),
        2.0 * _coordinates(residual, source, target).flip(-1),
    )


def test_alpha_zero_removes_both_directions():
    residual, source, target = _random_setup()
    patched = MatrixTools.swap_residual(residual, source, target, alpha=0.0)
    assert torch.allclose(
        _coordinates(patched, source, target),
        torch.zeros(3, 5, 2, dtype=torch.float64),
        atol=1e-10,
    )


def test_reference_clamps_coordinates_to_its_swapped_values():
    residual, source, target = _random_setup()
    reference = residual * 3.0 + 1.0
    patched = MatrixTools.swap_residual(residual, source, target, reference=reference)
    assert torch.allclose(
        _coordinates(patched, source, target),
        _coordinates(reference, source, target).flip(-1),
    )


def test_clamping_twice_is_stable_and_does_not_undo_the_swap():
    residual, source, target = _random_setup()
    once = MatrixTools.swap_residual(residual, source, target, reference=residual)
    twice = MatrixTools.swap_residual(once, source, target, reference=residual)
    assert torch.allclose(once, twice)
    assert not torch.allclose(twice, residual)


def test_reference_equal_to_residual_matches_the_default():
    residual, source, target = _random_setup()
    assert torch.allclose(
        MatrixTools.swap_residual(residual, source, target, reference=residual),
        MatrixTools.swap_residual(residual, source, target),
    )


def test_reference_shape_must_match():
    residual, source, target = _random_setup()
    with pytest.raises(ValueError, match="Reference shape"):
        MatrixTools.swap_residual(residual, source, target, reference=residual[0])


def test_identical_directions_leave_the_residual_unchanged():
    residual, source, _ = _random_setup()
    patched = MatrixTools.swap_residual(residual, source, source)
    assert torch.allclose(patched, residual)


def test_float32_residual_with_float16_directions():
    residual, source, target = _random_setup()
    patched = MatrixTools.swap_residual(
        residual.float(), source.to(torch.float16), target.to(torch.float16)
    )
    assert patched.dtype == torch.float32


@pytest.mark.parametrize(
    "residual, source, target",
    [
        (torch.zeros(4), torch.zeros(3), torch.zeros(3)),
        (torch.zeros(3), torch.zeros(3), torch.zeros(4)),
        (torch.zeros(3), torch.zeros(3, 1), torch.zeros(3, 1)),
        (torch.tensor(1.0), torch.zeros(3), torch.zeros(3)),
    ],
)
def test_invalid_shapes_raise(residual, source, target):
    with pytest.raises(ValueError):
        MatrixTools.swap_residual(residual, source, target)
