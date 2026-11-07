"""Tiny CPU cases for the versioned distribution's KL and unchanged sampling."""

from unittest.mock import patch

import pytest
import torch
from ray.rllib.models.torch.torch_action_dist import TorchCategorical

from moldr.e2e import action_dist
from moldr.e2e.action_dist import (
    ACTION_DIST_NAME,
    StableMaskedTorchCategorical,
    register_stable_action_dist,
)
from moldr.e2e.model import FLOAT_MIN

DTYPES = (torch.float32, torch.float64)


class _Diagnostics:
    def __init__(self):
        self.calls = []

    def _record_kl_diagnostics(self, underflow, mismatches):
        self.calls.append((underflow, mismatches))


@pytest.mark.parametrize("dtype", DTYPES)
def test_finite_logits_underflow_has_finite_kl_and_correct_gradient(dtype):
    extreme = -120.0 if dtype == torch.float32 else -1000.0
    old_logits = torch.tensor([[0.0, 0.0]], dtype=dtype)
    new_logits = torch.tensor([[0.0, extreme]], dtype=dtype, requires_grad=True)
    model = _Diagnostics()
    old = StableMaskedTorchCategorical(old_logits, model)
    new = StableMaskedTorchCategorical(new_logits)
    assert torch.isfinite(new.dist.logits).all()
    assert new.dist.probs[0, 1] == 0
    assert torch.isinf(
        TorchCategorical(old_logits).kl(TorchCategorical(new_logits))
    ).all()
    divergence = old.kl(new)
    expected = -extreme / 2 - torch.log(torch.tensor(2.0, dtype=dtype))
    torch.testing.assert_close(divergence, expected.reshape(1))
    divergence.sum().backward()
    torch.testing.assert_close(
        new_logits.grad, torch.tensor([[0.5, -0.5]], dtype=dtype)
    )
    assert len(model.calls) == 1
    underflow, mismatches = model.calls[0]
    assert underflow == 1 and mismatches == 0
    assert underflow.ndim == mismatches.ndim == 0
    assert underflow.dtype == mismatches.dtype == torch.int64
    assert underflow.device == new_logits.device
    assert not underflow.requires_grad and not mismatches.requires_grad


@pytest.mark.parametrize("dtype", DTYPES)
def test_ordinary_kl_and_both_gradients_match_stock(dtype):
    old_logits = torch.tensor(
        [[0.2, 1.0, -1.2], [2.0, -0.5, 0.3]], dtype=dtype, requires_grad=True
    )
    new_logits = torch.tensor(
        [[0.3, 0.8, -0.4], [1.3, 0.2, -0.1]], dtype=dtype, requires_grad=True
    )
    expected = TorchCategorical(old_logits).kl(TorchCategorical(new_logits))
    actual = StableMaskedTorchCategorical(old_logits).kl(
        StableMaskedTorchCategorical(new_logits)
    )
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(
        actual.sum(), (old_logits, new_logits), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.sum(), (old_logits, new_logits))
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zero", [FLOAT_MIN, float("-inf")])
def test_shared_mask_single_valid_and_all_masked_kl_have_finite_gradients(dtype, zero):
    # The final row is a defensive all-masked finite-sentinel case. The real
    # environment guarantees at least one valid action; true all--inf cannot
    # represent a normalized categorical distribution and is tested separately.
    old_rows = [[0.0, zero, -1.0], [zero, 2.0, zero]]
    new_rows = [[0.5, zero, -0.8], [zero, -3.0, zero]]
    if zero == FLOAT_MIN:
        old_rows.append([zero, zero, zero])
        new_rows.append([zero, zero, zero])
    old_logits = torch.tensor(old_rows, dtype=dtype, requires_grad=True)
    new_logits = torch.tensor(new_rows, dtype=dtype, requires_grad=True)
    model = _Diagnostics()
    old = StableMaskedTorchCategorical(old_logits, model)
    new = StableMaskedTorchCategorical(new_logits)
    actual = old.kl(new)
    expected = TorchCategorical(old_logits).kl(TorchCategorical(new_logits))
    torch.testing.assert_close(actual, expected)
    assert actual[1] == 0
    if zero == FLOAT_MIN:
        assert actual[2] == 0
    actual.sum().backward()
    assert (
        torch.isfinite(old_logits.grad).all() and torch.isfinite(new_logits.grad).all()
    )
    assert old_logits.grad[0, 1] == new_logits.grad[0, 1] == 0
    assert model.calls[0][0] == 0 and model.calls[0][1] == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zero", [FLOAT_MIN, float("-inf")])
@pytest.mark.parametrize("old_rare", [False, True])
def test_old_live_new_structural_zero_keeps_infinite_kl_even_if_p_underflows(
    dtype, zero, old_rare
):
    rare = -120.0 if dtype == torch.float32 else -1000.0
    old_logits = torch.tensor([[0.0, rare if old_rare else 0.0]], dtype=dtype)
    new_logits = torch.tensor([[0.0, zero]], dtype=dtype)
    model = _Diagnostics()
    old = StableMaskedTorchCategorical(old_logits, model)
    if old_rare:
        assert old.dist.probs[0, 1] == 0
    actual = old.kl(StableMaskedTorchCategorical(new_logits))
    assert torch.isposinf(actual).all()
    assert model.calls[0][0] == 0 and model.calls[0][1] == 1


@pytest.mark.parametrize("dtype", DTYPES)
def test_reopening_old_mask_is_finite_but_counts_a_changed_mask(dtype):
    model = _Diagnostics()
    old = StableMaskedTorchCategorical(
        torch.tensor([[0.0, FLOAT_MIN]], dtype=dtype), model
    )
    new = StableMaskedTorchCategorical(torch.tensor([[0.0, 0.0]], dtype=dtype))
    torch.testing.assert_close(old.kl(new), torch.tensor([2.0], dtype=dtype).log())
    assert model.calls[0][0] == 0 and model.calls[0][1] == 1


@pytest.mark.parametrize("dtype", DTYPES)
def test_underflow_and_mismatch_counters_count_terms_and_rows_separately(dtype):
    rare = -120.0 if dtype == torch.float32 else -1000.0
    model = _Diagnostics()
    old = StableMaskedTorchCategorical(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, FLOAT_MIN],
                [0.0, 0.0, 0.0, FLOAT_MIN],
                [0.0, FLOAT_MIN, FLOAT_MIN, FLOAT_MIN],
            ],
            dtype=dtype,
        ),
        model,
    )
    new = StableMaskedTorchCategorical(
        torch.tensor(
            [
                [0.0, rare, rare, FLOAT_MIN],
                [0.0, FLOAT_MIN, FLOAT_MIN, FLOAT_MIN],
                [0.0, 0.0, FLOAT_MIN, FLOAT_MIN],
            ],
            dtype=dtype,
        )
    )
    result = old.kl(new)
    assert (
        torch.isfinite(result[0])
        and torch.isposinf(result[1])
        and torch.isfinite(result[2])
    )
    underflow, mismatches = model.calls[0]
    assert underflow == 2  # Two finite, live q probabilities underflow in row 0.
    assert mismatches == 2  # Any changed mask counts once per row, in either direction.


@pytest.mark.parametrize("dtype", DTYPES)
def test_parent_sampling_logp_entropy_and_rng_are_unchanged(dtype):
    logits = torch.tensor([[0.3, FLOAT_MIN, -1.0], [1.0, 0.0, -0.3]], dtype=dtype)
    old = StableMaskedTorchCategorical(logits)
    new = StableMaskedTorchCategorical(
        logits + torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.0, 0.2]], dtype=dtype)
    )
    stock = TorchCategorical(logits)
    actions = torch.tensor([0, 2])
    torch.testing.assert_close(old.logp(actions), stock.logp(actions))
    torch.testing.assert_close(old.entropy(), stock.entropy())
    torch.testing.assert_close(old.deterministic_sample(), stock.deterministic_sample())
    torch.manual_seed(725)
    before = torch.get_rng_state()
    old.kl(new)
    assert torch.equal(torch.get_rng_state(), before)
    expected_samples = stock.sample()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(before)
    actual_samples = old.sample()
    assert torch.equal(actual_samples, expected_samples)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(old.sampled_action_logp(), stock.sampled_action_logp())
    assert StableMaskedTorchCategorical.sample is TorchCategorical.sample
    assert StableMaskedTorchCategorical.logp is TorchCategorical.logp
    assert StableMaskedTorchCategorical.entropy is TorchCategorical.entropy


def test_diagnostics_callback_has_no_scalar_extraction_or_cpu_transfer():
    model = _Diagnostics()
    old = StableMaskedTorchCategorical(torch.tensor([[0.0, 0.0]]), model)
    new = StableMaskedTorchCategorical(torch.tensor([[0.0, -120.0]]))
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("item in KL")),
        patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("cpu transfer in KL")
        ),
    ):
        result = old.kl(new)
    assert torch.isfinite(result).all() and model.calls[0][0] == 1


def test_diagnostics_can_use_current_model_when_previous_has_none():
    model = _Diagnostics()
    old = StableMaskedTorchCategorical(torch.tensor([[0.0, 0.0]]))
    new = StableMaskedTorchCategorical(torch.tensor([[0.0, -120.0]]), model)
    assert torch.isfinite(old.kl(new)).all()
    assert model.calls[0][0] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_kl_gradients_and_device_diagnostics_agree():
    results = []
    for device in ("cpu", "cuda"):
        model = _Diagnostics()
        old_logits = torch.tensor([[0.0, 0.0, FLOAT_MIN]], device=device)
        new_logits = torch.tensor(
            [[0.0, -120.0, FLOAT_MIN]], device=device, requires_grad=True
        )
        old = StableMaskedTorchCategorical(old_logits, model)
        new = StableMaskedTorchCategorical(new_logits)
        divergence = old.kl(new)
        divergence.sum().backward()
        underflow, mismatches = model.calls[0]
        assert underflow.device.type == mismatches.device.type == device
        assert underflow == 1 and mismatches == 0
        assert (
            torch.isfinite(divergence).all() and torch.isfinite(new_logits.grad).all()
        )
        results.append((divergence.detach().cpu(), new_logits.grad.detach().cpu()))
    for cpu_value, cuda_value in zip(*results, strict=True):
        torch.testing.assert_close(cpu_value, cuda_value)


def test_invalid_all_true_negative_infinity_logits_still_fail_validation():
    with pytest.raises(ValueError, match="invalid values"):
        StableMaskedTorchCategorical(torch.tensor([[float("-inf"), float("-inf")]]))


def test_registration_uses_new_name_without_changing_stock(monkeypatch):
    calls = []
    monkeypatch.setattr(
        action_dist.ModelCatalog,
        "register_custom_action_dist",
        lambda name, cls: calls.append((name, cls)),
    )
    assert (
        register_stable_action_dist()
        == ACTION_DIST_NAME
        == "moldr_stable_masked_categorical_v1"
    )
    assert calls == [(ACTION_DIST_NAME, StableMaskedTorchCategorical)]
    # The base distribution remains the stock implementation for old models.
    assert torch.isinf(
        TorchCategorical(torch.tensor([[0.0, 0.0]])).kl(
            TorchCategorical(torch.tensor([[0.0, -120.0]]))
        )
    ).all()
