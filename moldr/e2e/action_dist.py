"""Categorical KL that distinguishes probability underflow from action masks.

PyTorch 2.7's categorical KL treats ``q.probs == 0`` as a support mismatch.
Softmax can round a strictly positive probability to zero even while its log
probability is finite. Computing with normalized logits avoids that false inf.
Sampling, log probabilities, entropy and the random-number stream remain the
inherited RLlib implementations.
"""

from __future__ import annotations

import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchCategorical

from .model import FLOAT_MIN

ACTION_DIST_NAME = "moldr_stable_masked_categorical_v1"


class StableMaskedTorchCategorical(TorchCategorical):
    """Use log-domain KL, retaining true and explicit-mask support mismatches.

    Raw ``FLOAT_MIN`` inputs are the pointer policy's explicit mask sentinel;
    actual ``-inf`` normalized logits are structural zeros as well. The policy
    requires the same mask for the old and current distribution on each state.
    A structural zero in q where p is live still yields infinite KL, even if
    p's strictly positive probability also underflows numerically.

    If the model exposes ``_record_kl_diagnostics(underflow_terms, mismatch_rows)``,
    KL reports detached scalar int64 tensors on the current device. Mismatch
    rows count any change of structural support, in either direction. There is
    no device-to-host transfer or scalar extraction in this method.
    """

    def kl(self, other):
        log_p, log_q = self.dist.logits, other.dist.logits
        zero_p = (self.inputs == FLOAT_MIN) | torch.isneginf(log_p)
        zero_q = (other.inputs == FLOAT_MIN) | torch.isneginf(log_q)
        active = ~(zero_p | zero_q)
        # Remove structural-zero terms before subtraction: (-inf)-(-inf) and
        # 0*inf must not introduce NaNs into the forward or backward pass.
        delta = torch.where(active, log_p, 0.0) - torch.where(active, log_q, 0.0)
        divergence = (self.dist.probs * delta).sum(dim=-1)
        support_mismatch = (~zero_p & zero_q).any(dim=-1)
        divergence = torch.where(
            support_mismatch, torch.full_like(divergence, torch.inf), divergence
        )

        model = self.model if self.model is not None else other.model
        record = getattr(model, "_record_kl_diagnostics", None)
        if callable(record):
            with torch.no_grad():
                underflow = (
                    active
                    & torch.isfinite(log_p)
                    & torch.isfinite(log_q)
                    & (self.dist.probs > 0)
                    & (other.dist.probs == 0)
                ).sum(dtype=torch.int64)
                mismatches = (zero_p != zero_q).any(dim=-1).sum(dtype=torch.int64)
                record(underflow.detach(), mismatches.detach())
        return divergence


def register_stable_action_dist():
    """Register only the opt-in versioned distribution; do not patch Torch KL."""
    ModelCatalog.register_custom_action_dist(
        ACTION_DIST_NAME, StableMaskedTorchCategorical
    )
    return ACTION_DIST_NAME
