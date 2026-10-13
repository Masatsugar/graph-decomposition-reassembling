"""Pointer-style, action-masked policy for the end-to-end reassembly MDP.

At every step the agent chooses one option from a set of embedded molecules
(building blocks in phase A, candidate merges in phase B). The policy scores
each option from its embedding conditioned on the current molecule + phase, and
an action mask hides padded / invalid slots. The same network therefore drives
both decisions, which is what makes the attachment choice *learned* instead of
a greedy ``argmax``.

``PointerPolicyNet`` is plain PyTorch (unit-testable without Ray).
``MoldrPointerModel`` is the thin RLlib ``TorchModelV2`` wrapper; its Ray import
is guarded so this module stays importable (and ``PointerPolicyNet`` testable)
even where Ray is absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Large *finite* negative used to mask invalid action logits. softmax drives
# these to ~0 probability (exp(-1e9) underflows to 0) while, unlike
# torch.finfo.min, never overflowing float32 when RLlib reduces the stored
# action-distribution inputs (e.g. its debug batch summary).
FLOAT_MIN = -1.0e9


class PointerPolicyNet(nn.Module):
    """Scores a variable set of options against a context (base molecule).

    Shapes (B = batch, A = number of option slots, D = embedding dim)::

        base:    [B, D]
        phase:   [B, 2]
        options: [B, A, D]
        mask:    [B, A]   (1 = live option, 0 = padding/invalid)

    Returns ``(logits [B, A], value [B])`` with masked logits set to
    ``FLOAT_MIN`` so the action distribution never selects a padded slot.
    """

    def __init__(self, obs_dim: int = 300, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        ctx_in = obs_dim + 2  # base embedding + phase one-hot
        self.ctx_enc = nn.Sequential(nn.Linear(ctx_in, hidden), nn.ReLU())
        self.opt_enc = nn.Linear(obs_dim, hidden)
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.vf = nn.Sequential(
            nn.Linear(ctx_in, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, base, phase, options, mask):
        ctx_in = torch.cat([base, phase], dim=-1)  # [B, D+2]
        ctx = self.ctx_enc(ctx_in)  # [B, H]
        opt = self.opt_enc(options)  # [B, A, H]
        b, a, h = opt.shape
        ctx_t = ctx.unsqueeze(1).expand(b, a, h)  # [B, A, H]
        pair = torch.cat([ctx_t, opt], dim=-1)  # [B, A, 2H]
        scores = self.scorer(pair).squeeze(-1)  # [B, A]

        # log(mask): 0 where valid, -inf where masked -> clamp to FLOAT_MIN.
        inf_mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)
        logits = scores + inf_mask  # [B, A]

        value = self.vf(ctx_in).squeeze(-1)  # [B]
        return logits, value


# --------------------------------------------------------------------------- #
# RLlib wrapper (guarded import so PointerPolicyNet stays usable without Ray).
# --------------------------------------------------------------------------- #
try:
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

    _HAS_RAY = True
except Exception:  # pragma: no cover - exercised only where Ray is absent
    ModelCatalog = None
    TorchModelV2 = object
    _HAS_RAY = False


MODEL_NAME = "moldr_pointer"


class MoldrPointerModel(TorchModelV2, nn.Module):
    """RLlib ModelV2 adapter around :class:`PointerPolicyNet`.

    Expects a ``Dict`` observation with keys ``base``, ``phase``, ``options``
    and ``action_mask`` (see :class:`moldr.e2e.env.MolEnvE2E`). ``num_outputs``
    must equal the number of option slots (``action_space.n``).
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        custom = (model_config or {}).get("custom_model_config", {})
        obs_dim = int(custom.get("obs_dim", 300))
        hidden = int(custom.get("hidden", 256))
        self.net = PointerPolicyNet(obs_dim=obs_dim, hidden=hidden)
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        logits, value = self.net(
            obs["base"], obs["phase"], obs["options"], obs["action_mask"]
        )
        self._value = value
        return logits, state

    def value_function(self):
        return self._value


def register_pointer_model():
    """Register the pointer model with RLlib's ModelCatalog (idempotent)."""
    if not _HAS_RAY:
        raise RuntimeError("Ray RLlib is not available; cannot register model.")
    ModelCatalog.register_custom_model(MODEL_NAME, MoldrPointerModel)
    return MODEL_NAME
