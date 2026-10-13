"""Versioned E2E policy: shared molecule interface and PPO candidate values.

The old ``moldr_pointer`` model and its checkpoint layout remain in model.py.
This module exclusively registers ``moldr_pointer_v2``. The candidate value is
recomputed from training observations, never taken from the most recent RLlib
forward (which may have been an extra bootstrap/value forward).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from moldr.gnn import MPNNEncoder, graph_record
from moldr.profiling import span

from .model import _HAS_RAY, FLOAT_MIN, ModelCatalog, TorchModelV2

MODEL_NAME = "moldr_pointer_v2"
MODEL_NAME_V2 = MODEL_NAME


class PointerPolicyNetV2(nn.Module):
    """Actor, state value, and optional candidate value over molecule vectors.

    Rich context includes the base, phase, score/elapsed/remaining steps and
    masked mean/max option embeddings. Classic context is base plus phase.
    ``alpha=0`` keeps the candidate-value auxiliary loss without adding U to
    the actor. With ``candidate_value=False`` the U head does not exist.
    """

    def __init__(
        self, obs_dim=300, hidden=256, context="rich", candidate_value=False,
        alpha=1.0, temperature=1.0,
    ):
        super().__init__()
        if context not in ("rich", "classic"):
            raise ValueError("context must be 'rich' or 'classic'.")
        if temperature <= 0 or not np.isfinite(temperature) or not np.isfinite(alpha):
            raise ValueError("temperature must be positive; alpha and temperature must be finite.")
        self.obs_dim = int(obs_dim)
        self.context = context
        self.context_aggregation = "mean_max" if context == "rich" else "none"
        self.candidate_value_enabled = bool(candidate_value)
        self.alpha, self.temperature = float(alpha), float(temperature)
        ctx_dim = obs_dim + 2 if context == "classic" else 3 * obs_dim + 5
        self.ctx_enc = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ReLU())
        self.opt_enc = nn.Linear(obs_dim, hidden)
        self.scorer = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.vf = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.candidate_value_head = (
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            if candidate_value else None
        )

    @staticmethod
    def _score(head, context, options):
        # Factor the repeated context GEMM, retaining the original dense head.
        hidden = context.shape[-1]
        first = head[0]
        score = F.linear(context, first.weight[:, :hidden], first.bias).unsqueeze(1)
        score = score + F.linear(options, first.weight[:, hidden:])
        return head[2](head[1](score)).squeeze(-1)

    def _features(self, base, phase, options, mask, context):
        dtype = self.opt_enc.weight.dtype
        base, phase, options = base.to(dtype), phase.to(dtype), options.to(dtype)
        if self.context == "classic":
            features = torch.cat([base, phase], dim=-1)
        else:
            if context is None:
                raise ValueError("Rich policy context requires score, elapsed, and remaining steps.")
            active = mask > 0
            summed = torch.where(active.unsqueeze(-1), options, 0).sum(dim=1)
            count = active.sum(dim=1, keepdim=True).clamp_min(1)
            mean = summed / count
            maximum = options.masked_fill(~active.unsqueeze(-1), torch.finfo(dtype).min).amax(dim=1)
            maximum = torch.where(active.any(dim=1, keepdim=True), maximum, 0)
            features = torch.cat([base, phase, context.to(dtype), mean, maximum], dim=-1)
        return features, self.ctx_enc(features), self.opt_enc(options)

    def candidate_values(self, base, phase, options, mask, context=None):
        if self.candidate_value_head is None:
            raise RuntimeError("Candidate-value head is disabled.")
        _, ctx, opt = self._features(base, phase, options, mask, context)
        with span("PointerPolicyNetV2.candidate_values", "candidate_value"):
            return self._score(self.candidate_value_head, ctx, opt)

    def forward(self, base, phase, options, mask, context=None):
        features, ctx, opt = self._features(base, phase, options, mask, context)
        logits = self._score(self.scorer, ctx, opt)
        # Distribution computations stay finite in float16/bfloat16 models.
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        if self.candidate_value_head is not None and self.alpha != 0:
            with span("PointerPolicyNetV2.candidate_value_logits", "candidate_value"):
                u = self._score(self.candidate_value_head, ctx, opt)
                phase_b = (phase[:, 1] > 0.5).unsqueeze(-1)
                logits = logits + torch.where(phase_b, self.alpha * u.detach().to(logits.dtype) / self.temperature, 0)
        logits = logits.masked_fill(mask <= 0, FLOAT_MIN)
        return logits, self.vf(features).squeeze(-1)


class E2EPolicyNetV2(nn.Module):
    """A common policy interface for fixed Mol2Vec and frozen/trainable MPNN.

    Input is the version-2 observation dictionary with a leading batch axis.
    GNN fragment IDs refer to ``catalog_smiles``; attachment graphs use only
    the ``max_attach`` axis even when the fragment catalog is much larger.
    """

    def __init__(self, **custom):
        super().__init__()
        self.representation = custom.get("representation", "mol2vec")
        if self.representation not in ("mol2vec", "mpnn"):
            raise ValueError("representation must be 'mol2vec' or 'mpnn'.")
        self.trainable_encoder = bool(custom.get("trainable", self.representation == "mpnn"))
        self.embedding_cache_size = int(custom.get("embedding_cache_size", 0))
        if self.embedding_cache_size < 0:
            raise ValueError("embedding_cache_size must be nonnegative.")
        if self.trainable_encoder and self.embedding_cache_size:
            raise ValueError("Persistent embedding caches require a frozen encoder.")
        self._catalog_cache_order = OrderedDict()
        if self.representation == "mol2vec" and self.trainable_encoder:
            raise ValueError("Mol2Vec observations are fixed vectors; trainable Mol2Vec is unsupported.")
        self.auxiliary_loss_coeff = float(custom.get("auxiliary_loss_coeff", 0.5))
        if self.auxiliary_loss_coeff < 0 or not np.isfinite(self.auxiliary_loss_coeff):
            raise ValueError("auxiliary_loss_coeff must be finite and nonnegative.")
        self.max_attach = int(custom.get("max_attach", 128))
        if self.max_attach <= 0:
            raise ValueError("max_attach must be positive.")
        if self.representation == "mpnn":
            obs_dim = int(custom.get("graph_output", 128))
            self.encoder = MPNNEncoder(
                hidden=int(custom.get("graph_hidden", 128)), output_dim=obs_dim,
                layers=int(custom.get("layers", 3)),
                atom_chunk_size=int(custom.get("atom_chunk_size", 4096)),
                gradient_checkpointing=bool(custom.get("gradient_checkpointing", False)),
                trainable=self.trainable_encoder,
            )
            self._build_catalog(custom.get("catalog_smiles", []), obs_dim)
        else:
            obs_dim = int(custom.get("obs_dim", 300))
            self.encoder = nn.Identity()
        self.heads = PointerPolicyNetV2(
            obs_dim=obs_dim, hidden=int(custom.get("hidden", 256)),
            context=custom.get("context", "rich"),
            candidate_value=bool(custom.get("candidate_value", False)),
            alpha=float(custom.get("alpha", 1)), temperature=float(custom.get("temperature", 1)),
        )

    def _build_catalog(self, smiles, obs_dim):
        from moldr.chemutils import get_mol

        if not smiles:
            raise ValueError("MPNN requires a nonempty stable catalog_smiles vocabulary.")
        # Match the environment's Kekulé bond types and retained aromatic flags
        # so the same molecule has the same representation in either phase.
        mols = [get_mol(s) for s in smiles]
        if any(m is None or m.GetNumAtoms() == 0 for m in mols):
            raise ValueError("catalog_smiles contains an invalid or empty molecule.")
        max_atoms = max(m.GetNumAtoms() for m in mols)
        max_bonds = max(m.GetNumBonds() for m in mols)
        records = [graph_record(m, max_atoms, max_bonds) for m in mols]
        for key in ("atoms", "bonds", "counts"):
            self.register_buffer("catalog_" + key, torch.as_tensor(np.stack([r[key] for r in records])))
        # Persistent embedding caching is deliberately limited to frozen
        # encoders. These derived buffers are recomputed after checkpoint load.
        self.register_buffer("_catalog_embeddings", torch.zeros(len(smiles), obs_dim), persistent=False)
        self.register_buffer("_catalog_cached", torch.zeros(len(smiles), dtype=torch.bool), persistent=False)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        if self.representation == "mpnn":
            self._catalog_cached.zero_()
            self._catalog_cache_order.clear()

    def _fragments(self, ids):
        unique, inverse = torch.unique(ids, return_inverse=True)
        if torch.any(unique < 0) or torch.any(unique >= self.catalog_counts.shape[0]):
            raise ValueError("A live fragment_id lies outside catalog_smiles.")
        if self.trainable_encoder or not self.embedding_cache_size:
            embeddings = self.encoder(self.catalog_atoms[unique], self.catalog_bonds[unique], self.catalog_counts[unique])
        else:
            missing = unique[~self._catalog_cached[unique]]
            if missing.numel():
                embeddings = self.encoder(self.catalog_atoms[missing], self.catalog_bonds[missing], self.catalog_counts[missing])
                self._catalog_embeddings[missing] = embeddings
                self._catalog_cached[missing] = True
            embeddings = self._catalog_embeddings[unique]
            # Copy the selected values before evicting entries: a minibatch can
            # legitimately request more IDs than the persistent cache capacity.
            for identifier in unique.detach().cpu().tolist():
                self._catalog_cache_order.pop(identifier, None)
                self._catalog_cache_order[identifier] = None
            while len(self._catalog_cache_order) > self.embedding_cache_size:
                evicted, _ = self._catalog_cache_order.popitem(last=False)
                self._catalog_cached[evicted] = False
        return embeddings[inverse]

    def encode_observation(self, obs):
        if self.representation == "mol2vec":
            return obs["base"], obs["options"]
        base = self.encoder(obs["base_atoms"], obs["base_bonds"], obs["base_counts"])
        mask, phase = obs["action_mask"], obs["phase"]
        batch, slots = mask.shape
        options = base.new_zeros((batch, slots, self.encoder.output_dim))
        live_a = (mask > 0) & (phase[:, 0] > 0.5).unsqueeze(-1)
        if torch.any(live_a):
            positions = live_a.nonzero(as_tuple=False)
            ids = obs["fragment_ids"][live_a].long()
            options[positions[:, 0], positions[:, 1]] = self._fragments(ids)
        candidate_slots = obs["candidate_counts"].shape[1]
        if candidate_slots != self.max_attach or candidate_slots > slots:
            raise ValueError("Candidate graph axis must equal max_attach and fit the action space.")
        live_b = (mask[:, :candidate_slots] > 0) & (phase[:, 1] > 0.5).unsqueeze(-1)
        if torch.any((mask[:, candidate_slots:] > 0) & (phase[:, 1] > 0.5).unsqueeze(-1)):
            raise ValueError("Attachment action exceeds the max_attach graph axis.")
        if torch.any(live_b):
            positions = live_b.nonzero(as_tuple=False)
            counts = obs["candidate_counts"][live_b]
            if torch.any(counts[:, 0] <= 0):
                raise ValueError("Live attachment candidates require nonempty graphs.")
            embeddings = self.encoder(obs["candidate_atoms"][live_b], obs["candidate_bonds"][live_b], counts)
            options[positions[:, 0], positions[:, 1]] = embeddings
        return base, options

    def forward(self, obs):
        base, options = self.encode_observation(obs)
        return self.heads(base, obs["phase"], options, obs["action_mask"], obs.get("context"))

    def candidate_values(self, obs):
        base, options = self.encode_observation(obs)
        return self.heads.candidate_values(base, obs["phase"], options, obs["action_mask"], obs.get("context"))

    def auxiliary_loss(self, obs, actions, value_targets):
        """Regress only chosen phase-B candidates on PPO's return targets."""
        u = self.candidate_values(obs)
        actions = actions.long().reshape(-1)
        targets = value_targets.reshape(-1).float().detach()
        if actions.shape[0] != u.shape[0] or targets.shape[0] != u.shape[0]:
            raise ValueError("Candidate-value actions, targets, and observations must have equal batch sizes.")
        in_range = (actions >= 0) & (actions < u.shape[1])
        safe_actions = actions.clamp(0, u.shape[1] - 1)
        selected_mask = obs["action_mask"].gather(1, safe_actions.unsqueeze(-1)).squeeze(-1) > 0
        phase_b = (obs["phase"][:, 1] > 0.5) & in_range & selected_mask
        if not torch.any(phase_b):
            return u.sum() * 0
        selected = u.gather(1, safe_actions.unsqueeze(-1)).squeeze(-1)
        return self.auxiliary_loss_coeff * F.smooth_l1_loss(selected[phase_b].float(), targets[phase_b])


class MoldrPointerModelV2(TorchModelV2, nn.Module):
    """RLlib ModelV2 adapter; both rollout and PPO use the same mixed logits."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        if num_outputs != action_space.n:
            raise ValueError("Pointer model outputs must equal the number of action slots.")
        self.net = E2EPolicyNetV2(**(model_config or {}).get("custom_model_config", {}))
        self._value = None
        self._auxiliary_loss = None

    def forward(self, input_dict, state, seq_lens):
        logits, self._value = self.net(input_dict["obs"])
        return logits, state

    def value_function(self):
        if self._value is None:
            raise RuntimeError("value_function requires a preceding forward.")
        return self._value

    def custom_loss(self, policy_loss, loss_inputs):
        if not self.net.heads.candidate_value_enabled:
            return policy_loss
        obs = loss_inputs["obs"]
        if not isinstance(obs, Mapping):
            from ray.rllib.models.modelv2 import restore_original_dimensions

            obs = restore_original_dimensions(obs, self.obs_space, "torch")
        auxiliary = self.net.auxiliary_loss(obs, loss_inputs["actions"], loss_inputs["value_targets"])
        self._auxiliary_loss = auxiliary.detach()
        if isinstance(policy_loss, (list, tuple)):
            return [loss + auxiliary for loss in policy_loss]
        return policy_loss + auxiliary

    def metrics(self):
        # RLlib's old GPU learner reduces model metrics through NumPy, unlike
        # its tensor loss statistics; exporting a CUDA tensor breaks training.
        return {} if self._auxiliary_loss is None else {"candidate_value_loss": self._auxiliary_loss.item()}


def register_pointer_model_v2():
    if not _HAS_RAY:
        raise RuntimeError("Ray RLlib is not available; cannot register model.")
    ModelCatalog.register_custom_model(MODEL_NAME, MoldrPointerModelV2)
    return MODEL_NAME
