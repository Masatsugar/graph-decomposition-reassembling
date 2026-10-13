"""VAEProposer: turn a FragmentVAE into a *fixed-size* building-block library.

The PPO reconciliation
----------------------
PPO needs a fixed ``Discrete(N)`` action space whose index->block semantics are
stable for the whole run (env.py:60 does ``self.mols[action]`` against a list
built once per episode). A generative decoder, left to its own devices, yields a
variable/continuous proposal -- incompatible.

We reconcile the two with **N frozen latent anchors**. We sample ``z_1..z_N``
once from the prior ``N(0, I)`` with a fixed seed and *freeze* them. Anchor
``z_i`` is the permanent identity of action ``i``. The library is then

        block_i = vocab[ argmax_j  decode(z_i, context)_j ]

so:

* The action set is always ``Discrete(N)`` -- ``N`` == number of anchors, chosen
  before config build. The trained policy head never changes shape.
* Action ``i`` always means "the fragment family anchored at ``z_i``" -- a
  stable semantics the policy can learn, because ``z_i`` is fixed across the
  whole run.
* ``context`` (Mol2Vec 300-d of the current base) lets the *actual* fragment
  behind each anchor adapt to the growing molecule -- i.e. the generation is
  conditional on the base embedding, as the brief asks, **without** changing the
  action space.

Re-conditioning cadence (``recond_every``) is a knob:
  * ``"never"`` / context omitted -> a static decoded library == a drop-in
    replacement for the gSpan CSV (the safest variant; nothing in env changes).
  * ``"episode"`` -> re-decode the pool at each ``reset`` with the reset-state
    context. ``self.mols`` is rebuilt once per episode and then held fixed for
    the episode, matching the env's existing "built once, constant within an
    episode" contract.
  * ``"step"`` -> re-decode every step (maximally adaptive; note the index->block
    map is then only stable *across episodes*, not within one -- offered but not
    the default).

Validity is guaranteed structurally: the decoder only ever emits a vocabulary
index, and every vocabulary entry passed ``is_valid_block`` at build time. So
``get_mol`` in env.py:28 can never see an invalid/None/single-atom block.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from moldr.generative.vae import FragmentVAE
from moldr.generative.vocab import FragmentVocab


class VAEProposer:
    def __init__(
        self,
        vae: FragmentVAE,
        vocab: FragmentVocab,
        n_blocks: int,
        anchor_seed: int = 0,
        min_ring_fraction: float = 0.3,
        dedup: bool = True,
    ):
        """
        Parameters
        ----------
        n_blocks
            ``N`` -- the fixed action-space cardinality. Frozen anchors are
            sampled once; action ``i`` is permanently bound to anchor ``i``.
        min_ring_fraction
            Guarantee at least ``ceil(min_ring_fraction * N)`` ring-bearing
            (edge-mergeable) blocks in every proposed library, so edge-merge is
            not starved (merge_edge needs a ring on both sides).
        dedup
            If True, fill any positions that collided onto an already-used
            fragment with the next-best distinct candidate (deterministic), so
            the N positions are diverse.
        """
        if not isinstance(n_blocks, Integral) or n_blocks <= 0:
            raise ValueError("n_blocks must be a positive integer")
        if not np.isfinite(min_ring_fraction) or not 0 <= min_ring_fraction <= 1:
            raise ValueError("min_ring_fraction must be between 0 and 1")
        if vae.vocab_size != len(vocab):
            raise ValueError("VAE output size must match the fragment vocabulary")
        self.vae = vae.eval()
        self.vocab = vocab
        self.n_blocks = int(n_blocks)
        self.min_ring_fraction = float(min_ring_fraction)
        self.dedup = bool(dedup)
        self.anchor_seed = int(anchor_seed)
        # Fail fast if the ring-fraction guarantee is unsatisfiable: a single
        # ring entry suffices (Pass 2 duplicates ring ids up to `need`), so the
        # only impossible case is a vocabulary with zero ring fragments.
        need = int(np.ceil(self.min_ring_fraction * self.n_blocks))
        if need > 0 and not vocab.ring_mask.any():
            raise ValueError(
                f"min_ring_fraction={self.min_ring_fraction} requires >={need} "
                "ring-bearing blocks, but the vocabulary contains no ring "
                "fragments. Pass min_ring_fraction=0.0 or supply a vocabulary "
                "with at least one ring fragment."
            )
        # Frozen anchors: shape (N, latent_dim). Sampled once, never updated.
        g = torch.Generator().manual_seed(anchor_seed)
        self.anchors = torch.randn(
            (self.n_blocks, vae.latent_dim), generator=g
        ).to(next(vae.parameters()))

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def _logits(self, context: Optional[np.ndarray]) -> np.ndarray:
        """(N, |V|) decode logits, one row per frozen anchor."""
        parameter = next(self.vae.parameters())
        # Also support callers moving/casting the model after construction.
        self.anchors = self.anchors.to(parameter)
        if context is None:
            ctx = parameter.new_zeros((self.n_blocks, self.vae.context_dim))
        else:
            values = np.asarray(context, dtype=np.float32).reshape(-1)
            if values.size != self.vae.context_dim:
                raise ValueError(
                    f"context dim {values.size} != {self.vae.context_dim}"
                )
            if not np.isfinite(values).all():
                raise ValueError("context must contain only finite values")
            c = torch.as_tensor(values, device=parameter.device, dtype=parameter.dtype)
            ctx = c.unsqueeze(0).expand(self.n_blocks, -1)
        return self.vae.decode(self.anchors, ctx).float().cpu().numpy()

    def propose(self, context: Optional[np.ndarray] = None) -> List[str]:
        """Return exactly ``N`` valid SMILES, position ``i`` from anchor ``i``.

        Deterministic given (anchors, context). Guarantees: every entry is a
        valid kekulizable multi-atom open-valence block; at least
        ``min_ring_fraction`` carry a ring.
        """
        logits = self._logits(context)  # (N, M)
        ranked = np.argsort(-logits, axis=1)  # best-first vocab ids per anchor
        ring_mask = self.vocab.ring_mask
        chosen: List[int] = [-1] * self.n_blocks
        used: set = set()

        # Pass 1: each position takes its best (distinct, if dedup) vocab id.
        for i in range(self.n_blocks):
            for vid in ranked[i]:
                vid = int(vid)
                if self.dedup and vid in used and len(used) < len(self.vocab):
                    continue
                chosen[i] = vid
                used.add(vid)
                break
            if chosen[i] == -1:  # vocab smaller than N and dedup exhausted
                chosen[i] = int(ranked[i][0])

        # Pass 2: enforce the minimum ring fraction deterministically.
        need = int(np.ceil(self.min_ring_fraction * self.n_blocks))
        have = sum(1 for vid in chosen if ring_mask[vid])
        if have < need and ring_mask.any():
            ring_ids = [int(v) for v in np.where(ring_mask)[0]]
            # Replace the weakest non-ring positions (lowest top-logit first).
            order = sorted(
                range(self.n_blocks),
                key=lambda i: logits[i, chosen[i]],
            )
            ri = 0
            for i in order:
                if have >= need:
                    break
                if ring_mask[chosen[i]]:
                    continue
                # pick, for this anchor, its best ring fragment not already used
                repl = None
                for vid in ranked[i]:
                    vid = int(vid)
                    if ring_mask[vid] and (not self.dedup or vid not in used):
                        repl = vid
                        break
                if repl is None:  # fall back to any ring id
                    repl = ring_ids[ri % len(ring_ids)]
                    ri += 1
                used.discard(chosen[i])
                chosen[i] = repl
                used.add(repl)
                have += 1

        return [self.vocab[vid] for vid in chosen]

    def propose_library(self, context: Optional[np.ndarray] = None) -> List[str]:
        """Alias for the unconditional/episode library (train.py substitution)."""
        return self.propose(context)

    # ------------------------------------------------------------------- io
    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        self.vocab.save(d / "vocab.json")
        torch.save(self.vae.state_dict(), d / "vae.pt")
        np.savez(
            d / "proposer.npz",
            n_blocks=self.n_blocks,
            anchor_seed=self.anchor_seed,
            min_ring_fraction=self.min_ring_fraction,
            dedup=self.dedup,
            context_dim=self.vae.context_dim,
            latent_dim=self.vae.latent_dim,
            hidden_dim=self.vae.hidden_dim,
            desc_dim=self.vae.desc_dim,
        )

    @classmethod
    def from_spec(cls, spec: dict) -> "VAEProposer":
        """Build (or pass through) a proposer from a lightweight, picklable spec.

        ``spec`` is either an existing :class:`VAEProposer` under key
        ``"instance"``, or a dict with ``"dir"`` pointing at a :meth:`save`
        directory. This indirection keeps ``env_config`` cheap to ship to RLlib
        env-runners, each loading the model lazily.
        """
        if isinstance(spec, VAEProposer):
            return spec
        if "instance" in spec and isinstance(spec["instance"], VAEProposer):
            return spec["instance"]
        d = Path(spec["dir"])
        vocab = FragmentVocab.load(d / "vocab.json")
        with np.load(d / "proposer.npz") as meta:
            vae = FragmentVAE(
                vocab_size=len(vocab),
                context_dim=int(meta["context_dim"]),
                latent_dim=int(meta["latent_dim"]),
                # Preserve compatibility with older saves.
                hidden_dim=int(meta["hidden_dim"]) if "hidden_dim" in meta else 64,
                desc_dim=(
                    int(meta["desc_dim"])
                    if "desc_dim" in meta
                    else vocab.descriptors.shape[1]
                ),
            )
            vae.load_state_dict(
                torch.load(d / "vae.pt", map_location="cpu", weights_only=True)
            )
            return cls(
                vae=vae,
                vocab=vocab,
                n_blocks=int(meta["n_blocks"]),
                anchor_seed=int(meta["anchor_seed"]),
                min_ring_fraction=float(meta["min_ring_fraction"]),
                dedup=bool(meta["dedup"]),
            )
