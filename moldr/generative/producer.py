"""Producer: generative replacement for the gSpan building-block table.

Two integration modes are supported, matching the scout's integration points:

(a) *Write an equivalent CSV* (``write_library_csv``) -- one-column CSV of
    canonical kekule SMILES, identical in shape to
    ``MolsMining.save_csv`` (decompose.py:134). ``train.py`` is then completely
    untouched: it still does ``pd.read_csv(smi_file, index_col=0)`` and the
    ``len(smi) > 1`` filter. Point ``smi_file`` at the produced CSV.

(b) *In-memory* (``propose_building_blocks``) -- return a ``list[str]`` straight
    into ``get_default_config_v1(...)`` in place of the CSV read, for the fixed
    (static) library variant.

Both go through :class:`VAEProposer`, so every emitted block is valid and the
ring-fraction guarantee holds.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from moldr.generative.proposer import VAEProposer
from moldr.generative.vae import train_fragment_vae
from moldr.generative.vocab import FragmentVocab


def build_proposer_from_smiles(
    smiles: Sequence[str],
    n_blocks: int,
    decompose: bool = False,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 50,
    anchor_seed: int = 0,
    min_ring_fraction: float = 0.3,
    context_dim: int = 300,
    seed: int = 0,
) -> VAEProposer:
    """End-to-end: SMILES -> vocab -> trained FragmentVAE -> VAEProposer.

    ``decompose=True`` extracts junction-tree fragments from full molecules;
    ``False`` treats the inputs as candidate fragments directly (e.g. when
    seeding from the existing building-block CSVs).
    """
    vocab = (
        FragmentVocab.from_molecules(smiles)
        if decompose
        else FragmentVocab.from_smiles_list(smiles)
    )
    vae = train_fragment_vae(
        vocab,
        context_dim=context_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        seed=seed,
    )
    return VAEProposer(
        vae=vae,
        vocab=vocab,
        n_blocks=n_blocks,
        anchor_seed=anchor_seed,
        min_ring_fraction=min_ring_fraction,
    )


def propose_building_blocks(
    proposer: VAEProposer, context: Optional[np.ndarray] = None
) -> List[str]:
    """Fixed library (list[str]) for the direct in-memory substitution."""
    blocks = proposer.propose_library(context)
    return [s for s in blocks if len(s) > 1]  # replicate train.py:146 filter


def write_library_csv(
    proposer: VAEProposer,
    out_path: str | Path,
    context: Optional[np.ndarray] = None,
) -> Path:
    """Write a gSpan-equivalent one-column CSV (index + column ``"0"``)."""
    blocks = propose_building_blocks(proposer, context)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(blocks).to_csv(out_path)  # same call shape as decompose.py:134
    return out_path
