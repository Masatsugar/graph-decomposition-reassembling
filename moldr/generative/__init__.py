"""Generative building-block proposer for MOLDR.

This package implements a Fragment / JT-VAE-style proposer that *replaces the
fixed gSpan building-block table* with a learned, conditional fragment decoder,
while keeping RLlib PPO's fixed ``Discrete(N)`` action space intact.

Public API
----------
- :class:`moldr.generative.vocab.FragmentVocab`
- :class:`moldr.generative.vae.FragmentVAE`
- :func:`moldr.generative.vae.train_fragment_vae`
- :class:`moldr.generative.proposer.VAEProposer`
- :func:`moldr.generative.producer.write_library_csv`

See module docstrings for the design rationale and the PPO reconciliation.
"""

from moldr.generative.vocab import FragmentVocab, fragment_descriptor, DESCRIPTOR_DIM
from moldr.generative.vae import FragmentVAE, train_fragment_vae, vae_loss
from moldr.generative.proposer import VAEProposer

__all__ = [
    "FragmentVocab",
    "fragment_descriptor",
    "DESCRIPTOR_DIM",
    "FragmentVAE",
    "train_fragment_vae",
    "vae_loss",
    "VAEProposer",
]
