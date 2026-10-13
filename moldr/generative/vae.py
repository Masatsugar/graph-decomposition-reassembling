"""Conditional fragment VAE (JT-VAE / HierVAE lineage, distilled).

Architecture (deliberately tiny so a CPU unit-test demo is feasible; scale the
widths up for real GPU training):

    encoder:  fragment_descriptor (DESCRIPTOR_DIM)  --MLP-->  mu, logvar (latent_dim)
    decoder:  [ z (latent_dim) ; context (context_dim) ] --MLP--> logits (|V|)

``context`` is the Mol2Vec 300-d embedding of the *current base molecule* at
decode time (the observation PPO already sees). This is the "decode a fragment
conditioned on the current base embedding" requested by the design brief.

Training is a standard conditional-VAE objective: reconstruct the vocabulary
index of the input fragment (cross-entropy) + KL(q(z|f) || N(0, I)). Because the
output space is the fragment vocabulary, reconstruction is categorical and the
decoder can *only* ever point at a valid, pre-screened fragment.

No RDKit calls happen inside the nn.Module, so forward/backward run anywhere
torch runs (CPU included).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moldr.generative.vocab import FragmentVocab, DESCRIPTOR_DIM


class FragmentVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_dim: int = 300,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        desc_dim: int = DESCRIPTOR_DIM,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.desc_dim = desc_dim

        self.enc = nn.Sequential(
            nn.Linear(desc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def encode(self, desc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(desc)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return logits over the vocabulary for each (z, context) row."""
        return self.dec(torch.cat([z, context], dim=-1))

    def forward(
        self, desc: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(desc)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, context)
        return logits, mu, logvar


def vae_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    *,
    collect_metrics: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Return the differentiable objective and optional host-side metrics."""
    recon = F.cross_entropy(logits, target_ids, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    loss = recon + beta * kl
    # Reading CUDA scalars on the host synchronizes the device. The trainer
    # does not consume these metrics, so keep its entire minibatch on-device.
    parts = {}
    if collect_metrics:
        values = torch.stack((loss, recon, kl)).detach().cpu().tolist()
        parts = dict(zip(("loss", "recon", "kl"), values))
    return loss, parts


def train_fragment_vae(
    vocab: FragmentVocab,
    contexts: Optional[np.ndarray] = None,
    context_dim: int = 300,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
) -> FragmentVAE:
    """Train a FragmentVAE to autoencode the vocabulary.

    Parameters
    ----------
    vocab
        The fragment vocabulary (defines targets and encoder inputs).
    contexts
        Optional ``(M, context_dim)`` array giving a base-molecule context for
        each fragment (e.g. Mol2Vec of the parent molecule the fragment was
        decomposed from). ``None`` trains context-free (zeros), which is enough
        for the unconditional ``propose_library`` use and the CPU demo.

    Notes
    -----
    This is the tiny reference trainer used by the unit tests and small CPU
    runs. For real training on the user GPU, raise ``epochs``/``hidden_dim`` and
    feed real per-fragment contexts; the API is unchanged.
    """
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_dim <= 0:
        raise ValueError("context_dim must be positive")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    torch.manual_seed(seed)

    desc = torch.as_tensor(vocab.descriptors, dtype=torch.float32, device=device)
    target_ids = torch.arange(len(vocab), dtype=torch.long, device=device)
    if contexts is None:
        ctx = torch.zeros((len(vocab), context_dim), dtype=torch.float32, device=device)
    else:
        ctx = torch.as_tensor(contexts, dtype=torch.float32, device=device)
        if ctx.shape != (len(vocab), context_dim):
            raise ValueError(
                f"contexts shape {tuple(ctx.shape)} != {(len(vocab), context_dim)}"
            )
        if not torch.isfinite(ctx).all():
            raise ValueError("contexts must contain only finite values")

    model = FragmentVAE(
        vocab_size=len(vocab),
        context_dim=context_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        desc_dim=vocab.descriptors.shape[1],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(vocab)
    rng = torch.Generator(device=desc.device).manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=rng, device=desc.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits, mu, logvar = model(desc[idx], ctx[idx])
            loss, _ = vae_loss(
                logits, target_ids[idx], mu, logvar, beta=beta, collect_metrics=False
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model
