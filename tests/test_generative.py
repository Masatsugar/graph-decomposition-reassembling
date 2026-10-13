"""Unit tests for the generative (Fragment/JT-VAE) building-block proposer.

All tests run on CPU with a *tiny untrained-or-lightly-trained* model and a
handful of fragments -- no GPU, no network, no Ray. The PPO env contract is
exercised directly via reset()/step() (which is exactly what PPO consumes),
with Mol2Vec monkeypatched to a zero embedder so the tests need no gensim model.
"""

import numpy as np
import pytest

from moldr.chemutils import get_mol, get_smiles
from moldr.core.reassemble import merge_node, merge_edge, num_rings
from moldr.core.molgraph import sanitize_molgraph
from moldr.generative.vocab import (
    FragmentVocab,
    fragment_descriptor,
    is_valid_block,
    DESCRIPTOR_DIM,
)
from moldr.generative.vae import FragmentVAE, train_fragment_vae, vae_loss
from moldr.generative.proposer import VAEProposer
from moldr.generative import producer


# A small mix of chain and ring fragments (all multi-atom, open-valence).
SEED_FRAGMENTS = [
    "CCO",
    "CC(=O)O",
    "C#CC",
    "CN",
    "CCN",
    "CC=O",
    "c1ccccc1",
    "C1CCCCC1",
    "c1ccncc1",
    "C1CCNCC1",
    "C1=CCCCC1",
    "CCOC",
]

N_BLOCKS = 16


def _make_proposer(n_blocks=N_BLOCKS, epochs=10, min_ring_fraction=0.3):
    vocab = FragmentVocab.from_smiles_list(SEED_FRAGMENTS)
    vae = train_fragment_vae(
        vocab, latent_dim=8, hidden_dim=16, epochs=epochs, seed=0
    )
    return VAEProposer(
        vae=vae,
        vocab=vocab,
        n_blocks=n_blocks,
        anchor_seed=0,
        min_ring_fraction=min_ring_fraction,
    ), vocab


# --------------------------------------------------------------------------- #
# 1. Vocabulary validity contract
# --------------------------------------------------------------------------- #
def test_vocab_entries_satisfy_block_contract():
    vocab = FragmentVocab.from_smiles_list(SEED_FRAGMENTS + ["C", "N", "O", "bad[["])
    assert len(vocab) >= 8
    for s in vocab.smiles:
        assert is_valid_block(s)              # parses + kekulizes + multi-atom
        m = get_mol(s)
        assert m is not None
        assert m.GetNumAtoms() >= 2           # len>1 / multi-atom (train.py:146)
        assert any(a.GetImplicitValence() > 0 for a in m.GetAtoms())  # node-mergeable
    # single atoms and junk were filtered out
    assert "C" not in vocab.smiles and "N" not in vocab.smiles
    # descriptor shape is as advertised
    assert vocab.descriptors.shape == (len(vocab), DESCRIPTOR_DIM)
    assert fragment_descriptor(get_mol("c1ccccc1")).shape == (DESCRIPTOR_DIM,)


def test_vocab_from_molecules_decomposes():
    # Full molecules -> JT fragments; result is still a valid vocab.
    mols = ["CC1CCC2=CC=CC=C2O1", "c1ccccc1CCN", "CC1C(=O)NC(=O)S1"]
    vocab = FragmentVocab.from_molecules(mols)
    assert len(vocab) >= 2
    assert all(is_valid_block(s) for s in vocab.smiles)
    assert vocab.ring_mask.any()  # decomposition surfaced at least one ring


# --------------------------------------------------------------------------- #
# 2. Proposer output: fixed N, valid, stable, ring-guaranteed
# --------------------------------------------------------------------------- #
def test_proposer_emits_exactly_n_valid_blocks():
    proposer, _ = _make_proposer()
    lib = proposer.propose_library()          # context-free
    assert len(lib) == N_BLOCKS
    for s in lib:
        assert get_mol(s) is not None         # env.py:28 can never get None
        assert len(s) > 1


def test_proposer_is_deterministic_and_positions_stable():
    proposer, _ = _make_proposer()
    ctx = np.zeros(300, dtype=np.float32)
    a = proposer.propose(ctx)
    b = proposer.propose(ctx)
    assert a == b                             # same anchors+context -> same library
    # Each anchor index maps to a stable position (semantics PPO can learn).
    assert len(a) == N_BLOCKS


def test_proposer_guarantees_ring_fraction():
    proposer, _ = _make_proposer(min_ring_fraction=0.5)
    lib = proposer.propose_library()
    n_ring = sum(1 for s in lib if num_rings(get_mol(s)) >= 1)
    assert n_ring >= int(np.ceil(0.5 * N_BLOCKS))


def test_proposer_conditioning_can_change_pool():
    # Needs a trained decoder that actually uses context; train a bit longer with
    # per-fragment contexts so the context weights are non-trivial.
    vocab = FragmentVocab.from_smiles_list(SEED_FRAGMENTS)
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(len(vocab), 300)).astype(np.float32)
    vae = train_fragment_vae(
        vocab, contexts=contexts, latent_dim=8, hidden_dim=16, epochs=200, seed=0
    )
    proposer = VAEProposer(vae, vocab, n_blocks=N_BLOCKS, min_ring_fraction=0.0, dedup=False)
    lib0 = proposer.propose(np.zeros(300, dtype=np.float32))
    lib1 = proposer.propose(np.ones(300, dtype=np.float32) * 3.0)
    # Different base context => at least one position decodes differently.
    assert lib0 != lib1
    # ...and both libraries are still fully valid.
    assert all(get_mol(s) is not None for s in lib0 + lib1)


# --------------------------------------------------------------------------- #
# 3. Merge compatibility: blocks are broadly mergeable (not dead ends)
# --------------------------------------------------------------------------- #
def _products(base_smiles, frag_smiles):
    base = get_mol(base_smiles)
    frag = get_mol(frag_smiles)
    node = merge_node(base, frag)
    edge = merge_edge(base, frag) or []
    prods = sanitize_molgraph(list(node) + list(edge))
    return [get_smiles(m) for m in prods], len(node), len(edge)


def test_every_proposed_block_merges_against_simple_base():
    proposer, _ = _make_proposer()
    lib = proposer.propose_library()
    # Against a trivial "C" base, node-merge must fire for every open-valence
    # block => no proposed action is an immediate dead end.
    for s in lib:
        smis, n_node, _ = _products("C", s)
        assert n_node > 0, f"block {s} produced no node-merge candidates"
        assert len(smis) > 0, f"block {s} produced no valid products"


def test_ring_blocks_enable_edge_merge_against_ring_base():
    vocab = FragmentVocab.from_smiles_list(SEED_FRAGMENTS)
    ring_frags = [s for s in vocab.smiles if num_rings(get_mol(s)) >= 1]
    assert ring_frags
    saw_edge = False
    for s in ring_frags:
        _, _, n_edge = _products("c1ccccc1", s)  # ring base + ring frag
        if n_edge > 0:
            saw_edge = True
    assert saw_edge, "no ring fragment produced an edge-merge (ring fusion)"


# --------------------------------------------------------------------------- #
# 4. Env integration: reset/step honor the Discrete(N) contract
# --------------------------------------------------------------------------- #
class _ZeroMol2Vec:
    """Stand-in for Mol2Vec: no gensim/model file needed for the unit test."""

    def __init__(self, *a, **k):
        pass

    def fit_transform(self, mols):
        return np.zeros((len(mols), 300), dtype=np.float32)


def _scoring_function(smiles):
    m = get_mol(smiles)
    return len(m.GetAtoms()) / 100.0 if m is not None else 0.0


def _env_config(proposer, recond_every="episode"):
    import gymnasium as gym

    high = np.array([np.finfo(np.float32).max for _ in range(300)])
    init_lib = proposer.propose_library()
    return {
        "ACTION_SPACE": gym.spaces.Discrete(len(init_lib)),
        "OBS_SPACE": gym.spaces.Box(-high, high, dtype=np.float32),
        "BUILDING_BLOCKS": init_lib,
        "SCORE_FUNCTION": _scoring_function,
        "SCORE_THRESHOLD": 0.9,
        "FINAL_WEIGHT": 1.0,
        "BASE_SMILES": "C",
        "MODEL_PATH": "unused",
        "LENGTH": 100,
        "PROPOSER": proposer,
        "RECOND_EVERY": recond_every,
    }


def test_env_reset_and_step_with_proposer(monkeypatch):
    import moldr.env as env_mod

    monkeypatch.setattr(env_mod, "Mol2Vec", _ZeroMol2Vec)
    proposer, _ = _make_proposer()
    env = env_mod.MolEnvValueMax(_env_config(proposer, recond_every="episode"))

    obs, info = env.reset()
    assert obs.shape == (300,)
    # Pool is exactly N, all valid, N == action space.
    assert len(env.mols) == env.action_space.n == N_BLOCKS
    assert all(m is not None for m in env.mols)

    obs, reward, done, trunc, info = env.step(0)
    assert obs.shape == (300,)
    assert np.isfinite(reward)
    assert isinstance(done, bool)


def test_env_without_proposer_is_unchanged(monkeypatch):
    import moldr.env as env_mod

    monkeypatch.setattr(env_mod, "Mol2Vec", _ZeroMol2Vec)
    import gymnasium as gym

    high = np.array([np.finfo(np.float32).max for _ in range(300)])
    blocks = ["CCO", "c1ccccc1", "CC(=O)O"]
    cfg = {
        "ACTION_SPACE": gym.spaces.Discrete(len(blocks)),
        "OBS_SPACE": gym.spaces.Box(-high, high, dtype=np.float32),
        "BUILDING_BLOCKS": blocks,
        "SCORE_FUNCTION": _scoring_function,
        "SCORE_THRESHOLD": 0.9,
        "FINAL_WEIGHT": 1.0,
        "BASE_SMILES": "C",
        "MODEL_PATH": "unused",
        "LENGTH": 100,
    }
    env = env_mod.MolEnvValueMax(cfg)
    env.reset()
    # No PROPOSER key => static table, identical to legacy behavior.
    assert env._proposer is None
    assert [get_smiles(m) for m in env.mols] == [get_smiles(get_mol(b)) for b in blocks]


# --------------------------------------------------------------------------- #
# 5. VAE training smoke test (CPU, a few steps, finite loss)
# --------------------------------------------------------------------------- #
def test_vae_training_step_is_finite_and_learns():
    import torch

    vocab = FragmentVocab.from_smiles_list(SEED_FRAGMENTS)
    model = FragmentVAE(vocab_size=len(vocab), latent_dim=8, hidden_dim=16)
    desc = torch.as_tensor(vocab.descriptors)
    ctx = torch.zeros((len(vocab), 300))
    tgt = torch.arange(len(vocab))

    logits, mu, logvar = model(desc, ctx)
    loss0, parts = vae_loss(logits, tgt, mu, logvar)
    assert np.isfinite(parts["loss"]) and np.isfinite(parts["kl"])

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(100):
        logits, mu, logvar = model(desc, ctx)
        loss, _ = vae_loss(logits, tgt, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Reconstruction improved (tiny vocab is easily memorized).
    logits, mu, logvar = model(desc, ctx)
    _, parts2 = vae_loss(logits, tgt, mu, logvar)
    assert parts2["recon"] < parts["recon"]


# --------------------------------------------------------------------------- #
# 6. Producer CSV is drop-in equivalent to the gSpan table
# --------------------------------------------------------------------------- #
def test_write_library_csv_matches_gspan_shape(tmp_path):
    import pandas as pd

    proposer, _ = _make_proposer()
    out = producer.write_library_csv(proposer, tmp_path / "lib.csv")
    # Re-read EXACTLY as train.py does (train.py:140 / :146).
    blocks = pd.read_csv(out, index_col=0).values.flatten()
    blocks = [smi for smi in blocks if len(smi) > 1]
    assert len(blocks) == N_BLOCKS
    assert all(get_mol(s) is not None for s in blocks)


def test_build_proposer_from_smiles_end_to_end():
    proposer = producer.build_proposer_from_smiles(
        SEED_FRAGMENTS, n_blocks=N_BLOCKS, epochs=5
    )
    lib = producer.propose_building_blocks(proposer)
    assert len(lib) == N_BLOCKS
    assert all(get_mol(s) is not None for s in lib)
