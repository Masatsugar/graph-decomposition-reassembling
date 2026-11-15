"""Generative proposer: save/reload round trip + end-to-end env integration.

These cover the two gaps the proposer's own unit tests left open:
1. the save -> from_spec round trip (the RLlib env-runner shipping path), and
2. feeding the generative pool into the Phase-2 end-to-end env (MolEnvE2E), so
   the pointer policy selects among *generated* building blocks.
"""

import tempfile

import numpy as np
import pytest

from moldr.chemutils import get_mol
from moldr.generative.producer import build_proposer_from_smiles
from moldr.generative.proposer import VAEProposer

MODEL_PATH = "models/model_300dim.pkl"
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
]
N_BLOCKS = 8


def _tiny_proposer(hidden_dim=16, n_blocks=N_BLOCKS):
    return build_proposer_from_smiles(
        SEED_FRAGMENTS, n_blocks=n_blocks, hidden_dim=hidden_dim, epochs=5, seed=0
    )


@pytest.mark.parametrize("hidden_dim", [16, 32, 64])
def test_save_from_spec_round_trip(hidden_dim):
    """Regression: hidden_dim must survive save/reload (it was previously
    dropped, breaking reload for any non-default width)."""
    prop = _tiny_proposer(hidden_dim=hidden_dim)
    d = tempfile.mkdtemp()
    prop.save(d)
    reloaded = VAEProposer.from_spec({"dir": d})
    assert reloaded.n_blocks == prop.n_blocks
    assert reloaded.propose_library(None) == prop.propose_library(None)


def test_from_spec_passthrough_instance():
    prop = _tiny_proposer()
    assert VAEProposer.from_spec(prop) is prop
    assert VAEProposer.from_spec({"instance": prop}) is prop


def test_save_reload_preserves_custom_descriptor_dimension(tmp_path):
    from moldr.generative.vae import FragmentVAE
    from moldr.generative.vocab import FragmentVocab

    vocab = FragmentVocab.from_smiles_list(SEED_FRAGMENTS)
    model = FragmentVAE(len(vocab), desc_dim=9, hidden_dim=16)
    prop = VAEProposer(model, vocab, n_blocks=N_BLOCKS)
    prop.save(tmp_path)
    restored = VAEProposer.from_spec({"dir": tmp_path})
    assert restored.vae.desc_dim == 9
    assert restored.propose_library() == prop.propose_library()


def test_ring_fraction_unsatisfiable_raises():
    """The ring-fraction guarantee must fail fast when the vocabulary has no
    ring fragments, rather than silently returning a ring-free library."""
    from moldr.generative.vocab import FragmentVocab
    from moldr.generative.vae import train_fragment_vae

    acyclic = ["CCO", "CCN", "CC=O", "CCOC", "CN", "CC(=O)O"]
    vocab = FragmentVocab.from_smiles_list(acyclic)
    assert not vocab.ring_mask.any()
    vae = train_fragment_vae(vocab, latent_dim=8, hidden_dim=16, epochs=2, seed=0)

    with pytest.raises(ValueError):
        VAEProposer(vae, vocab, n_blocks=8, min_ring_fraction=0.3)

    # min_ring_fraction=0.0 is satisfiable even with a ring-free vocab.
    prop = VAEProposer(vae, vocab, n_blocks=8, min_ring_fraction=0.0)
    assert len(prop.propose_library(None)) == 8


def test_e2e_env_uses_generated_pool():
    from moldr.e2e.env import MolEnvE2E, Phase

    prop = _tiny_proposer()
    env = MolEnvE2E(
        {
            "BUILDING_BLOCKS": ["C"],  # deliberately wrong; proposer must override
            "SCORE_FUNCTION": lambda s: (
                0.0 if get_mol(s) is None else min(1.0, get_mol(s).GetNumAtoms() / 50.0)
            ),
            "MODEL_PATH": MODEL_PATH,
            "PROPOSER": prop,
            "RECOND_EVERY": "episode",
            "MAX_ATTACH": 16,
        }
    )
    # The proposer owns N and the pool, overriding the passed table.
    assert env.n_bb == N_BLOCKS
    assert len(env.building_blocks) == N_BLOCKS
    assert all(get_mol(s) is not None for s in env.building_blocks)

    obs, _ = env.reset(seed=0)
    assert env.phase == Phase.SELECT_BB
    assert obs["action_mask"].sum() == N_BLOCKS
    # Every building block is valid after the episode re-decode.
    assert all(get_mol(s) is not None for s in env.building_blocks)


def test_e2e_generative_rollout_is_valid():
    from moldr.e2e.env import MolEnvE2E

    prop = _tiny_proposer()
    env = MolEnvE2E(
        {
            "BUILDING_BLOCKS": ["C"],
            "SCORE_FUNCTION": lambda s: (
                0.0 if get_mol(s) is None else min(1.0, get_mol(s).GetNumAtoms() / 50.0)
            ),
            "MODEL_PATH": MODEL_PATH,
            "PROPOSER": prop,
            "RECOND_EVERY": "step",
            "MAX_ATTACH": 16,
        }
    )
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    done = False
    steps = 0
    while not done and steps < 20:
        valid = np.flatnonzero(obs["action_mask"] > 0)
        obs, reward, done, trunc, info = env.step(int(rng.choice(valid)))
        steps += 1
        assert get_mol(env.base_smiles) is not None
        assert np.isfinite(reward)
        # Pool stays size-N and valid even as it is re-decoded per step.
        assert len(env.building_blocks) == N_BLOCKS
        assert all(get_mol(s) is not None for s in env.building_blocks)
