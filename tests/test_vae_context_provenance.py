"""Context-free VAE weights must never masquerade as a conditional decoder."""

import numpy as np
import pytest
import torch

from moldr.generative.producer import build_proposer_from_smiles
from moldr.generative.proposer import VAEProposer
from moldr.generative.vae import FragmentVAE, train_fragment_vae
from moldr.generative.vocab import FragmentVocab

FRAGMENTS = ["CC", "CO", "C1CC1", "CCN"]


@pytest.fixture
def vocab():
    return FragmentVocab.from_smiles_list(FRAGMENTS)


def train(vocab, contexts=None, epochs=2, **kwargs):
    return train_fragment_vae(
        vocab,
        contexts=contexts,
        context_dim=3,
        latent_dim=4,
        hidden_dim=8,
        epochs=epochs,
        seed=2,
        **kwargs,
    )


def proposer(model, vocab):
    return VAEProposer(model, vocab, n_blocks=4, dedup=False, min_ring_fraction=0)


@pytest.mark.parametrize("context_kind", ["missing", "zeros"])
def test_context_free_training_records_false_and_keeps_context_weights_untrained(
    vocab, context_kind
):
    contexts = (
        None
        if context_kind == "missing"
        else np.zeros((len(vocab), 3), dtype=np.float32)
    )
    untrained = train(vocab, contexts=contexts, epochs=0)
    trained = train(vocab, contexts=contexts)
    assert not trained.context_trained
    torch.testing.assert_close(
        trained.dec[0].weight[:, 4:], untrained.dec[0].weight[:, 4:], rtol=0, atol=0
    )
    prop = proposer(trained, vocab)
    assert not prop.context_trained
    np.testing.assert_array_equal(prop._logits(np.ones(3) * 100), prop._logits(None))
    assert prop.propose(np.ones(3) * 100) == prop.propose(None)


def test_aligned_nonzero_context_training_records_true_and_updates_context_weights(
    vocab,
):
    contexts = np.random.default_rng(2).normal(size=(len(vocab), 3)).astype(np.float32)
    untrained = train(vocab, contexts=contexts, epochs=0)
    trained = train(vocab, contexts=contexts)
    assert trained.context_trained
    assert not torch.equal(trained.dec[0].weight[:, 4:], untrained.dec[0].weight[:, 4:])
    prop = proposer(trained, vocab)
    assert prop.context_trained
    assert not np.array_equal(prop._logits(np.ones(3) * 10), prop._logits(None))


@pytest.mark.parametrize("epochs,lr", [(0, 1e-3), (2, 0)])
def test_contexts_without_parameter_optimization_do_not_claim_training(
    vocab, epochs, lr
):
    model = train(vocab, contexts=np.ones((len(vocab), 3)), epochs=epochs, lr=lr)
    assert not model.context_trained
    prop = proposer(model, vocab)
    np.testing.assert_array_equal(prop._logits(np.ones(3)), prop._logits(None))


@pytest.mark.parametrize("conditional", [False, True])
def test_proposer_save_reload_preserves_context_training_provenance(
    vocab, conditional, tmp_path
):
    contexts = np.ones((len(vocab), 3)) if conditional else None
    model = train(vocab, contexts=contexts)
    prop = proposer(model, vocab)
    prop.save(tmp_path)
    with np.load(tmp_path / "proposer.npz") as metadata:
        assert bool(metadata["context_trained"]) == conditional
    restored = VAEProposer.from_spec({"dir": tmp_path})
    assert restored.context_trained == conditional
    assert restored.vae.context_trained == conditional
    np.testing.assert_array_equal(
        restored._logits(np.ones(3)), prop._logits(np.ones(3))
    )
    assert set(restored.vae.state_dict()) == set(model.state_dict())
    assert not any("context_trained" in name for name in model.state_dict())


def test_old_save_without_provenance_conservatively_restores_context_free(
    vocab, tmp_path
):
    prop = proposer(train(vocab, contexts=np.ones((len(vocab), 3))), vocab)
    prop.save(tmp_path)
    with np.load(tmp_path / "proposer.npz") as metadata:
        values = {
            key: metadata[key] for key in metadata.files if key != "context_trained"
        }
    np.savez(tmp_path / "proposer.npz", **values)
    restored = VAEProposer.from_spec({"dir": tmp_path})
    assert not restored.context_trained
    np.testing.assert_array_equal(
        restored._logits(np.ones(3) * 100), restored._logits(None)
    )
    assert restored.propose(None) == prop.propose(None)


def test_custom_trainer_can_explicitly_record_provenance_without_state_dict_change(
    vocab, tmp_path
):
    model = FragmentVAE(len(vocab), context_dim=3, hidden_dim=8, context_trained=True)
    prop = proposer(model, vocab)
    assert prop.context_trained
    model.context_trained = False
    assert not prop.context_trained
    model.context_trained = True
    prop.save(tmp_path)
    assert VAEProposer.from_spec({"dir": tmp_path}).context_trained


@pytest.mark.parametrize(
    "conditional,epochs,expected",
    [(False, 2, False), (True, 0, False), (True, 2, True)],
)
def test_producer_propagates_training_provenance(conditional, epochs, expected):
    contexts = np.ones((len(FRAGMENTS), 3)) if conditional else None
    prop = build_proposer_from_smiles(
        FRAGMENTS,
        n_blocks=4,
        contexts=contexts,
        context_dim=3,
        hidden_dim=8,
        epochs=epochs,
        seed=2,
        min_ring_fraction=0,
    )
    assert prop.context_trained == expected
    assert prop.vae.context_trained == expected


def test_ignored_inference_context_still_preserves_old_validation_contract(vocab):
    prop = proposer(train(vocab), vocab)
    with pytest.raises(ValueError, match="context dim"):
        prop.propose(np.ones(4))
    with pytest.raises(ValueError, match="finite"):
        prop.propose(np.full(3, np.nan))
