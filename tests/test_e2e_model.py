"""Tests for the pointer policy net and (if Ray is present) the RLlib wrapper."""

import builtins
import copy
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from moldr.e2e.model import FLOAT_MIN, PointerPolicyNet


def test_pointer_net_shapes():
    net = PointerPolicyNet(obs_dim=16, hidden=32)
    b, a, d = 4, 7, 16
    base = torch.randn(b, d)
    phase = torch.zeros(b, 2)
    phase[:, 0] = 1.0
    options = torch.randn(b, a, d)
    mask = torch.ones(b, a)
    logits, value = net(base, phase, options, mask)
    assert logits.shape == (b, a)
    assert value.shape == (b,)


def test_pointer_net_remains_importable_without_ray(monkeypatch):
    original_import = builtins.__import__

    def import_without_ray(name, *args, **kwargs):
        if name.startswith("ray"):
            raise ModuleNotFoundError("Ray unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ray)
    module_path = Path(__file__).parents[1] / "moldr" / "e2e" / "model.py"
    namespace = runpy.run_path(str(module_path))
    assert isinstance(namespace["PointerPolicyNet"](obs_dim=8), torch.nn.Module)
    with pytest.raises(RuntimeError, match="Ray RLlib is not available"):
        namespace["register_pointer_model"]()


def test_pointer_net_masks_logits():
    net = PointerPolicyNet(obs_dim=8, hidden=16)
    b, a, d = 2, 5, 8
    base = torch.randn(b, d)
    phase = torch.zeros(b, 2)
    options = torch.randn(b, a, d)
    mask = torch.ones(b, a)
    mask[:, 3:] = 0.0  # last two slots invalid
    logits, _ = net(base, phase, options, mask)
    # masked slots must be driven to a very large negative (near the mask fill),
    # well below any real score...
    assert torch.all(logits[:, 3:] <= FLOAT_MIN + 1e3)
    assert torch.all(logits[:, :3] > FLOAT_MIN + 1e3)
    # ...and never selected by argmax or by sampling.
    assert torch.all(logits.argmax(dim=-1) < 3)
    probs = torch.softmax(logits, dim=-1)
    assert torch.all(probs[:, 3:] < 1e-6)


def test_pointer_net_backward():
    net = PointerPolicyNet(obs_dim=8, hidden=16)
    base = torch.randn(3, 8)
    phase = torch.zeros(3, 2)
    options = torch.randn(3, 4, 8)
    mask = torch.ones(3, 4)
    logits, value = net(base, phase, options, mask)
    loss = logits.sum() + value.sum()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pointer_net_low_precision_masks_and_gradients_are_finite(dtype):
    net = PointerPolicyNet(obs_dim=8, hidden=16).to(dtype=dtype)
    base = torch.randn(3, 8, dtype=dtype)
    phase = torch.zeros(3, 2, dtype=dtype)
    options = torch.randn(3, 5, 8, dtype=dtype)
    mask = torch.tensor([[1, 1, 0, 0, 0]] * 3, dtype=dtype)
    logits, value = net(base, phase, options, mask)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    distribution = torch.distributions.Categorical(logits=logits)
    assert torch.equal(distribution.probs[:, 2:], torch.zeros(3, 3))
    loss = -distribution.log_prob(torch.zeros(3, dtype=torch.long)).mean()
    loss = loss - distribution.entropy().mean() + value.float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_factored_scorer_preserves_outputs_and_gradients():
    """Avoiding repeated context GEMMs must retain the original policy."""
    torch.manual_seed(7)
    net = PointerPolicyNet(obs_dim=8, hidden=16)
    reference = copy.deepcopy(net)
    base = torch.randn(3, 8)
    phase = torch.randn(3, 2)
    options = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5)
    logits, value = net(base, phase, options, mask)

    ctx_input = torch.cat([base, phase], dim=-1)
    ctx = reference.ctx_enc(ctx_input).unsqueeze(1).expand(-1, 5, -1)
    opts = reference.opt_enc(options)
    ref_logits = reference.scorer(torch.cat([ctx, opts], dim=-1)).squeeze(-1)
    ref_value = reference.vf(ctx_input).squeeze(-1)
    torch.testing.assert_close(logits, ref_logits)
    torch.testing.assert_close(value, ref_value)
    (logits.square().sum() + value.square().sum()).backward()
    (ref_logits.square().sum() + ref_value.square().sum()).backward()
    for optimized, original in zip(net.parameters(), reference.parameters()):
        torch.testing.assert_close(optimized.grad, original.grad)


# --------------------------------------------------------------------------- #
# Ray-dependent smoke test: build a PPO on the e2e env and run one iteration.
# --------------------------------------------------------------------------- #
def _assert_finite_ppo_update(algo):
    before = {name: value.copy() for name, value in algo.get_policy().get_weights().items()}
    result = algo.train()
    stats = result["info"]["learner"]["default_policy"]["learner_stats"]
    for key in ("total_loss", "policy_loss", "vf_loss", "entropy", "kl"):
        assert np.isfinite(stats[key]), (key, stats[key])
    after = algo.get_policy().get_weights()
    assert all(np.isfinite(value).all() for value in after.values())
    assert any(not np.array_equal(before[name], after[name]) for name in before)


@pytest.mark.parametrize("num_workers", [0, 1])
def test_rllib_ppo_smoke(num_workers):
    pytest.importorskip("ray")
    import ray

    from moldr.e2e.config import get_e2e_config

    class _FakeObjective:
        class _Wrapped:
            @staticmethod
            def score(smiles):
                from moldr.chemutils import get_mol

                mol = get_mol(smiles)
                return 0.0 if mol is None else min(1.0, mol.GetNumAtoms() / 50.0)

        name = "smoke"
        wrapped_objective = _Wrapped()

    ray.init(ignore_reinit_error=True, num_cpus=2, log_to_driver=False)
    try:
        config = get_e2e_config(
            _FakeObjective(),
            building_blocks_smiles=["CC", "CCO", "c1ccccc1", "C1CCCCC1"],
            model_path="models/model_300dim.pkl",
            num_workers=num_workers,  # include serialized remote env execution
            num_gpus=0,
            max_attach=16,
            hidden=32,
            score_threshold=0.3,
            length=20,
            max_episode_steps=24,
        )
        config = config.training(train_batch_size=64, minibatch_size=16, num_epochs=1)
        algo = config.build()
        _assert_finite_ppo_update(algo)
        algo.stop()
    finally:
        ray.shutdown()


def test_rllib_ppo_generative_smoke():
    """End-to-end + generative: PPO on MolEnvE2E whose phase-A pool is produced
    by the VAE proposer (built, trained tiny, run one iteration on CPU)."""
    pytest.importorskip("ray")
    import ray

    from moldr.e2e.config import get_e2e_config
    from moldr.generative.producer import build_proposer_from_smiles

    class _FakeObjective:
        class _Wrapped:
            @staticmethod
            def score(smiles):
                from moldr.chemutils import get_mol

                mol = get_mol(smiles)
                return 0.0 if mol is None else min(1.0, mol.GetNumAtoms() / 50.0)

        name = "smoke-gen"
        wrapped_objective = _Wrapped()

    proposer = build_proposer_from_smiles(
        ["CCO", "CC(=O)O", "c1ccccc1", "C1CCCCC1", "c1ccncc1", "CCN"],
        n_blocks=8,
        hidden_dim=16,
        epochs=5,
    )
    ray.init(ignore_reinit_error=True, num_cpus=2, log_to_driver=False)
    try:
        config = get_e2e_config(
            _FakeObjective(),
            model_path="models/model_300dim.pkl",
            num_workers=0,
            num_gpus=0,
            max_attach=16,
            hidden=32,
            score_threshold=0.3,
            length=20,
            max_episode_steps=24,
            proposer=proposer,
            recond_every="episode",
        )
        config = config.training(train_batch_size=64, minibatch_size=16, num_epochs=1)
        algo = config.build()
        _assert_finite_ppo_update(algo)
        algo.stop()
    finally:
        ray.shutdown()
