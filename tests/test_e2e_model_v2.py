import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
from rdkit import Chem
from torch.nn import functional as F

from moldr.e2e.model import FLOAT_MIN
from moldr.e2e.model_v2 import E2EPolicyNetV2, MoldrPointerModelV2
from moldr.gnn import MPNNEncoder, graph_record


def _vector_obs():
    torch.manual_seed(7)
    return {
        "base": torch.randn(3, 8), "options": torch.randn(3, 4, 8),
        "phase": torch.tensor([[1., 0.], [0., 1.], [0., 1.]]),
        "action_mask": torch.tensor([[1., 1., 0., 0.]] * 3),
        "context": torch.tensor([[.2, 0., 10.], [.3, 1., 9.], [.4, 2., 8.]]),
    }


def _graph_obs():
    def records(smiles):
        return [graph_record(Chem.MolFromSmiles(s), 8, 10) for s in smiles]

    bases = records(["C", "CCO"])
    candidates = records(["CC", "CO"])
    obs = {
        "phase": torch.tensor([[1., 0.], [0., 1.]]),
        "action_mask": torch.tensor([[1., 1., 1., 0., 0.], [1., 1., 0., 0., 0.]]),
        "context": torch.tensor([[.2, 0., 10.], [.3, 1., 9.]]),
        "fragment_ids": torch.tensor([[0, 1, 2, -1, -1], [-1] * 5]),
    }
    for key in ("atoms", "bonds", "counts"):
        obs["base_" + key] = torch.tensor(np.stack([r[key] for r in bases]))
        packed = np.stack([r[key] for r in candidates])
        obs["candidate_" + key] = torch.tensor(np.stack([np.zeros_like(packed), packed]))
    return obs


def _graph_net(trainable, **extra):
    return E2EPolicyNetV2(
        representation="mpnn", trainable=trainable, graph_hidden=16, graph_output=8,
        hidden=16, max_attach=2, atom_chunk_size=4,
        catalog_smiles=["CC", "CO", "C1CC1"], **extra,
    )


def test_standard_actor_has_no_candidate_head_and_masks_are_exact():
    net = E2EPolicyNetV2(obs_dim=8, hidden=16)
    obs = _vector_obs()
    logits, value = net(obs)
    assert net.heads.candidate_value_head is None
    assert logits.shape == (3, 4) and value.shape == (3,)
    assert torch.equal(logits[:, 2:], torch.full((3, 2), FLOAT_MIN))
    dist = torch.distributions.Categorical(logits=logits)
    assert torch.equal(dist.probs[:, 2:], torch.zeros(3, 2))
    # PPO sees exactly the logits from action selection, including the mask.
    selected = dist.sample()
    logp = dist.log_prob(selected)
    replay_logits, _ = net(obs)
    torch.testing.assert_close(logp, torch.distributions.Categorical(logits=replay_logits).log_prob(selected))


def test_actor_value_only_changes_phase_b_and_stops_value_gradient():
    obs = _vector_obs()
    net = E2EPolicyNetV2(obs_dim=8, hidden=16, candidate_value=True, alpha=2, temperature=3)
    actor = copy.deepcopy(net)
    actor.heads.alpha = 0
    logits, _ = net(obs)
    actor_logits, _ = actor(obs)
    u = net.candidate_values(obs)
    torch.testing.assert_close(logits[0], actor_logits[0])
    torch.testing.assert_close(logits[1:, :2], actor_logits[1:, :2] + 2 * u[1:, :2] / 3)
    assert torch.equal(logits[:, 2:], torch.full((3, 2), FLOAT_MIN))
    loss = -torch.distributions.Categorical(logits=logits).log_prob(torch.zeros(3, dtype=torch.long)).mean()
    loss.backward()
    assert all(p.grad is None for p in net.heads.candidate_value_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.heads.scorer.parameters())


def test_auxiliary_matches_selected_phase_b_huber_and_ignores_phase_a_targets():
    obs = _vector_obs()
    net = E2EPolicyNetV2(obs_dim=8, hidden=16, candidate_value=True, alpha=0)
    actions = torch.tensor([0, 1, 0])
    targets = torch.tensor([1000., 2., 3.])
    u = net.candidate_values(obs)
    expected = .5 * F.smooth_l1_loss(torch.stack([u[1, 1], u[2, 0]]), targets[1:])
    loss = net.auxiliary_loss(obs, actions, targets)
    torch.testing.assert_close(loss, expected)
    altered_targets = targets.clone()
    altered_targets[0] = -1000
    torch.testing.assert_close(net.auxiliary_loss(obs, actions, altered_targets), expected)
    gradients = torch.autograd.grad(expected, u)[0]
    assert torch.count_nonzero(gradients[0]) == 0
    assert gradients[1, 0] == 0 and gradients[2, 1] == 0
    assert gradients[1, 1] != 0 and gradients[2, 0] != 0
    loss.backward()
    assert any(p.grad is not None for p in net.heads.candidate_value_head.parameters())
    assert all(p.grad is None for p in net.heads.scorer.parameters())


def test_auxiliary_without_phase_b_is_differentiable_zero():
    obs = _vector_obs()
    obs["phase"][:] = torch.tensor([1., 0.])
    net = E2EPolicyNetV2(obs_dim=8, hidden=16, candidate_value=True)
    loss = net.auxiliary_loss(obs, torch.zeros(3, dtype=torch.long), torch.ones(3))
    assert loss.item() == 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.heads.candidate_value_head.parameters())


def test_rich_context_ignores_padded_options():
    obs = _vector_obs()
    net = E2EPolicyNetV2(obs_dim=8, hidden=16)
    logits, value = net(obs)
    obs["options"][:, 2:] = 1e8
    other_logits, other_value = net(obs)
    torch.testing.assert_close(logits, other_logits)
    torch.testing.assert_close(value, other_value)


def test_rich_context_masked_mean_max_handles_negative_and_empty_options():
    net = E2EPolicyNetV2(obs_dim=8, hidden=16)
    obs = _vector_obs()
    obs["options"].fill_(-3)
    obs["options"][:, 1] = -1
    obs["action_mask"][1] = 0
    features, _, _ = net.heads._features(obs["base"], obs["phase"], obs["options"], obs["action_mask"], obs["context"])
    torch.testing.assert_close(features[0, -16:-8], torch.full((8,), -2.))
    torch.testing.assert_close(features[0, -8:], torch.full((8,), -1.))
    assert torch.equal(features[1, -16:], torch.zeros(16))
    assert net.heads.context_aggregation == "mean_max"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_mask_loss_and_gradient_are_finite(dtype):
    net = E2EPolicyNetV2(obs_dim=8, hidden=16, candidate_value=True).to(dtype=dtype)
    obs = {k: v.to(dtype) for k, v in _vector_obs().items()}
    logits, value = net(obs)
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    actions = torch.tensor([0, 1, 0])
    loss = -torch.distributions.Categorical(logits=logits).log_prob(actions).mean() + value.float().square().mean()
    loss = loss + net.auxiliary_loss(obs, actions, torch.ones(3, dtype=dtype))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_wrapper_custom_loss_recomputes_training_batch_and_preserves_loss_list():
    pytest.importorskip("ray")
    obs = _vector_obs()
    space = gym.spaces.Dict({
        key: gym.spaces.Box(-1e4, 1e4, shape=tuple(value.shape[1:]), dtype=np.float32)
        for key, value in obs.items()
    })
    model = MoldrPointerModelV2(space, gym.spaces.Discrete(4), 4, {"custom_model_config": {
        "obs_dim": 8, "hidden": 16, "candidate_value": True,
    }}, "test_v2")
    actions, targets = torch.tensor([0, 1, 0]), torch.tensor([1., 2., 3.])
    expected = model.net.auxiliary_loss(obs, actions, targets)
    unrelated = {k: v.clone() for k, v in obs.items()}
    unrelated["base"] += 100
    model.forward({"obs": unrelated}, [], None)
    value_before = model.value_function().clone()
    losses = model.custom_loss([torch.tensor(2.), torch.tensor(3.)], {"obs": obs, "actions": actions, "value_targets": targets})
    assert isinstance(losses, list) and len(losses) == 2
    torch.testing.assert_close(losses[0], 2 + expected)
    torch.testing.assert_close(losses[1], 3 + expected)
    torch.testing.assert_close(model.value_function(), value_before)
    assert isinstance(model.metrics()["candidate_value_loss"], float)


@pytest.mark.parametrize("trainable", [False, True])
def test_mpnn_policy_encoder_updates_only_when_trainable(trainable):
    obs = _graph_obs()
    net = _graph_net(trainable, candidate_value=True, gradient_checkpointing=True)
    before = {name: p.detach().clone() for name, p in net.encoder.named_parameters()}
    optimizer = torch.optim.Adam(net.parameters(), lr=.01)
    logits, value = net(obs)
    actions = torch.tensor([1, 0])
    loss = -torch.distributions.Categorical(logits=logits).log_prob(actions).mean() + value.square().mean()
    loss = loss + net.auxiliary_loss(obs, actions, torch.ones(2))
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    optimizer.step()
    changed = any(not torch.equal(before[name], p) for name, p in net.encoder.named_parameters())
    assert changed is trainable
    if not trainable:
        assert all(p.grad is None for p in net.encoder.parameters())
    assert obs["candidate_atoms"].shape[1] == 2 < obs["action_mask"].shape[1]


def test_frozen_catalog_cache_is_invalidated_after_weight_restore():
    net = _graph_net(False, embedding_cache_size=3)
    net(_graph_obs())
    assert net._catalog_cached.all()
    net.load_state_dict(net.state_dict())
    assert not net._catalog_cached.any()


def test_graph_records_include_charge_isotope_hydrogen_aromaticity_and_do_not_mutate():
    mol = Chem.MolFromSmiles("[13CH3][NH3+].c1ccccc1")
    before = Chem.MolToSmiles(mol)
    record = graph_record(mol, 12, 12)
    assert all(array.dtype == np.int32 for array in record.values())
    assert record["atoms"][0, 2] == 13
    assert record["atoms"][0, 3] == 3
    assert record["atoms"][1, 1] == 1
    assert record["atoms"][2:8, 4].all()
    assert Chem.MolToSmiles(mol) == before
    with pytest.raises(ValueError, match="never truncated"):
        graph_record(mol, 2, 2)


def test_graph_integer_features_reject_overflow_instead_of_wrapping():
    mol = Chem.MolFromSmiles("C")

    class HugeIsotope:
        def GetIsotope(self):
            return 2**35

        def __getattr__(self, name):
            return getattr(mol.GetAtomWithIdx(0), name)

    class Records:
        def GetNumAtoms(self):
            return 1

        def GetNumBonds(self):
            return 0

        def GetAtoms(self):
            return [HugeIsotope()]

        def GetBonds(self):
            return []

    with pytest.raises(ValueError, match="features are never truncated"):
        graph_record(Records(), 1, 0)


def test_chunked_checkpointed_encoder_preserves_full_graph_and_gradients():
    records = [graph_record(Chem.MolFromSmiles(s), 14, 16) for s in ["CCCCCCCCCC", "c1ccccc1", "C"]]
    tensors = {k: torch.tensor(np.stack([r[k] for r in records])) for k in ("atoms", "bonds", "counts")}
    reference = MPNNEncoder(hidden=16, output_dim=8, atom_chunk_size=4096)
    chunked = MPNNEncoder(hidden=16, output_dim=8, atom_chunk_size=3, gradient_checkpointing=True)
    chunked.load_state_dict(reference.state_dict())
    expected = reference(**tensors)
    actual = chunked(**tensors)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for a, b in zip(reference.parameters(), chunked.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-4)
    tensors["counts"][0, 0] = 15
    with pytest.raises(ValueError, match="never truncated"):
        chunked(**tensors)


def test_mpnn_uses_bond_attributes():
    records = [graph_record(Chem.MolFromSmiles("CC"), 2, 1)] * 2
    atoms = torch.tensor(np.stack([r["atoms"] for r in records]))
    bonds = torch.tensor(np.stack([r["bonds"] for r in records]))
    counts = torch.tensor(np.stack([r["counts"] for r in records]))
    bonds[1, 0, 2] = 2
    encoder = MPNNEncoder(hidden=16, output_dim=8)
    output = encoder(atoms, bonds, counts)
    assert not torch.allclose(output[0], output[1])


def test_int32_and_int64_graph_inputs_have_identical_outputs_and_gradients():
    record = graph_record(Chem.MolFromSmiles("[13CH3][NH3+]"), 8, 10)
    args32 = {key: torch.as_tensor(value).unsqueeze(0) for key, value in record.items()}
    encoder = MPNNEncoder(hidden=16, output_dim=8)
    reference = copy.deepcopy(encoder)
    output32 = encoder(**args32)
    output64 = reference(**{key: value.long() for key, value in args32.items()})
    torch.testing.assert_close(output32, output64, rtol=0, atol=0)
    output32.sum().backward()
    output64.sum().backward()
    for a, b in zip(encoder.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_trainable_mol2vec_rejected():
    with pytest.raises(ValueError, match="trainable Mol2Vec is unsupported"):
        E2EPolicyNetV2(trainable=True)


def test_kl_diagnostics_are_exact_cumulative_counts_outside_weight_state():
    model = MoldrPointerModelV2(
        gym.spaces.Box(-1, 1, (8,)), gym.spaces.Discrete(4), 4,
        {"custom_model_config": {"obs_dim": 8, "hidden": 16}}, "kl_reporting",
    )
    weights = set(model.state_dict())
    assert model.kl_diagnostics() == {
        "finite_logit_underflow_terms": 0, "mask_mismatch_rows": 0,
    }
    model._record_kl_diagnostics(torch.tensor(2), torch.tensor(0))
    model._record_kl_diagnostics(torch.tensor(3), torch.tensor(1))
    assert model.kl_diagnostics() == {
        "finite_logit_underflow_terms": 5, "mask_mismatch_rows": 1,
    }
    assert set(model.state_dict()) == weights
    assert model.metrics() == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_mpnn_cpu_cuda_numerical_agreement_and_update():
    cpu = _graph_net(True, candidate_value=True)
    cuda = copy.deepcopy(cpu).cuda()
    obs = _graph_obs()
    cpu_output = cpu(obs)
    cuda_output = cuda({k: v.cuda() for k, v in obs.items()})
    for a, b in zip(cpu_output, cuda_output):
        torch.testing.assert_close(a, b.cpu(), atol=2e-5, rtol=1e-4)
    before = [p.detach().clone() for p in cuda.encoder.parameters()]
    optimizer = torch.optim.Adam(cuda.parameters(), lr=.01)
    loss = cuda_output[0][:, :2].square().mean() + cuda_output[1].square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in cuda.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, cuda.encoder.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_mpnn_cuda_inference_and_restoration_repeat_exactly():
    net = _graph_net(True, candidate_value=True).cuda()
    obs = {key: value.cuda() for key, value in _graph_obs().items()}
    with torch.no_grad():
        expected = net(obs)
        for _ in range(8):
            actual = net(obs)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        state = copy.deepcopy(net.state_dict())
        restored = _graph_net(True, candidate_value=True).cuda()
        restored.load_state_dict(state)
        actual = restored(obs)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
