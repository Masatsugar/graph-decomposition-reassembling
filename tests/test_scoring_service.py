import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import QED

from moldr.scoring import (
    BUILTIN_OBJECTIVES,
    OBJECTIVE_REGISTRY,
    ScoringService,
    get_objective,
    register_objective,
)


@pytest.mark.parametrize(
    "smiles", ["CCO", "C[NH3+]", "[13CH3]O", "c1cc[nH]c1", "C1CC2CCC1C2"]
)
def test_qed_mol_and_smiles_paths_match(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    expected = QED.qed(molecule)
    assert ScoringService("QED").score(smiles) == pytest.approx(expected)
    assert ScoringService("QED", input_mode="mol").score(molecule) == pytest.approx(
        expected
    )


def test_mol_path_with_cache_disabled_does_not_serialize_or_parse(monkeypatch):
    molecule = Chem.MolFromSmiles("CCO")
    expected = QED.qed(molecule)

    def forbidden(*args, **kwargs):
        raise AssertionError("Mol scoring serialized or reparsed")

    monkeypatch.setattr(Chem, "MolToSmiles", forbidden)
    monkeypatch.setattr(Chem, "MolFromSmiles", forbidden)
    assert ScoringService("qed", input_mode="mol").score(molecule) == pytest.approx(
        expected
    )


def test_cache_lru_and_separate_lifetime_counts():
    scorer = ScoringService("qed", cache_size=2)
    values = scorer.score_many(["CC", "CO", "CC", "CCC", "CO"])
    assert np.isfinite(values).all()
    assert scorer.stats == {"requests": 5, "computations": 4, "hits": 1}
    scorer.clear_cache()
    scorer.score("CC")
    assert scorer.stats == {"requests": 6, "computations": 5, "hits": 1}


def test_custom_cache_requires_explicit_purity_declaration():
    with pytest.raises(ValueError, match="deterministic"):
        ScoringService(lambda smiles: 1.0, cache_size=1)
    scorer = ScoringService(
        lambda smiles: len(smiles), cache_size=1, deterministic=True
    )
    assert not scorer.is_registered
    assert scorer.score("CC") == scorer.score("CC") == 2
    assert scorer.stats == {"requests": 2, "computations": 1, "hits": 1}


def test_registered_objective_without_mol_path_is_rejected():
    register_objective("test-no-mol", lambda smiles: len(smiles))
    assert get_objective("TEST-NO-MOL").deterministic
    with pytest.raises(ValueError, match="Mol evaluation"):
        ScoringService("test-no-mol", input_mode="mol")


def test_custom_smiles_cache_does_not_conflate_input_strings():
    scorer = ScoringService(len, cache_size=4, deterministic=True)
    assert scorer.score("C(C)O") == 5
    assert scorer.score("CCO") == 3
    assert scorer.stats == {"requests": 2, "computations": 2, "hits": 0}


@pytest.mark.parametrize("input_mode", ["smiles", "mol"])
def test_nonfinite_scores_are_rejected_and_counted(input_mode):
    scorer = ScoringService(lambda value: float("nan"), input_mode=input_mode)
    with pytest.raises(ValueError, match="finite"):
        scorer.score("CC")
    assert scorer.stats == {"requests": 1, "computations": 1, "hits": 0}


@pytest.mark.parametrize("input_mode", ["smiles", "mol"])
def test_registered_metadata_survives_fresh_worker_without_registry_entry(
    monkeypatch, input_mode
):
    import cloudpickle

    objective = register_objective(
        "worker-custom",
        lambda smiles: 0.1 * Chem.MolFromSmiles(smiles).GetNumAtoms(),
        lambda mol: 0.1 * mol.GetNumAtoms(),
    )
    metadata = cloudpickle.loads(
        cloudpickle.dumps(
            {
                "SCORING": {
                    "objective": objective,
                    "input_mode": input_mode,
                    "cache_size": 4,
                }
            }
        )
    )
    # Fresh workers import built-ins, but never executed driver registration.
    monkeypatch.delitem(OBJECTIVE_REGISTRY, objective.name)
    with pytest.raises(ValueError, match="Unknown objective"):
        get_objective(objective.name)
    scorer = ScoringService(**metadata["SCORING"])
    assert scorer.is_registered and scorer.deterministic
    assert scorer.objective.name == "worker-custom"
    value = "CCO" if input_mode == "smiles" else Chem.MolFromSmiles("CCO")
    assert scorer.score(value) == scorer.score(value) == pytest.approx(0.3)
    assert scorer.stats == {"requests": 2, "computations": 1, "hits": 1}


def test_builtin_snapshot_is_immutable_and_distinguishes_runtime_replacement(
    monkeypatch,
):
    baseline = BUILTIN_OBJECTIVES["qed"]
    with pytest.raises(TypeError):
        BUILTIN_OBJECTIVES["qed"] = baseline
    monkeypatch.setitem(OBJECTIVE_REGISTRY, "qed", baseline)
    replacement = register_objective("QED", lambda smiles: 0.125, lambda mol: 0.125)
    assert BUILTIN_OBJECTIVES["qed"] is baseline
    assert get_objective("qed") is replacement and replacement is not baseline
    scorer = ScoringService(replacement)
    assert scorer.is_registered and scorer.score("CCO") == 0.125


@pytest.mark.parametrize(
    "alias,key", [("QED_SA", "qed-sa"), ("logp8", "logp-8"), ("logp-1", "logp-minus1")]
)
def test_runtime_builtin_alias_replacement_is_resolved_consistently(
    monkeypatch, alias, key
):
    baseline = BUILTIN_OBJECTIVES[key]
    monkeypatch.setitem(OBJECTIVE_REGISTRY, key, baseline)
    replacement = register_objective(alias, lambda smiles: 0.25)
    assert get_objective(alias) is get_objective(key) is replacement
    assert BUILTIN_OBJECTIVES[key] is baseline


def test_nondeterministic_metadata_still_rejects_cache():
    objective = register_objective(
        "worker-random", lambda smiles: 0.0, deterministic=False
    )
    assert not ScoringService(objective).deterministic
    with pytest.raises(ValueError, match="deterministic"):
        ScoringService(objective, cache_size=1)


def test_metadata_preserves_missing_mol_evaluator_rejection():
    objective = register_objective("worker-no-mol", lambda smiles: len(smiles))
    with pytest.raises(ValueError, match="Mol evaluation"):
        ScoringService(objective, input_mode="mol")


def test_custom_mol_metadata_cache_preserves_input_atom_order():
    objective = register_objective(
        "worker-atom-order",
        lambda smiles: 0.0,
        lambda mol: mol.GetAtomWithIdx(0).GetAtomicNum(),
    )
    scorer = ScoringService(objective, input_mode="mol", cache_size=4)
    oxygen_first, carbon_first = Chem.MolFromSmiles("OC"), Chem.MolFromSmiles("CO")
    assert Chem.MolToSmiles(oxygen_first) == Chem.MolToSmiles(carbon_first)
    assert scorer.score(oxygen_first) == 8
    assert scorer.score(carbon_first) == 6
    assert scorer.score(oxygen_first) == 8
    assert scorer.stats == {"requests": 3, "computations": 2, "hits": 1}


def test_registered_metadata_keeps_lookahead_available_after_serialization(monkeypatch):
    import cloudpickle

    from moldr.e2e.env_v2 import MolEnvE2EV2
    from moldr.evaluation import value_lookahead

    objective = register_objective(
        "worker-lookahead",
        lambda smiles: 0.05 * Chem.MolFromSmiles(smiles).GetNumAtoms(),
        lambda mol: 0.05 * mol.GetNumAtoms(),
    )
    metadata = cloudpickle.loads(
        cloudpickle.dumps(
            {
                "BUILDING_BLOCKS": ["CC", "CO"],
                "BASE_SMILES": "C",
                "LENGTH": 5,
                "MAX_ATTACH": 4,
                "MAX_EPISODE_STEPS": 4,
                "SCORE_THRESHOLD": 2.0,
                "REPRESENTATION": {"name": "mpnn"},
                "SCORING": {"objective": objective, "cache_size": 8},
            }
        )
    )
    monkeypatch.delitem(OBJECTIVE_REGISTRY, objective.name)
    env = MolEnvE2EV2(metadata)
    env.reset(seed=0)
    env.step(1)
    env.validate_lookahead()
    before = env.scorer.requests
    choice = value_lookahead(env, lambda observation: 0.0)
    count = len(env._candidates)
    env.step(choice.action)
    assert choice.score_requests == count
    assert env.scorer.requests - before == count + 1
    assert env.scorer.hits >= 1
