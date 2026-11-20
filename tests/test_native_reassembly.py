"""Differential checks for the optional C++ candidate generator."""

from importlib.util import find_spec

import pytest
from rdkit import Chem

from moldr.chemistry import CandidateGenerator
from moldr.chemutils import get_mol
from tests.test_local_reassembly import candidate_state

if find_spec("moldr.core._native_reassembly") is None:
    pytest.skip("Optional C++ extension has not been built", allow_module_level=True)


def test_cpp_handles_default_chemistry_without_python_enumeration(monkeypatch):
    def unexpected_builder(*args):
        pytest.fail("Default C++ path used the Python candidate builder")

    monkeypatch.setattr("moldr.chemistry.prepare_local_pair", unexpected_builder)
    candidates = CandidateGenerator(backend="cpp").generate(
        get_mol("CC"), get_mol("CN")
    )
    assert [c.smiles for c in candidates] == ["CCC", "CCN", "CNC"]


@pytest.mark.parametrize(
    "base,fragment",
    [
        ("S", "c1ccncc1"),
        ("[13CH3]CO", "C1CCCCC1"),
        ("C1CC[NH2+]CC1", "C1CCCCC1"),
        ("C1CC2CCC1C2", "C1CCC2CC1C2"),
        ("[CH2]C", "CO"),
        ("CC.Cl", "CO"),
        ("N->[Cu]", "CC"),
        ("[He]", "[He]"),
    ],
)
@pytest.mark.parametrize("parse", [Chem.MolFromSmiles, get_mol])
def test_native_matches_python(base, fragment, parse):
    base, fragment = parse(base), parse(fragment)
    base = Chem.RenumberAtoms(base, list(reversed(range(base.GetNumAtoms()))))
    base.GetAtomWithIdx(0).SetAtomMapNum(17)
    before = [
        mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in (base, fragment)
    ]
    python, native = CandidateGenerator(), CandidateGenerator(backend="cpp")
    expected = python.generate(Chem.Mol(base), Chem.Mol(fragment))
    actual = native.generate(base, fragment)
    assert candidate_state(actual) == candidate_state(expected)
    assert native.stats == python.stats
    assert before == [
        mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in (base, fragment)
    ]


@pytest.mark.parametrize("base", [None, "", "F[C@H](Cl)Br", "F/C=C/F"])
def test_native_rejects_invalid_inputs(base):
    with pytest.raises(ValueError):
        CandidateGenerator(backend="cpp").generate(
            None if base is None else get_mol(base), get_mol("CC")
        )


def test_native_checks_invalid_base_before_missing_fragment():
    base = get_mol("F/C=C/F")
    with pytest.raises(ValueError, match="Stereochemical input is unsupported"):
        CandidateGenerator(backend="cpp").generate(base, None)


@pytest.mark.parametrize("bond_smiles", ["N->[Cu]", "N<-[Cu]", "C~C", "C$C"])
@pytest.mark.parametrize("special_is_base", [True, False])
def test_native_defers_special_bonds_to_adjacency(bond_smiles, special_is_base):
    from moldr.core import _native_reassembly

    special, ring = get_mol(bond_smiles), get_mol("C1CCCCC1")
    base, fragment = (special, ring) if special_is_base else (ring, special)
    assert _native_reassembly.generate(base, fragment) is None
    python, native = CandidateGenerator(), CandidateGenerator(backend="cpp")
    expected = python.generate(Chem.Mol(base), Chem.Mol(fragment))
    assert candidate_state(native.generate(base, fragment)) == candidate_state(expected)
    assert native.stats == python.stats


@pytest.mark.parametrize(
    "options",
    [
        {"shared_preparation": False},
        {"bond_scan": "python"},
        {"dedup_key": "kekule"},
        {"mol_transfer": "smiles"},
        {"candidate_representation": "mapped", "mol_transfer": "smiles"},
    ],
)
def test_cpp_config_uses_python_for_alternate_chemistry(monkeypatch, options):
    from moldr.core import _native_reassembly

    def unexpected_native(*args):
        pytest.fail("Alternate chemistry used the C++ candidate builder")

    monkeypatch.setattr(_native_reassembly, "generate", unexpected_native)
    base, fragment = get_mol("C1CCCCC1"), get_mol("CO")
    python = CandidateGenerator(**options)
    cpp = CandidateGenerator(backend="cpp", **options)
    assert candidate_state(cpp.generate(base, fragment)) == candidate_state(
        python.generate(base, fragment)
    )
    assert cpp.stats == python.stats


def test_native_drives_same_e2e_attachment():
    from moldr.e2e.env_v2 import MolEnvE2EV2

    config = {
        "BUILDING_BLOCKS": ["CO"],
        "BASE_SMILES": "C",
        "MAX_ATTACH": 1,
        "MAX_EPISODE_STEPS": 6,
        "LENGTH": 20,
        "SCORE_THRESHOLD": 2.0,
        "SCORING": {"objective": "qed"},
        "REPRESENTATION": {"name": "mpnn", "max_atoms": 24, "max_bonds": 40},
    }
    results = []
    for backend in ("python", "cpp"):
        env = MolEnvE2EV2({**config, "CHEMISTRY": {"backend": backend}})
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert env._candidate_smiles == ["CC"]
        assert info["n_candidates_total"] == 2
        _, reward, done, truncated, info = env.step(0)
        results.append((env.base_smiles, reward, done, truncated, info["score"]))
    assert results[0] == results[1]
