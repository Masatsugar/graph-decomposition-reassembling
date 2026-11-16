"""Behavioral regressions for local edits in the public candidate generator."""

import pytest
from rdkit import Chem

from moldr import chemistry
from moldr.chemistry import CandidateGenerator
from moldr.chemutils import get_mol


def molecular_state(mol):
    """Keep atom/bond insertion order and chemistry that SMILES alone omits."""
    return (
        tuple(
            (
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                atom.GetIsotope(),
                atom.GetNumExplicitHs(),
                atom.GetNumImplicitHs(),
                atom.GetNoImplicit(),
                atom.GetNumRadicalElectrons(),
                atom.GetChiralTag(),
                atom.GetAtomMapNum(),
                atom.GetIsAromatic(),
                atom.GetHybridization(),
            )
            for atom in mol.GetAtoms()
        ),
        tuple(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType(),
                bond.GetIsAromatic(),
                bond.GetIsConjugated(),
                bond.GetStereo(),
                bond.GetBondDir(),
                tuple(bond.GetStereoAtoms()),
            )
            for bond in mol.GetBonds()
        ),
        tuple(mol.GetRingInfo().AtomRings()),
        tuple(mol.GetRingInfo().BondRings()),
    )


def candidate_state(candidates):
    return [
        (candidate.smiles, molecular_state(candidate.mol)) for candidate in candidates
    ]


def adjacency_reference(monkeypatch, base, fragment, **options):
    """Use the unchanged adjacency builder with the same service configuration."""
    with monkeypatch.context() as patch:
        patch.setattr(chemistry, "prepare_local_pair", lambda *args: None)
        generator = CandidateGenerator(**options)
        result = generator.generate(base, fragment)
    return result, generator.stats


@pytest.mark.parametrize(
    "base,fragment",
    [
        ("C", "CO"),
        ("CCO", "C#N"),
        ("CC(=O)O", "CO"),
        ("c1ccccc1", "C1CCCCC1"),
        ("c1ccncc1", "C1CCOCC1"),
        ("c1ccc2ccccc2c1", "CCO"),
        ("c1cc[nH]c1", "CC"),
        ("C1CC2CCC1C2", "C1CCC2CC1C2"),
        ("[13CH2]1CCCCC1", "C1CCCCC1"),
        ("C1CC[NH2+]CC1", "C1CCCCC1"),
        ("[13CH3]CO", "CC"),
        ("C[NH3+]", "CC"),
        ("[CH2]C", "CO"),
        ("CS(=O)(=O)C", "CCO"),
        ("CC.Cl", "CO"),
    ],
)
@pytest.mark.parametrize("aromatic_input", [False, True])
@pytest.mark.parametrize("mol_transfer", ["direct", "smiles"])
def test_local_edits_preserve_ordered_candidates_and_stats(
    monkeypatch, base, fragment, aromatic_input, mol_transfer
):
    parse = Chem.MolFromSmiles if aromatic_input else get_mol
    base_mol, fragment_mol = parse(base), parse(fragment)
    # Atom order need not be canonical, even when input molecules are identical.
    base_mol = Chem.RenumberAtoms(
        base_mol, list(reversed(range(base_mol.GetNumAtoms())))
    )
    expected, expected_stats = adjacency_reference(
        monkeypatch, base_mol, fragment_mol, mol_transfer=mol_transfer
    )
    generator = CandidateGenerator(mol_transfer=mol_transfer)
    result = generator.generate(base_mol, fragment_mol)
    assert candidate_state(result) == candidate_state(expected)
    assert generator.stats == expected_stats


def test_default_generator_uses_local_builder(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default supported inputs used the adjacency builder")

    monkeypatch.setattr(chemistry, "prepare_pair", forbidden)
    result = CandidateGenerator().generate(get_mol("CC"), get_mol("CN"))
    assert [candidate.smiles for candidate in result] == ["CCC", "CCN", "CNC"]


def test_aromatic_symmetry_does_not_prune_distinct_kekule_candidates(monkeypatch):
    base, fragment = get_mol("S"), get_mol("c1ccncc1")
    expected, stats = adjacency_reference(monkeypatch, base, fragment)
    generator = CandidateGenerator()
    result = generator.generate(base, fragment)
    assert len(result) == 6
    assert {"C1=CC=[SH]N=C1", "C1=C[SH]=CN=C1"}.issubset(
        candidate.smiles for candidate in result
    )
    assert candidate_state(result) == candidate_state(expected)
    assert generator.stats == stats


def test_charged_atom_is_not_rejected_using_neutral_valence(monkeypatch):
    base, fragment = get_mol("CCC"), get_mol("CO")
    # The public API accepts RDKit molecules, including implicit-H charged atoms.
    fragment.GetAtomWithIdx(1).SetFormalCharge(1)
    Chem.SanitizeMol(fragment)
    expected, stats = adjacency_reference(monkeypatch, base, fragment)
    generator = CandidateGenerator()
    result = generator.generate(base, fragment)
    assert "C[O+](C)C" in [candidate.smiles for candidate in result]
    assert candidate_state(result) == candidate_state(expected)
    assert generator.stats == stats


def test_fragment_changes_do_not_reuse_stale_features(monkeypatch):
    generator = CandidateGenerator()
    base, fragment = get_mol("CC"), get_mol("CC")
    generator.generate(base, fragment)
    fragment.GetAtomWithIdx(1).SetAtomicNum(7)
    Chem.SanitizeMol(fragment)
    result = generator.generate(base, fragment)
    assert [candidate.smiles for candidate in result] == ["CCC", "CCN", "CNC"]

    # Fragment pools can replace objects and change atom count between calls.
    for smiles in ["C", "CN", "CC", "CO", "C", "C1CCCCC1", "CN"]:
        fragment = get_mol(smiles)
        expected, stats = adjacency_reference(monkeypatch, base, fragment)
        assert candidate_state(generator.generate(base, fragment)) == candidate_state(
            expected
        )
        assert generator.stats == stats
        del fragment


def test_dative_bond_uses_existing_conversion_instead_of_disappearing(monkeypatch):
    base, fragment = get_mol("N->[Cu]"), get_mol("CC")
    expected, stats = adjacency_reference(monkeypatch, base, fragment)
    generator = CandidateGenerator()
    result = generator.generate(base, fragment)
    assert [candidate.smiles for candidate in result] == [
        "C[CH2][Cu]",
        "C[NH][Cu]",
    ]
    assert candidate_state(result) == candidate_state(expected)
    assert generator.stats == stats


@pytest.mark.parametrize("base", [None, "", "F[C@H](Cl)Br", "F/C=C/F"])
def test_local_generation_preserves_input_rejections(base):
    base_mol = None if base is None else get_mol(base)
    with pytest.raises(ValueError):
        CandidateGenerator().generate(base_mol, get_mol("CC"))


def test_local_generation_preserves_input_properties_and_clears_output_maps():
    base, fragment = get_mol("[13CH3]CO"), get_mol("C1CCCCC1")
    for mol in (base, fragment):
        mol.SetProp("caller_note", "keep")
        mol.GetAtomWithIdx(0).SetAtomMapNum(42)
        mol.GetAtomWithIdx(0).SetProp("caller_atom_note", "keep")
        mol.GetBondWithIdx(0).SetProp("caller_bond_note", "keep")
    before = [
        mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in (base, fragment)
    ]
    result = CandidateGenerator().generate(base, fragment)
    assert result
    assert before == [
        mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in (base, fragment)
    ]
    assert all(
        not atom.HasProp("molAtomMapNumber")
        for candidate in result
        for atom in candidate.mol.GetAtoms()
    )


def test_e2e_attachment_limit_retains_sorted_candidate_and_reward():
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
    for shared in (False, True):
        env = MolEnvE2EV2({**config, "CHEMISTRY": {"shared_preparation": shared}})
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert env._candidate_smiles == ["CC"]
        assert info["n_candidates_total"] == 2
        _, reward, done, truncated, info = env.step(0)
        results.append((env.base_smiles, reward, done, truncated, info["score"]))
    assert results[0] == results[1]
    assert results[0][0] == "CC"
