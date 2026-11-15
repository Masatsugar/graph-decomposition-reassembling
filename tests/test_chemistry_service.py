"""Chemistry equality against a frozen oracle and independent ablations."""

from itertools import product

import pytest
from rdkit import Chem

from moldr.chemistry import CandidateGenerator
from moldr.chemutils import canonical_smiles, get_mol, get_smiles, mol_with_atom_index
from moldr.core.reassemble import merge_edge, merge_node
from tests.reference_reassembly import candidates as reference_candidates

PAIRS = [
    ("C", "CCO"),
    ("CCO", "C#N"),
    ("CC(=O)O", "CO"),
    ("c1ccccc1", "C1CCCCC1"),
    ("c1ccncc1", "C1CCOCC1"),
    ("C[NH3+]", "CC"),
    ("[13CH3]CO", "CC"),
    ("c1cc[nH]c1", "CC"),
    ("[NH2]C", "CO"),
    ("C1CC2CCC1C2", "C1CCC2CC1C2"),
    ("[13CH2]1CCCCC1", "C1CCCCC1"),
    ("C1CC[NH2+]CC1", "C1CCCCC1"),
]


@pytest.mark.parametrize("base,fragment", PAIRS)
def test_default_raw_and_final_order_matches_frozen_reference(base, fragment):
    base_mol, fragment_mol = get_mol(base), get_mol(fragment)
    nodes, edges, final = reference_candidates(base_mol, fragment_mol)
    assert [
        candidate.smiles for candidate in merge_node(base_mol, fragment_mol)
    ] == nodes
    assert [
        candidate.smiles for candidate in merge_edge(base_mol, fragment_mol)
    ] == edges
    service = CandidateGenerator()
    result = service.generate(base_mol, fragment_mol)
    result_set = {canonical_smiles(candidate.mol) for candidate in result}
    ref_set = {canonical_smiles(get_mol(smi)) for smi in final}
    assert service.stats["valid"] == len(result)
    assert len(result) == len(result_set)
    assert result_set == ref_set


@pytest.mark.parametrize(
    "map_storage,shared,bond_scan",
    list(product(["int", "string"], [True, False], ["nonzero", "python"])),
)
def test_p1_switches_independently_preserve_raw_and_final_order(
    map_storage, shared, bond_scan
):
    base, fragment = get_mol("N1CCCCC1"), get_mol("C1CCOCC1")
    nodes, edges, final = reference_candidates(base, fragment)
    options = {
        "map_storage": map_storage,
        "shared_preparation": shared,
        "bond_scan": bond_scan,
    }
    assert [
        candidate.smiles for candidate in merge_node(base, fragment, **options)
    ] == nodes
    assert [
        candidate.smiles for candidate in merge_edge(base, fragment, **options)
    ] == edges
    assert [
        candidate.smiles
        for candidate in CandidateGenerator(options).generate(base, fragment)
    ] == [
        candidate.smiles for candidate in CandidateGenerator().generate(base, fragment)
    ]
    legacy = CandidateGenerator(
        {
            **options,
            "candidate_representation": "mapped",
            "dedup_key": "kekule",
            "mol_transfer": "smiles",
        }
    ).generate(base, fragment)
    assert [candidate.smiles for candidate in legacy] == final


@pytest.mark.parametrize("storage", ["int", "string"])
def test_atom_zero_map_is_present(storage):
    mol = mol_with_atom_index(get_mol("CO"), storage)
    assert mol.GetAtomWithIdx(0).HasProp("molAtomMapNumber")
    assert mol.GetAtomWithIdx(0).GetIntProp("molAtomMapNumber") == 0
    assert ":0]" in get_smiles(mol)


def test_generation_does_not_mutate_input_properties_or_bond_order():
    base, fragment = get_mol("C[NH3+]"), get_mol("CO")
    base.GetAtomWithIdx(0).SetIntProp("molAtomMapNumber", 42)
    before = [
        mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in (base, fragment)
    ]
    CandidateGenerator().generate(base, fragment)
    merge_node(base, fragment)
    merge_edge(base, fragment)
    after = [
        mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in (base, fragment)
    ]
    assert before == after


@pytest.mark.parametrize(
    "representation,dedup,transfer",
    [
        ("mapped", "kekule", "smiles"),
        ("mapless", "kekule", "smiles"),
        ("mapless", "aromatic", "smiles"),
        ("mapped", "aromatic", "smiles"),
        ("mapless", "kekule", "direct"),
    ],
)
@pytest.mark.parametrize("base,fragment", PAIRS)
def test_p2_switches_compare_molecular_sets_and_counts_deterministically(
    base, fragment, representation, dedup, transfer
):
    molecules = get_mol(base), get_mol(fragment)
    reference = CandidateGenerator().generate(*molecules)
    generator = CandidateGenerator(
        candidate_representation=representation, dedup_key=dedup, mol_transfer=transfer
    )
    result = generator.generate(*molecules)
    canonical = lambda candidates: {
        canonical_smiles(candidate.mol) for candidate in candidates
    }
    baseline_set, experimental_set = canonical(reference), canonical(result)
    assert experimental_set == baseline_set
    if dedup == "aromatic":
        assert len(result) == len(reference)
    else:
        assert len(result) >= len(reference)
    assert generator.stats["valid"] == len(result)
    assert generator.stats["duplicates"] >= 0
    assert [candidate.smiles for candidate in generator.generate(*molecules)] == [
        candidate.smiles for candidate in result
    ]
    assert all(
        not atom.HasProp("molAtomMapNumber")
        for candidate in result
        for atom in candidate.mol.GetAtoms()
    )


def test_pair_preparation_is_shared_across_both_merge_modes(monkeypatch):
    import moldr.core.reassemble as module

    calls = []
    real = module.Chem.CombineMols

    def count(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(module.Chem, "CombineMols", count)
    CandidateGenerator().generate(get_mol("C1CCCCC1"), get_mol("C1CCCCC1"))
    assert len(calls) == 1
    calls.clear()
    generator = CandidateGenerator(shared_preparation=False)
    generator.generate(get_mol("C1CCCCC1"), get_mol("C1CCCCC1"))
    assert len(calls) == generator.stats["generated"]


def test_direct_route_never_reparses_candidate_smiles(monkeypatch):
    import moldr.chemistry as module

    base, fragment = get_mol("C1CCCCC1"), get_mol("C1CCCCC1")

    def forbidden(*args):
        raise AssertionError("direct candidate transfer parsed SMILES")

    monkeypatch.setattr(module, "get_mol", forbidden)
    assert CandidateGenerator(
        candidate_representation="mapless", mol_transfer="direct"
    ).generate(base, fragment)


@pytest.mark.parametrize(
    "base,fragment", [(None, "CC"), ("", "CC"), ("C[C@H](O)F", "CC"), ("CC", "C/C=C/C")]
)
def test_invalid_inputs_fail_like_reference(base, fragment):
    molecules = get_mol(base) if base is not None else None, get_mol(fragment)
    with pytest.raises(ValueError):
        reference_candidates(*molecules)
    with pytest.raises(ValueError):
        CandidateGenerator().generate(*molecules)


def test_direct_requires_mapless():
    with pytest.raises(ValueError, match="mapless"):
        CandidateGenerator(candidate_representation="mapped", mol_transfer="direct")


def test_fused_ring_candidates_are_not_dropped():
    base = get_mol("CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1")
    fragment = get_mol("Brc1ccccc1")
    generator = CandidateGenerator()
    candidates = generator.generate(base, fragment)
    assert generator.stats["invalid"] == 0
    assert len(candidates) == 19
    assert len({canonical_smiles(candidate.mol) for candidate in candidates}) == 19


def test_kekule_resonance_duplicates_are_eliminated():
    base = get_mol("c1ccc2ccccc2c1")
    fragment = get_mol("CCO")
    generator = CandidateGenerator()
    candidates = generator.generate(base, fragment)
    canonical = {canonical_smiles(candidate.mol) for candidate in candidates}
    assert len(candidates) == 4
    assert len(canonical) == 4

    kekule = CandidateGenerator(dedup_key="kekule").generate(base, fragment)
    assert len(kekule) == 8
    assert {canonical_smiles(candidate.mol) for candidate in kekule} == canonical
