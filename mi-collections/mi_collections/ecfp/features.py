from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx
import numpy as np
import pandas as pd
import rdkit
from mi_collections.chemutils import _set_node_label, get_atomic_num, mol_to_graph
from numpy import ndarray
from rdkit.Chem import AllChem, Draw, rdchem
from tqdm import tqdm


def get_custom_node_features(mol) -> List[str]:
    features = np.array(
        [
            [
                a.IsInRing(),
                a.GetAtomicNum(),
                a.GetDegree(),
                a.GetExplicitValence(),
                a.GetImplicitValence(),
                a.GetFormalCharge(),
                a.GetTotalNumHs(),
            ]
            for a in mol.GetAtoms()
        ],
        dtype=np.int32,
    )
    n_feat = [
        "".join([str(f) for f in features[atom]]) for atom in range(len(mol.GetAtoms()))
    ]
    return n_feat


def important_structures(mols, infos, query_keys):
    morgan_tuples = []
    for key in query_keys:
        key = int(key)
        for mol, info in zip(mols, infos):
            if key in info.keys():
                morgan_tuples.append((mol, key, info))
                break

    options = Draw.MolDraw2DSVG(100, 100).drawOptions()
    options.padding = 0.1
    return Draw.DrawMorganBits(
        morgan_tuples,
        molsPerRow=5,
        legends=[f"{x[1]:04}" for x in morgan_tuples],
        drawOptions=options,
        subImgSize=(150, 100),
    )


def get_morgan_tuples(mols, infos, query_keys):
    morgan_tuples = []
    for key in query_keys:
        key = int(key)
        for mol, info in zip(mols, infos):
            if key in info.keys():
                morgan_tuples.append((mol, key, info))
                break

    return morgan_tuples


def get_infos(
    mols: List[rdchem.Mol], n_radius: int = 2, **kwargs
) -> List[Dict[int, tuple]]:
    """

    Args:
        mols:
        n_radius:
        **kwargs:

    Returns:
        Info

    """
    infos = []
    for mol in mols:
        info = {}
        AllChem.GetMorganFingerprint(mol=mol, radius=n_radius, bitInfo=info, **kwargs)
        infos.append(info)
    return infos


def draw_morgan_bits(
    morgan_tuples: List[Tuple[int, int]], subImgSize: tuple = (150, 150)
) -> object:
    """

    :param morgan_tuples:
    :param subImgSize:
    :return: DrawMorganBits
    """
    options = Draw.MolDraw2DSVG(100, 100).drawOptions()
    options.padding = 0.1
    return Draw.DrawMorganBits(
        morgan_tuples,
        molsPerRow=5,
        legends=[f"{x[1]:04}" for x in morgan_tuples],
        drawOptions=options,
        subImgSize=subImgSize,
    )


def get_identifier(
    mol: rdkit.Chem.Mol, radius: int, nbits: Optional[int] = None, **kwargs
) -> Tuple[Dict[int, str], Dict[int, tuple]]:
    """Calculate identifiers of subgraphs in molecules

    Args:
        :param mol:
        :param radius:
        :param nbits:
        :param kwargs:
    :return:

    """
    bit_info = {}
    radii = list(range(int(radius) + 1))
    if isinstance(nbits, int):
        _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits, bitInfo=bit_info)
    else:
        _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=bit_info, **kwargs)
    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}
    for element in bit_info:
        for atom_idx, radius_at in bit_info[element]:
            dict_atoms[atom_idx][radius_at] = str(
                element
            )  # {atom number: {fp radius: identifier}}

    return dict_atoms, bit_info


def calc_node_features(mol, use_features: bool = True) -> List[str]:
    n_feat = get_custom_node_features(mol) if use_features else get_atomic_num(mol)
    return n_feat


def set_node_label(mols, use_features=True) -> List[networkx.Graph]:
    graphs = []
    for mol in mols:
        graph = mol_to_graph(mol)
        label = calc_node_features(mol, use_features)
        _set_node_label(graph, label)
        graphs.append(graph)
    return graphs


def get_mapping(X, X2) -> dict:
    """ """
    mapping = {}
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            key = X.iloc[j, i]
            val = X2.iloc[j, i]
            key = int(key) if key is not None else None
            if key:
                mapping.update({key: int(val)})
    return mapping


def check_duplicate(X, X_bit, nid=0, test_dict=None):
    mapping = get_mapping(X, X_bit)
    if test_dict is None:
        test_dict = defaultdict(list)

    # res_nid = None
    for k, v in mapping.items():
        if v not in test_dict.keys():
            test_dict[v].append(k)
        # else:
        #     test_dict[v].append(k)
        #     res_nid = nid
    #
    # score_sorted = sorted(test_dict.items(), key=lambda x:x[0])
    return test_dict, nid


def get_all_identifiers(
    mol, radius: int = 2, is_dropna: bool = False, **kwargs
) -> Union[
    List[Any],
    ndarray,
    Tuple[ndarray, Optional[ndarray], Optional[ndarray], Optional[ndarray]],
]:
    """Calculate all identifiers (substructures) of mols.

    Args:
        mol:
        radius:
        is_dropna:

    Returns:
        All identifiers of sub graphs in molecules

    """
    X, _ = get_identifier(mol, radius, **kwargs)
    X = pd.DataFrame(X).T
    if is_dropna:
        x, _ = get_identifier(mol, radius, **kwargs)
        xs = pd.DataFrame(x).to_numpy().flatten()
        return [x for x in xs if x is not None]
    i_list = np.unique(X.fillna("-1").to_numpy().flatten())
    return i_list


def check_bit_collision(identifiers, nbits=2048, dups=None):
    """Calculate mappings from each bit to identifier list.

    Args:
        identifiers:
        nbits:
        dups:

    Returns: {"id2bit": id2bit, "duplicates": duplicates}

    """
    if dups is None:
        dups = {}
    id2bit = defaultdict(list)
    for res in tqdm(identifiers):
        for key in res:
            key = int(key)
            if key == -1:
                continue

            idx = key % nbits
            if key not in id2bit[idx]:
                id2bit[idx].append(key)

    results = dict(sorted(id2bit.items(), key=lambda x: x[0]))
    counter = []
    for k, v in tqdm(id2bit.items()):
        if len(v) > 1:
            counter.append(len(v))
    dups.update({nbits: sum(counter)})
    return {"id2bit": results, "duplicate": dups}
