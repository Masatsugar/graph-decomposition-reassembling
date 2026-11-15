"""Fragment vocabulary for the generative proposer.

The decoder never emits raw atoms or SMILES characters. It emits an *index into
a pre-validated fragment vocabulary*. That is the JT-VAE / HierVAE trick that
guarantees chemical validity: every token the model can produce is a known-good
substructure, so a decode can never yield an invalid, un-kekulizable, or
single-atom block.

A vocabulary entry survives the exact gates the live MOLDR pipeline imposes on a
building block (see the scout contract):

1. ``get_mol(s) is not None``          -> parses *and* kekulizes (env.py:28)
2. ``len(s) > 1``                       -> multi-atom (train.py:146)
3. >=1 atom with positive implicit valence -> node-mergeable
   (check_valance, reassemble.py:216)
4. no specified stereochemistry -> supported by the graph-reassembly engine
   (it cannot preserve atom chirality or bond stereo when rebuilding products)

Each entry is additionally tagged ``has_ring`` (``num_rings >= 1``) so the
proposer can guarantee a minimum fraction of edge-mergeable (ring) blocks
(merge_edge requires a ring on both sides, reassemble.py:354).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import rdkit.Chem as Chem
from rdkit import RDLogger

from moldr.chemutils import get_mol, get_smiles, validate_reassembly_mol

RDLogger.DisableLog("rdApp.*")

# Elements we count explicitly in the descriptor; everything else -> "other".
_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "B"]
# Descriptor layout: see fragment_descriptor().
DESCRIPTOR_DIM = len(_ELEMENTS) + 7


def num_rings(mol: Chem.Mol) -> int:
    """Ring count that works across RDKit versions (mirrors reassemble.num_rings)."""
    sssr = Chem.GetSSSR(mol)
    return sssr if isinstance(sssr, int) else len(sssr)


def has_open_valence(mol: Chem.Mol) -> bool:
    """True if at least one atom has spare implicit valence (node-mergeable)."""
    return any(_implicit_valence(a) > 0 for a in mol.GetAtoms())


def _implicit_valence(atom: Chem.Atom) -> int:
    """Use the current RDKit API, while retaining older-release support."""
    if hasattr(atom, "GetValence"):
        return atom.GetValence(Chem.ValenceType.IMPLICIT)
    return atom.GetImplicitValence()


def is_valid_block(smiles: str) -> bool:
    """Exactly the validity contract the env + train pipeline imposes on a block."""
    if not isinstance(smiles, str) or len(smiles) <= 1:
        return False
    mol = get_mol(smiles)  # MolFromSmiles + Kekulize, returns None on failure
    if mol is None:
        return False
    try:
        validate_reassembly_mol(mol)
    except ValueError:
        return False
    if mol.GetNumAtoms() < 2:
        return False
    return has_open_valence(mol)


def fragment_descriptor(mol: Chem.Mol) -> np.ndarray:
    """Cheap, deterministic, RDKit-only structural descriptor of a fragment.

    Used as the VAE *encoder* input so vocabulary building needs no Mol2Vec /
    gensim model (keeps the pipeline light and the CPU demo fast). Layout
    (length == ``DESCRIPTOR_DIM``):

        [count(C), count(N), ... count(B),          # per-element atom counts
         n_atoms, n_bonds, n_rings, n_open_valence, # topology
         n_aromatic_atoms, max_ring_size, exact_mw] # misc
    """
    counts = {e: 0 for e in _ELEMENTS}
    n_aromatic = 0
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym in counts:
            counts[sym] += 1
        if a.GetIsAromatic():
            n_aromatic += 1
    ri = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]
    max_ring = max(ring_sizes) if ring_sizes else 0
    n_open = sum(1 for a in mol.GetAtoms() if _implicit_valence(a) > 0)
    mw = sum(a.GetMass() for a in mol.GetAtoms())
    feats = [float(counts[e]) for e in _ELEMENTS] + [
        float(mol.GetNumAtoms()),
        float(mol.GetNumBonds()),
        float(num_rings(mol)),
        float(n_open),
        float(n_aromatic),
        float(max_ring),
        float(mw) / 100.0,  # scaled so magnitudes are O(1)
    ]
    return np.asarray(feats, dtype=np.float32)


class FragmentVocab:
    """A fixed, ordered list of pre-validated fragment SMILES.

    Order is stable (and persisted) because the decoder's output dimension is
    indexed by vocabulary position.
    """

    def __init__(self, smiles: Sequence[str]):
        # Canonicalise to kekule form, validate, and de-duplicate while keeping
        # first-seen order -> deterministic, stable indices.
        seen = set()
        kept: List[str] = []
        for s in smiles:
            if not is_valid_block(s):
                continue
            canon = get_smiles(get_mol(s))  # kekule canonical, matches table form
            if canon in seen or not is_valid_block(canon):
                continue
            seen.add(canon)
            kept.append(canon)
        if not kept:
            raise ValueError("FragmentVocab is empty after validity filtering.")
        self.smiles: List[str] = kept
        self._ring = np.asarray(
            [num_rings(get_mol(s)) >= 1 for s in self.smiles], dtype=bool
        )
        self._desc = np.stack([fragment_descriptor(get_mol(s)) for s in self.smiles])

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> str:
        return self.smiles[idx]

    @property
    def ring_mask(self) -> np.ndarray:
        """Boolean mask: entry i contains >=1 ring (edge-mergeable)."""
        return self._ring

    @property
    def descriptors(self) -> np.ndarray:
        """(M, DESCRIPTOR_DIM) encoder-input features, one row per entry."""
        return self._desc

    def index_of(self, smiles: str) -> Optional[int]:
        canon = get_smiles(get_mol(smiles)) if is_valid_block(smiles) else None
        if canon is None:
            return None
        try:
            return self.smiles.index(canon)
        except ValueError:
            return None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_smiles_list(cls, smiles: Sequence[str]) -> "FragmentVocab":
        """Bootstrap a vocabulary directly from candidate fragment SMILES.

        Use this to seed from the existing building-block CSVs (which are already
        fragment-sized), or from any curated fragment list.
        """
        return cls(smiles)

    @classmethod
    def from_molecules(cls, smiles: Sequence[str]) -> "FragmentVocab":
        """Extract JT fragments from full molecules via tree decomposition.

        Mirrors the junction-tree route already in the repo: ``tree_decomp`` +
        ``get_clique_mol`` (moltree.py), the same substructures the ``--jt``
        building-block option uses.
        """
        # Imported lazily: pulls in scipy via moltree only when this path is used.
        from moldr.core.moltree import tree_decomp
        from moldr.chemutils import get_clique_mol

        frags: List[str] = []
        for s in smiles:
            mol = get_mol(s)
            if mol is None:
                continue
            try:
                cliques, _ = tree_decomp(mol)
            except Exception:
                continue
            for clique in cliques:
                try:
                    cmol = get_clique_mol(mol, clique)
                    if cmol is None:
                        continue
                    frags.append(get_smiles(cmol))
                except Exception:
                    continue
        return cls(frags)

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"smiles": self.smiles}))

    @classmethod
    def load(cls, path: str | Path) -> "FragmentVocab":
        data = json.loads(Path(path).read_text())
        return cls(data["smiles"])
