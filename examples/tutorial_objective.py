"""An editable, deterministic molecule score for the Python tutorial."""

import math

from rdkit import Chem
from rdkit.Chem import QED

from moldr.scoring import register_objective


def register_tutorial_objective(target_atoms=8.0, size_width=4.0, qed_weight=0.5):
    """Register a weighted QED/size score; larger values are better.

    Parameters are captured by the evaluator functions. The experiment's
    existing provenance layer saves these functions and values together.
    Edit ``score_mol`` below to substitute your research objective.
    """
    values = (target_atoms, size_width, qed_weight)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Objective parameters must be finite")
    if target_atoms <= 0 or size_width <= 0 or not 0 <= qed_weight <= 1:
        raise ValueError("Require target_atoms > 0, size_width > 0, and 0 <= qed_weight <= 1")

    def score_mol(mol):
        if mol is None:
            raise ValueError("The objective requires a valid molecule")
        distance = (mol.GetNumHeavyAtoms() - target_atoms) / size_width
        size_score = math.exp(-0.5 * distance * distance)
        score = qed_weight * QED.qed(mol) + (1.0 - qed_weight) * size_score
        if not math.isfinite(score):
            raise ValueError("The objective returned a nonfinite score")
        return float(score)

    def score_smiles(smiles):
        return score_mol(Chem.MolFromSmiles(smiles))

    return register_objective(
        "tutorial-qed-size", score_smiles, mol_function=score_mol, deterministic=True
    )
