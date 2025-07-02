from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.scoring_function import (
    ScoringFunctionBasedOnRdkitMol,
    ScoringFunctionWrapper,
)
from guacamol.standard_benchmarks import logP_benchmark, similarity
from rdkit import Chem

from moldr.sascore import calculateScore


def rediscovery(target) -> GoalDirectedBenchmark:
    """

    Parameters
    ----------
    target: ["Celecoxib rediscovery", "Troglitazone rediscovery",  "Thiothixene rediscovery", "Aripiprazole similarity"]

    Returns
    -------

    """
    if target == "Celecoxib rediscovery":
        sim = similarity(
            smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
            name="Celecoxib",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        )
    elif target == "Troglitazone rediscovery":
        sim = similarity(
            smiles="Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
            name="Troglitazone",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        )
    elif target == "Thiothixene rediscovery":
        sim = similarity(
            smiles="CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
            name="Thiothixene",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        )
    elif target == "Aripiprazole similarity":
        sim = similarity(
            smiles="Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl",
            name="Aripiprazole",
            fp_type="ECFP4",
            threshold=0.75,
        )
    elif target == "Albuterol similarity":  # OK
        sim = similarity(
            smiles="CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
            name="Albuterol",
            fp_type="FCFP4",
            threshold=0.75,
        )
    elif target == "Mestranol similarity":
        sim = similarity(
            smiles="COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
            name="Mestranol",
            fp_type="AP",
            threshold=0.75,
        )
    else:
        raise ValueError(
            "Select an appropriate target. See GuacaMol Rediscovery dataset"
        )
    return sim


class QED_SA(ScoringFunctionBasedOnRdkitMol):
    """
    Multi-objective function: max(0, QED - 0.1 * SA)
    """

    def __init__(self, ratio=0.1) -> None:
        super().__init__()
        self.ratio = ratio

    def score_mol(self, mol: Chem.Mol) -> float:
        return max(0, Chem.QED.qed(mol) - 0.1 * calculateScore(mol))


def qed_sa() -> GoalDirectedBenchmark:
    objective = QED_SA()
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="QED-SA", objective=objective, contribution_specification=specification
    )
