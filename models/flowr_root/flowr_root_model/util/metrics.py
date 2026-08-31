import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import (
    rdchem,
    rdMolTransforms,
)
from torchmetrics import Metric

import flowr_root_model.util.rdkit as smolRD
from flowr_root_model.eval.evaluate_pose import shape_tanimoto_similarity
from flowr_root_model.util.molecule import Molecule
from flowr_root_model.util.sampling.utils import (
    angle_distance,
    atom_types_distance,
    bond_length_distance,
    bond_types_distance,
    number_nodes_distance,
    valency_distance,
)
from posebusters import PoseBusters

warnings.filterwarnings(
    "ignore",
    category=Warning,
    message="WARNING: Search space volume is greater than 27000 Angstrom^3 (See FAQ)",
)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Restore
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


ALLOWED_VALENCIES = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],  # In QM9, N+ seems to be present in the form NH+ and NH2+
        -1: 2,
    },
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}


def _is_valid_valence(valence, allowed, charge):
    if isinstance(allowed, int):
        valid = allowed == valence

    elif isinstance(allowed, list):
        valid = valence in allowed

    elif isinstance(allowed, dict):
        allowed = allowed.get(charge)
        if allowed is None:
            return False

        valid = _is_valid_valence(valence, allowed, charge)

    return valid


def _is_valid_float(num):
    return num not in [None, float("inf"), float("-inf"), float("nan")]


class GenerativeMetric(Metric):
    # TODO add metric attributes - see torchmetrics doc

    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class PairMetric(Metric):
    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(
        self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]
    ) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class Validity(GenerativeMetric):
    def __init__(self, connected=False, **kwargs):
        super().__init__(**kwargs)
        self.connected = connected

        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        is_valid = [smolRD.mol_is_valid(mol, connected=self.connected) for mol in mols]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.total


class DistributionDistance(GenerativeMetric):
    def __init__(self, dataset_info, train_mols=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_info = dataset_info
        self.train_mols = [mol for mol in train_mols if mol is not None]
        self.atom_encoder = dataset_info.atom_encoder
        self.atom_decoder = dataset_info.atom_decoder

        self.add_state("num_nodes_w1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("atom_types_tv", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("edge_types_tv", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("charge_w1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("valency_w1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "bond_lengths_w1", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("angles_w1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dihedrals_cw1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, gen_ligs: list[Chem.rdchem.Mol], state="val") -> None:
        gen_ligs = [mol for mol in gen_ligs if mol is not None]
        molecules = [Molecule(mol, device=self.device) for mol in gen_ligs]
        if len(molecules) == 0:
            return

        # Compute statistics
        stat = self.dataset_info.statistics[state]

        self.num_nodes_w1 += number_nodes_distance(molecules, stat.num_nodes)
        atom_types_tv, atom_tv_per_class = atom_types_distance(
            molecules, stat.atom_types, save_histogram=False
        )
        self.atom_types_tv += atom_types_tv
        edge_types_tv, bond_tv_per_class, sparsity_level = bond_types_distance(
            molecules, stat.bond_types, save_histogram=False
        )
        self.edge_types_tv += edge_types_tv
        valency_w1, valency_w1_per_class = valency_distance(
            molecules, stat.valencies, stat.atom_types, self.atom_encoder
        )
        self.valency_w1 += valency_w1
        bond_lengths_w1, bond_lengths_w1_per_type = bond_length_distance(
            molecules, stat.bond_lengths, stat.bond_types
        )
        self.bond_lengths_w1 += bond_lengths_w1
        angles_w1, angles_w1_per_type = angle_distance(
            molecules,
            stat.bond_angles,
            stat.atom_types,
            stat.valencies,
            atom_decoder=self.atom_decoder,
            save_histogram=False,
        )
        self.angles_w1 += angles_w1
        self.dihedrals_cw1 += calc_circular_wasserstein_distance(
            gen_ligs, self.train_mols
        )
        self.total += 1

    def compute(self) -> torch.Tensor:
        return {
            "num_nodes_w1": self.num_nodes_w1 / self.total,
            "atom_types_tv": self.atom_types_tv / self.total,
            "edge_types_tv": self.edge_types_tv / self.total,
            "valency_w1": self.valency_w1 / self.total,
            "bond_lengths_w1": self.bond_lengths_w1 / self.total,
            "angles_w1": self.angles_w1 / self.total,
            "dihedrals_cw1": self.dihedrals_cw1 / self.total,
        }


class PoseBustersValidity(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("valid", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_mols", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gen_ligs: list[Chem.rdchem.Mol],
        pdb_files: list,
        filter_pdbs: bool = False,
    ) -> None:

        total = len(gen_ligs)
        out = smolRD.sanitize_list(
            gen_ligs,
            pdbs=pdb_files,
            filter_pdb=filter_pdbs,
        )
        if isinstance(out, tuple):
            gen_ligs, pdb_files = out
        elif len(out) == 0:
            return

        validities = []
        for lig, pdb in zip(gen_ligs, pdb_files):
            assert isinstance(lig, Chem.Mol), "Expected a single ligand"
            lig = [lig]
            buster_dock = PoseBusters(config="dock")
            buster_dock_df = buster_dock.bust(lig, None, pdb)
            validities.extend(list(buster_dock_df.all(axis=1)))

        self.n_mols += total
        self.valid += sum(validities)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.n_mols


# TODO I don't think this will work with DDP
class Uniqueness(Metric):
    """Note: only tracks uniqueness of molecules which can be converted into SMILES"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_smiles = []

    def reset(self):
        self.valid_smiles = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [
            smolRD.smiles_from_mol(mol, canonical=True)
            for mol in mols
            if mol is not None
        ]
        valid_smiles = [smi for smi in smiles if smi is not None]
        self.valid_smiles.extend(valid_smiles)

    def compute(self) -> torch.Tensor:
        num_unique = len(set(self.valid_smiles))
        uniqueness = torch.tensor(num_unique) / len(self.valid_smiles)
        return uniqueness


class Novelty(GenerativeMetric):
    def __init__(self, existing_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)

        n_workers = min(4, len(os.sched_getaffinity(0)))
        # executor = ProcessPoolExecutor(max_workers=n_workers)
        executor = ThreadPoolExecutor(max_workers=n_workers)

        futures = [
            executor.submit(smolRD.smiles_from_mol, mol, canonical=True)
            for mol in existing_mols
        ]
        smiles = [future.result() for future in futures]
        smiles = [smi for smi in smiles if smi is not None]

        executor.shutdown()

        self.smiles = set(smiles)

        self.add_state("novel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [
            smolRD.smiles_from_mol(mol, canonical=True)
            for mol in mols
            if mol is not None
        ]
        valid_smiles = [smi for smi in smiles if smi is not None]
        novel = [smi not in self.smiles for smi in valid_smiles]

        self.novel += sum(novel)
        self.total += len(novel)

    def compute(self) -> torch.Tensor:
        return self.novel.float() / self.total


class EnergyValidity(GenerativeMetric):
    def __init__(self, optimise=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise

        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        num_mols = len(mols)

        if self.optimise:
            mols = [smolRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [smolRD.calc_energy(mol) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

        self.n_valid += len(valid_energies)
        self.total += num_mols

    def compute(self) -> torch.Tensor:
        return self.n_valid.float() / self.total


class AverageEnergy(GenerativeMetric):
    """Average energy for molecules for which energy can be calculated

    Note that the energy cannot be calculated for some molecules (specifically invalid ones) and the pose optimisation
    is not guaranteed to succeed. Molecules for which the energy cannot be calculated do not count towards the metric.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, optimise=False, per_atom=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise
        self.per_atom = per_atom

        self.add_state("energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_valid_energies", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        if self.optimise:
            mols = [smolRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [
            smolRD.calc_energy(mol, per_atom=self.per_atom)
            for mol in mols
            if mol is not None
        ]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

        self.energy += sum(valid_energies)
        self.n_valid_energies += len(valid_energies)

    def compute(self) -> torch.Tensor:
        return self.energy / self.n_valid_energies


class AverageStrainEnergy(GenerativeMetric):
    """
    The strain energy is the energy difference between a molecule's pose and its optimised pose. Estimated using RDKit.
    Only calculated when all of the following are true:
    1. The molecule is valid and an energy can be calculated
    2. The pose optimisation succeeds
    3. The energy can be calculated for the optimised pose

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results. Use the EnergyValidity metric with the optimise flag set to True to track the proportion of
    molecules for which this metric can be calculated.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, per_atom=False, **kwargs):
        super().__init__(**kwargs)

        self.per_atom = per_atom

        self.add_state(
            "total_energy_diff", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        opt_mols = [
            (idx, smolRD.optimise_mol(mol))
            for idx, mol in list(enumerate(mols))
            if mol is not None
        ]
        energies = [
            (idx, smolRD.calc_energy(mol, per_atom=self.per_atom))
            for idx, mol in opt_mols
            if mol is not None
        ]
        valids = [(idx, energy) for idx, energy in energies if energy is not None]

        if len(valids) == 0:
            return

        valid_indices, valid_energies = tuple(zip(*valids))
        original_energies = [
            smolRD.calc_energy(mols[idx], per_atom=self.per_atom)
            for idx in valid_indices
        ]
        energy_diffs = [
            orig - opt for orig, opt in zip(original_energies, valid_energies)
        ]

        self.total_energy_diff += sum(energy_diffs)
        self.n_valid += len(energy_diffs)

    def compute(self) -> torch.Tensor:
        return self.total_energy_diff / self.n_valid


class AverageOptRmsd(GenerativeMetric):
    """
    Average RMSD between a molecule and its optimised pose. Only calculated when all of the following are true:
    1. The molecule is valid
    2. The pose optimisation succeeds

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valids = [
            (idx, smolRD.optimise_mol(mol))
            for idx, mol in list(enumerate(mols))
            if mol is not None
        ]
        valids = [(idx, mol) for idx, mol in valids if mol is not None]

        if len(valids) == 0:
            return

        valid_indices, opt_mols = tuple(zip(*valids))
        original_mols = [mols[idx] for idx in valid_indices]
        rmsds = [
            smolRD.conf_distance(mol1, mol2)
            for mol1, mol2 in zip(original_mols, opt_mols)
        ]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid


class MolecularAccuracy(PairMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]
    ) -> None:
        predicted_smiles = [
            smolRD.smiles_from_mol(pred, canonical=True) for pred in predicted
        ]
        actual_smiles = [smolRD.smiles_from_mol(act, canonical=True) for act in actual]
        matches = [
            pred == act
            for pred, act in zip(predicted_smiles, actual_smiles)
            if act is not None
        ]

        self.n_correct += sum(matches)
        self.total += len(matches)

    def compute(self) -> torch.Tensor:
        return self.n_correct.float() / self.total


class MolecularPairRMSD(PairMetric):
    def __init__(self, fix_order: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.fix_order = fix_order
        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]
    ) -> None:
        valid_pairs = [
            (pred, act)
            for pred, act in zip(predicted, actual)
            if pred is not None and act is not None
        ]
        rmsds = [
            smolRD.conf_distance(pred, act, fix_order=self.fix_order)
            for pred, act in valid_pairs
        ]
        rmsds = [rmsd for rmsd in rmsds if rmsd is not None]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.tensor:
        return self.total_rmsd / self.n_valid


class MolecularPairShapeTanimotoSim(PairMetric):
    def __init__(self, align: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.align = align
        self.add_state(
            "total_shape_tanimoto_sim", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]
    ) -> None:
        valid_pairs = [
            (pred, act)
            for pred, act in zip(predicted, actual)
            if pred is not None and act is not None
        ]
        values = [
            shape_tanimoto_similarity(pred, act, align=self.align)
            for pred, act in valid_pairs
        ]
        values = [value for value in values if value is not None]

        self.total_shape_tanimoto_sim += sum(values)
        self.n_valid += len(values)

    def compute(self) -> torch.tensor:
        return self.total_shape_tanimoto_sim / self.n_valid


def calc_dihedrals(molecules):
    """
    Calculates dihedral angles (in degrees) for all rotatable bonds in a list of RDKit molecules.
    Assumes each molecule has at least one conformer with valid 3D coordinates.
    """
    dihedrals = []
    for mol in molecules:
        try:
            conf = mol.GetConformer()
        except ValueError:
            continue
        for bond in mol.GetBonds():
            if bond.IsInRing():
                continue
            if bond.GetBondType() != rdchem.BondType.SINGLE:
                continue

            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            if begin_atom.GetDegree() < 2 or end_atom.GetDegree() < 2:
                continue

            begin_neighbors = [
                nbr.GetIdx()
                for nbr in begin_atom.GetNeighbors()
                if nbr.GetIdx() != end_atom.GetIdx()
            ]
            end_neighbors = [
                nbr.GetIdx()
                for nbr in end_atom.GetNeighbors()
                if nbr.GetIdx() != begin_atom.GetIdx()
            ]

            if not begin_neighbors or not end_neighbors:
                continue

            idx1 = begin_neighbors[0]
            idx2 = begin_atom.GetIdx()
            idx3 = end_atom.GetIdx()
            idx4 = end_neighbors[0]

            try:
                angle = rdMolTransforms.GetDihedralDeg(conf, idx1, idx2, idx3, idx4)
                dihedrals.append(angle)
            except Exception:
                continue
    return dihedrals


def calc_circular_wasserstein_distance(molecules1, molecules2, nbins=36):
    """
    Calculates the circular Wasserstein distance between two dihedral angle distributions.

    The method bins the dihedral angles over the range [-180, 180] and computes the
    cumulative distribution functions (CDFs) for both. The optimal alignment over the circle
    is achieved by subtracting the median difference between the CDFs. The distance is then
    computed as the sum of absolute differences (multiplied by the bin width).

    Args:
        molecules1 (list): List of RDKit molecule objects for distribution 1.
        molecules2 (list): List of RDKit molecule objects for distribution 2.
        nbins (int): Number of bins to use for the histograms (default: 36, i.e., 10° per bin).

    Returns:
        float: The circular Wasserstein distance between the two distributions.
    """
    # Compute dihedral angles for each set of molecules
    angles1 = np.array(calc_dihedrals(molecules1))
    angles2 = np.array(calc_dihedrals(molecules2))

    # Define histogram bins for the periodic angles ([-180, 180])
    bins = np.linspace(-180, 180, nbins + 1)
    bin_width = bins[1] - bins[0]

    # Compute histograms (counts)
    hist1, _ = np.histogram(angles1, bins=bins)
    hist2, _ = np.histogram(angles2, bins=bins)

    # Normalize histograms to obtain probability mass functions.
    if hist1.sum() > 0:
        p1 = hist1.astype(float) / hist1.sum()
    else:
        p1 = np.zeros_like(hist1)

    if hist2.sum() > 0:
        p2 = hist2.astype(float) / hist2.sum()
    else:
        p2 = np.zeros_like(hist2)

    # Compute cumulative distributions (CDFs)
    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)

    # Calculate the difference between the CDFs
    diff = cdf1 - cdf2
    # The optimal circular alignment is found by subtracting the median difference
    shift = np.median(diff)
    # Circular Wasserstein distance is computed as the total absolute deviation, scaled by bin width
    distance = np.sum(np.abs(diff - shift)) * bin_width
    return distance
