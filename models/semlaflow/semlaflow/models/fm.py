import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, OneCycleLR

import semlaflow.util.functional as smolF
import semlaflow.util.rdkit as smolRD
from semlaflow.models.semla import MolecularGenerator
from semlaflow.util.molrepr import GeometricMol
from semlaflow.util.tokeniser import Vocabulary

_T = torch.Tensor
_BatchT = dict[str, _T]


def build_ode_time_grid(strategy: str, steps: int) -> list[float]:
    """
    Build increasing time knots in ``[0, 1]`` for flow-matching integration.

    Parameters
    ----------
    strategy : str
        ``"linear"`` for uniform spacing or ``"log"`` for logarithmic spacing.
    steps : int
        Number of integration intervals (grid has ``steps + 1`` points).

    Returns
    -------
    list of float
        Time values ``t_0, ..., t_steps`` with ``t_0 = 0`` and ``t_steps = 1``.

    Raises
    ------
    ValueError
        If ``strategy`` is not ``"linear"`` or ``"log"``.

    Examples
    --------
    >>> build_ode_time_grid("linear", 2)
    [0.0, 0.5, 1.0]
    """
    if strategy == "linear":
        return np.linspace(0, 1, steps + 1).tolist()

    if strategy == "log":
        time_points = (1 - np.geomspace(0.01, 1.0, steps + 1)).tolist()
        time_points.reverse()
        return time_points

    raise ValueError(f"Unknown ODE integration strategy '{strategy}'")


class Integrator:
    def __init__(
        self,
        steps,
        coord_noise_std=0.0,
        type_strategy="mask",
        bond_strategy="mask",
        cat_noise_level=0,
        type_mask_index=None,
        bond_mask_index=None,
        eps=1e-5,
    ):
        self._check_cat_sampling_strategy(type_strategy, type_mask_index, "type")
        self._check_cat_sampling_strategy(bond_strategy, bond_mask_index, "bond")

        self.steps = steps
        self.coord_noise_std = coord_noise_std
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.cat_noise_level = cat_noise_level
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.eps = eps
        # Avoid per-step torch.eye allocations in linear categorical flows (Heun calls these often).
        self._linear_atomics_eye_cache: Dict[Tuple, _T] = {}
        self._linear_bonds_eye_cache: Dict[Tuple, _T] = {}

    @property
    def hparams(self):
        return {
            "integration-steps": self.steps,
            "integration-coord-noise-std": self.coord_noise_std,
            "integration-type-strategy": self.type_strategy,
            "integration-bond-strategy": self.bond_strategy,
            "integration-cat-noise-level": self.cat_noise_level,
        }

    def coord_velocity(
        self, curr: _BatchT, predicted: _BatchT, t: _T, add_noise: bool = True
    ) -> _T:
        """
        Flow-matching coordinate velocity ``(x1_pred - x) / (1 - t)`` with optional noise.

        Parameters
        ----------
        curr : dict[str, Tensor]
            Current sample batch.
        predicted : dict[str, Tensor]
            Model predictions at the same time as ``curr``.
        t : Tensor
            Current time, shape ``(batch,)``.
        add_noise : bool, optional
            If ``True``, add Gaussian exploration noise scaled by ``coord_noise_std``.

        Returns
        -------
        Tensor
            Velocity field for coordinates, same shape as ``curr["coords"]``.

        Examples
        --------
        >>> # Velocity uses predicted vs current coordinates and time ``t``.
        >>> v = integrator.coord_velocity(curr, predicted, t)  # doctest: +SKIP
        """
        coord_velocity = (predicted["coords"] - curr["coords"]) / (1 - t.view(-1, 1, 1))
        if add_noise:
            coord_velocity = coord_velocity + torch.randn_like(coord_velocity) * self.coord_noise_std
        return coord_velocity

    def linear_atomics_increment(
        self,
        curr_atomics: _T,
        predicted: _BatchT,
        prior: _BatchT,
        step_size: float,
    ) -> _T:
        """
        Discrete-time increment for linear categorical atom flow (matches ``step``).

        Parameters
        ----------
        curr_atomics : Tensor
            Current atom distribution tensor.
        predicted : dict[str, Tensor]
            Model predictions (uses ``predicted["atomics"]``).
        prior : dict[str, Tensor]
            Prior noise one-hot tensor for atoms.
        step_size : float
            Time step ``h``.

        Returns
        -------
        Tensor
            Increment ``h * v`` to add to ``curr_atomics``.

        Examples
        --------
        >>> inc = integrator.linear_atomics_increment(
        ...     curr["atomics"], predicted, prior, step_size
        ... )  # doctest: +SKIP
        """
        device = curr_atomics.device
        dtype = curr_atomics.dtype
        vocab_size = predicted["atomics"].size(-1)
        key = (device, dtype, vocab_size)
        if key not in self._linear_atomics_eye_cache:
            self._linear_atomics_eye_cache[key] = torch.eye(
                vocab_size, device=device, dtype=dtype
            ).unsqueeze(0).unsqueeze(0)
        one_hots = self._linear_atomics_eye_cache[key]
        type_velocity = one_hots - prior["atomics"].unsqueeze(-1)
        type_velocity = (type_velocity * predicted["atomics"].unsqueeze(-2)).sum(-1)
        return step_size * type_velocity

    def linear_bonds_increment(
        self,
        curr_bonds: _T,
        predicted: _BatchT,
        prior: _BatchT,
        step_size: float,
    ) -> _T:
        """
        Discrete-time increment for linear categorical bond flow (matches ``step``).

        Parameters
        ----------
        curr_bonds : Tensor
            Current bond distribution tensor.
        predicted : dict[str, Tensor]
            Model predictions (uses ``predicted["bonds"]``).
        prior : dict[str, Tensor]
            Prior noise one-hot tensor for bonds.
        step_size : float
            Time step ``h``.

        Returns
        -------
        Tensor
            Increment ``h * v`` to add to ``curr_bonds``.

        Examples
        --------
        >>> inc = integrator.linear_bonds_increment(
        ...     curr["bonds"], predicted, prior, step_size
        ... )  # doctest: +SKIP
        """
        device = curr_bonds.device
        dtype = curr_bonds.dtype
        n_bonds = predicted["bonds"].size(-1)
        key = (device, dtype, n_bonds)
        if key not in self._linear_bonds_eye_cache:
            self._linear_bonds_eye_cache[key] = torch.eye(
                n_bonds, device=device, dtype=dtype
            ).view(1, 1, 1, n_bonds, n_bonds)
        one_hots = self._linear_bonds_eye_cache[key]
        bond_velocity = one_hots - prior["bonds"].unsqueeze(-1)
        bond_velocity = (bond_velocity * predicted["bonds"].unsqueeze(-2)).sum(-1)
        return step_size * bond_velocity

    @staticmethod
    def blend_predicted_distributions(pred1: _BatchT, pred2: _BatchT) -> _BatchT:
        """
        Average probability distributions for approximate Heun on discrete samplers.

        Parameters
        ----------
        pred1 : dict[str, Tensor]
            First forward softmax outputs.
        pred2 : dict[str, Tensor]
            Second forward softmax outputs.

        Returns
        -------
        dict[str, Tensor]
            Blended ``coords``, ``atomics``, ``bonds``, ``charges``, and ``mask``.

        Examples
        --------
        >>> blended = Integrator.blend_predicted_distributions(p1, p2)  # doctest: +SKIP
        """
        atomics = F.normalize(pred1["atomics"] + pred2["atomics"], p=1, dim=-1)
        bonds = F.normalize(pred1["bonds"] + pred2["bonds"], p=1, dim=-1)
        charges = F.normalize(pred1["charges"] + pred2["charges"], p=1, dim=-1)
        return {
            "coords": 0.5 * (pred1["coords"] + pred2["coords"]),
            "atomics": atomics,
            "bonds": bonds,
            "charges": charges,
            "mask": pred1["mask"],
        }

    def step(
        self, curr: _BatchT, predicted: _BatchT, prior: _BatchT, t: _T, step_size: float
    ) -> _BatchT:
        # *** Coord update step ***
        coord_velocity = self.coord_velocity(curr, predicted, t, add_noise=True)
        coords = curr["coords"] + (step_size * coord_velocity)

        # *** Atom type update step ***
        if self.type_strategy == "linear":
            atomics = curr["atomics"] + self.linear_atomics_increment(
                curr["atomics"], predicted, prior, step_size
            )

        # Dirichlet refers to sampling from a dirichlet dist, not dirichlet FM
        elif self.type_strategy == "dirichlet":
            type_velocity = torch.distributions.Dirichlet(
                predicted["atomics"] + self.eps
            ).sample()
            atomics = curr["atomics"] + (step_size * type_velocity)

        # Masking strategy from Discrete Flow Models paper (https://arxiv.org/abs/2402.04997)
        elif self.type_strategy == "mask":
            atomics = self._mask_sampling_step(
                curr["atomics"],
                predicted["atomics"],
                t,
                self.type_mask_index,
                step_size,
            )

        # Uniform sampling strategy from Discrete Flow Models paper
        elif self.type_strategy == "uniform-sample":
            atomics = self._uniform_sample_step(
                curr["atomics"], predicted["atomics"], t, step_size
            )

        # *** Bond update step ***
        if self.type_strategy == "linear":
            bonds = curr["bonds"] + self.linear_bonds_increment(
                curr["bonds"], predicted, prior, step_size
            )

        elif self.type_strategy == "dirichlet":
            bond_velocity = torch.distributions.Dirichlet(
                predicted["bonds"] + self.eps
            ).sample()
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.bond_strategy == "mask":
            bonds = self._mask_sampling_step(
                curr["bonds"], predicted["bonds"], t, self.bond_mask_index, step_size
            )

        elif self.bond_strategy == "uniform-sample":
            bonds = self._uniform_sample_step(
                curr["bonds"], predicted["bonds"], t, step_size
            )

        updated = {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "mask": curr["mask"],
        }
        return updated

    def step_discrete_only(
        self,
        curr: _BatchT,
        predicted: _BatchT,
        prior: _BatchT,
        t: _T,
        step_size: float,
    ) -> _BatchT:
        """
        Advance only categorical atomics and bonds; leave coordinates unchanged.

        Used when coordinates are updated by an external solver (e.g. DPM-Solver++) while
        discrete channels follow the same Euler-style dynamics as :meth:`step`.

        Parameters
        ----------
        curr : dict[str, Tensor]
            Current sample batch.
        predicted : dict[str, Tensor]
            Model predictions at the same time as ``curr``.
        prior : dict[str, Tensor]
            Prior noise batch (same as in :meth:`step`).
        t : Tensor
            Current time, shape ``(batch,)``.
        step_size : float
            Time step ``h`` (match the flow time spacing used for discrete updates).

        Returns
        -------
        dict[str, Tensor]
            Updated batch with ``coords`` equal to ``curr["coords"]`` and new ``atomics`` /
            ``bonds``.

        Examples
        --------
        >>> out = integrator.step_discrete_only(
        ...     curr, predicted, prior, t, step_size
        ... )  # doctest: +SKIP
        """
        coords = curr["coords"]

        if self.type_strategy == "linear":
            atomics = curr["atomics"] + self.linear_atomics_increment(
                curr["atomics"], predicted, prior, step_size
            )

        elif self.type_strategy == "dirichlet":
            type_velocity = torch.distributions.Dirichlet(
                predicted["atomics"] + self.eps
            ).sample()
            atomics = curr["atomics"] + (step_size * type_velocity)

        elif self.type_strategy == "mask":
            atomics = self._mask_sampling_step(
                curr["atomics"],
                predicted["atomics"],
                t,
                self.type_mask_index,
                step_size,
            )

        elif self.type_strategy == "uniform-sample":
            atomics = self._uniform_sample_step(
                curr["atomics"], predicted["atomics"], t, step_size
            )

        if self.type_strategy == "linear":
            bonds = curr["bonds"] + self.linear_bonds_increment(
                curr["bonds"], predicted, prior, step_size
            )

        elif self.type_strategy == "dirichlet":
            bond_velocity = torch.distributions.Dirichlet(
                predicted["bonds"] + self.eps
            ).sample()
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.bond_strategy == "mask":
            bonds = self._mask_sampling_step(
                curr["bonds"], predicted["bonds"], t, self.bond_mask_index, step_size
            )

        elif self.bond_strategy == "uniform-sample":
            bonds = self._uniform_sample_step(
                curr["bonds"], predicted["bonds"], t, step_size
            )

        return {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "mask": curr["mask"],
        }

    def heun_step(
        self,
        curr: _BatchT,
        pred1: _BatchT,
        pred2: _BatchT,
        prior: _BatchT,
        t: _T,
        step_size: float,
        trial: Optional[_BatchT] = None,
    ) -> _BatchT:
        """
        Heun (RK2) update using two model evaluations per interval.

        Continuous coordinates and linear categorical flows use a trapezoidal velocity average.
        Stochastic discrete strategies use a single Euler-style step with blended predictions.

        Pass the same ``trial`` state used for the second model evaluation (Euler predictor). If
        ``trial`` is ``None``, it is computed via :meth:`step` (extra work and a second draw of
        coordinate noise vs. a precomputed trial).

        Parameters
        ----------
        curr : dict[str, Tensor]
            State at time ``t``.
        pred1 : dict[str, Tensor]
            Softmax predictions from the first forward pass at ``(curr, t)``.
        pred2 : dict[str, Tensor]
            Softmax predictions from the second forward at ``(trial, t + h)``.
        prior : dict[str, Tensor]
            Prior noise batch (same as in ``step``).
        t : Tensor
            Batch times before the step, shape ``(batch,)``.
        step_size : float
            Interval length ``h``.
        trial : dict[str, Tensor], optional
            Euler predictor state after one :meth:`step` from ``(curr, pred1)``. Must match the
            state passed to the second forward pass.

        Returns
        -------
        dict[str, Tensor]
            Updated state at ``t + h``.

        Examples
        --------
        >>> curr_next = integrator.heun_step(
        ...     curr, pred1, pred2, prior, t, step_size, trial=trial
        ... )  # doctest: +SKIP
        """
        if trial is None:
            trial = self.step(curr, pred1, prior, t, step_size)
        t_next = t + step_size

        need_discrete_blend = self.type_strategy in (
            "mask",
            "uniform-sample",
        ) or self.bond_strategy in ("mask", "uniform-sample")
        blended = (
            self.blend_predicted_distributions(pred1, pred2)
            if need_discrete_blend
            else None
        )

        v1 = self.coord_velocity(curr, pred1, t, add_noise=True)
        v2 = self.coord_velocity(trial, pred2, t_next, add_noise=False)
        coords = curr["coords"] + (0.5 * step_size) * (v1 + v2)

        if self.type_strategy == "linear":
            inc1 = self.linear_atomics_increment(
                curr["atomics"], pred1, prior, step_size
            )
            inc2 = self.linear_atomics_increment(
                trial["atomics"], pred2, prior, step_size
            )
            atomics = curr["atomics"] + 0.5 * (inc1 + inc2)
        elif self.type_strategy == "dirichlet":
            k1 = torch.distributions.Dirichlet(pred1["atomics"] + self.eps).sample()
            k2 = torch.distributions.Dirichlet(pred2["atomics"] + self.eps).sample()
            atomics = curr["atomics"] + (0.5 * step_size) * (k1 + k2)
        else:
            assert blended is not None
            if self.type_strategy == "mask":
                atomics = self._mask_sampling_step(
                    curr["atomics"],
                    blended["atomics"],
                    t,
                    self.type_mask_index,
                    step_size,
                )
            else:
                atomics = self._uniform_sample_step(
                    curr["atomics"], blended["atomics"], t, step_size
                )

        if self.type_strategy == "linear":
            incb1 = self.linear_bonds_increment(
                curr["bonds"], pred1, prior, step_size
            )
            incb2 = self.linear_bonds_increment(
                trial["bonds"], pred2, prior, step_size
            )
            bonds = curr["bonds"] + 0.5 * (incb1 + incb2)
        elif self.type_strategy == "dirichlet":
            kb1 = torch.distributions.Dirichlet(pred1["bonds"] + self.eps).sample()
            kb2 = torch.distributions.Dirichlet(pred2["bonds"] + self.eps).sample()
            bonds = curr["bonds"] + (0.5 * step_size) * (kb1 + kb2)
        else:
            assert blended is not None
            if self.bond_strategy == "mask":
                bonds = self._mask_sampling_step(
                    curr["bonds"],
                    blended["bonds"],
                    t,
                    self.bond_mask_index,
                    step_size,
                )
            else:
                bonds = self._uniform_sample_step(
                    curr["bonds"], blended["bonds"], t, step_size
                )

        return {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "mask": curr["mask"],
        }

    # TODO test with mask sampling
    def _mask_sampling_step(self, curr_dist, pred_dist, t, mask_index, step_size):
        n_categories = pred_dist.size(-1)

        pred = torch.distributions.Categorical(pred_dist).sample()
        curr = torch.argmax(curr_dist, dim=-1)

        ones = [1] * (len(pred.shape) - 1)
        times = t.view(-1, *ones)

        # Choose elements to unmask
        limit = step_size * (1 + (self.cat_noise_level * times)) / (1 - times)
        unmask = torch.rand_like(pred.float()) < limit
        unmask = unmask * (curr == mask_index)

        # Choose elements to mask
        mask = torch.rand_like(pred.float()) < step_size * self.cat_noise_level
        mask = mask * (curr != mask_index)
        mask[t + step_size >= 1.0] = 0.0

        # Applying unmasking and re-masking
        curr[unmask] = pred[unmask]
        curr[mask] = mask_index

        return smolF.one_hot_encode_tensor(curr, n_categories)

    def _uniform_sample_step(self, curr_dist, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        # Setup batched time tensor and noise tensor
        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = self.cat_noise_level

        # Off-diagonal step probs
        mult = (1 + ((2 * noise) * (n_categories - 1) * times)) / (1 - times)
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        # On-diagonal step probs
        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        # Sample and convert back to one-hot so that all strategies represent data the same way
        samples = torch.distributions.Categorical(step_probs).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories)

    def _check_cat_sampling_strategy(self, strategy, mask_index, name):
        if strategy not in ["linear", "dirichlet", "mask", "uniform-sample"]:
            raise ValueError(f"{name} sampling strategy '{strategy}' is not supported.")

        if strategy == "mask" and mask_index is None:
            raise ValueError(
                f"{name}_mask_index must be provided if using the mask sampling strategy."
            )


class MolBuilder:
    def __init__(self, vocab, n_workers=16):
        self.vocab = vocab
        self.n_workers = n_workers
        self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _startup(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(self.n_workers)

    def mols_from_smiles(self, smiles, explicit_hs=False):
        self._startup()
        futures = [
            self._executor.submit(smolRD.mol_from_smiles, smi, explicit_hs)
            for smi in smiles
        ]
        mols = [future.result() for future in futures]
        self.shutdown()
        return mols

    def mols_from_tensors(
        self,
        coords,
        atom_dists,
        mask,
        bond_dists=None,
        charge_dists=None,
        sanitise=True,
    ):
        extracted = self._extract_mols(
            coords, atom_dists, mask, bond_dists=bond_dists, charge_dists=charge_dists
        )

        self._startup()
        build_fn = partial(self._mol_from_tensors, sanitise=sanitise)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        mols = [future.result() for future in futures]
        self.shutdown()

        return mols

    # TODO move into from_tensors method of GeometricMolBatch
    def smol_from_tensors(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords, atom_dists, mask, bond_dists=bond_dists, charge_dists=charge_dists
        )

        # mol_dicts = {}
        # for mol_coords, atom_dist, bond_dist, charge_dist in extracted:
        #     mol = {
        #         "coords": mol_coords,
        #         "atomics": atom_dist,
        #         "bonds": bond_dist,
        #         "charges": charge_dist
        #     }
        #     mol_dicts.append(mol)

        self._startup()
        build_fn = partial(self._smol_from_tensors)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        smol_mols = [future.result() for future in futures]
        self.shutdown()

        return smol_mols

    def _mol_from_tensors(
        self, coords, atom_dists, bond_dists=None, charge_dists=None, sanitise=True
    ):
        tokens = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists) if bond_dists is not None else None
        charges = (
            self._mol_extract_charges(charge_dists)
            if charge_dists is not None
            else None
        )
        return smolRD.mol_from_atoms(
            coords.float().numpy(),
            tokens,
            bonds=bonds,
            charges=charges,
            sanitise=sanitise,
        )

    def _smol_from_tensors(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.size(0)

        charges = torch.tensor(self._mol_extract_charges(charge_dists))
        bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
        bond_types = bond_dists[bond_indices[:, 0], bond_indices[:, 1], :]

        mol = GeometricMol(coords, atom_dists, bond_indices, bond_types, charges)
        return mol

    def mol_stabilities(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords, atom_dists, mask, bond_dists=bond_dists, charge_dists=charge_dists
        )
        mol_atom_stabilities = [self.atom_stabilities(*items) for items in extracted]
        return mol_atom_stabilities

    def atom_stabilities(self, coords, atom_dists, bond_dists, charge_dists):
        # Stability scoring was removed with the central metrics package.
        return [True] * coords.shape[0]

    # Separate each molecule from the batch
    def _extract_mols(
        self, coords, atom_dists, mask, bond_dists=None, charge_dists=None
    ):
        coords_list = []
        atom_dists_list = []
        bond_dists_list = []
        charge_dists_list = []

        n_atoms = mask.sum(dim=1)
        for idx in range(coords.size(0)):
            mol_atoms = n_atoms[idx]
            mol_coords = coords[idx, :mol_atoms, :].cpu()
            mol_token_dists = atom_dists[idx, :mol_atoms, :].cpu()

            coords_list.append(mol_coords)
            atom_dists_list.append(mol_token_dists)

            if bond_dists is not None:
                mol_bond_dists = bond_dists[idx, :mol_atoms, :mol_atoms, :].cpu()
                bond_dists_list.append(mol_bond_dists)
            else:
                bond_dists_list.append(None)

            if charge_dists is not None:
                mol_charge_dists = charge_dists[idx, :mol_atoms, :].cpu()
                charge_dists_list.append(mol_charge_dists)
            else:
                charge_dists_list.append(None)

        zipped = zip(coords_list, atom_dists_list, bond_dists_list, charge_dists_list)
        return zipped

    # Take index with highest probability and convert to token
    def _mol_extract_atomics(self, atom_dists):
        vocab_indices = torch.argmax(atom_dists, dim=1).tolist()
        tokens = self.vocab.tokens_from_indices(vocab_indices)
        return tokens

    # Convert to atomic number bond list format
    def _mol_extract_bonds(self, bond_dists):
        bond_types = torch.argmax(bond_dists, dim=-1)
        bonds = smolF.bonds_from_adj(bond_types)
        return bonds.long().numpy()

    # Convert index from model to actual atom charge
    def _mol_extract_charges(self, charge_dists):
        charge_types = torch.argmax(charge_dists, dim=-1).tolist()
        charges = [smolRD.IDX_CHARGE_MAP[idx] for idx in charge_types]
        return np.array(charges)


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************


class MolecularCFM(L.LightningModule):
    def __init__(
        self,
        gen: MolecularGenerator,
        vocab: Vocabulary,
        lr: float,
        integrator: Integrator,
        coord_scale: float = 1.0,
        type_strategy: str = "ce",
        bond_strategy: str = "ce",
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        pairwise_metrics: bool = True,
        use_ema: bool = True,
        compile_model: bool = True,
        self_condition: bool = False,
        distill: bool = False,
        lr_schedule: str = "constant",
        sampling_strategy: str = "log",
        ode_solver: str = "euler",
        dpm_solver_order: int = 2,
        dpm_flow_shift: float = 1.0,
        warm_up_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        train_smiles: Optional[list[str]] = None,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if type_strategy not in ["mse", "ce", "mask"]:
            raise ValueError(
                f"Unsupported type training strategy '{type_strategy}'. "
                + "Supported are `mse`, `ce` or `mask`."
            )

        if bond_strategy not in ["ce", "mask"]:
            raise ValueError(
                f"Unsupported bond training strategy '{bond_strategy}'. Supported are `ce` or `mask`."
            )

        if lr_schedule not in ["constant", "one-cycle"]:
            raise ValueError(
                f"LR scheduler {lr_schedule} not supported. Supported are `constant` or `one-cycle`."
            )

        if lr_schedule == "one-cycle" and total_steps is None:
            raise ValueError(
                "total_steps must be provided when using the one-cycle LR scheduler."
            )

        if distill and (type_strategy == "mask" or bond_strategy == "mask"):
            raise ValueError(
                "Distilled training with masking strategy is not supported."
            )

        if lr_schedule == "one-cycle" and warm_up_steps is not None:
            print("Note: warm_up_steps is currently ignored if schedule is one-cycle")

        if ode_solver not in ["euler", "heun", "dpmpp"]:
            raise ValueError(
                f"Unsupported ode_solver '{ode_solver}'. Supported are `euler`, `heun`, and `dpmpp`."
            )

        self.gen = gen
        self.vocab = vocab
        self.lr = lr
        self.coord_scale = coord_scale
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.type_loss_weight = type_loss_weight
        self.bond_loss_weight = bond_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.pairwise_metrics = pairwise_metrics
        self.compile_model = compile_model
        self.self_condition = self_condition
        self.distill = distill
        self.lr_schedule = lr_schedule
        self.sampling_strategy = sampling_strategy
        self.ode_solver = ode_solver
        self.dpm_solver_order = dpm_solver_order
        self.dpm_flow_shift = dpm_flow_shift
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index

        builder = MolBuilder(vocab)

        if use_ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
            ema_gen = torch.optim.swa_utils.AveragedModel(gen, multi_avg_fn=avg_fn)

        if compile_model:
            self.gen = self._compile_model(gen)

        self.integrator = integrator
        self.builder = builder
        self.ema_gen = ema_gen if use_ema else None

        # Anything else passed into kwargs will also be saved
        hparams = {
            "lr": lr,
            "coord_scale": coord_scale,
            "type_loss_weight": type_loss_weight,
            "bond_loss_weight": bond_loss_weight,
            "type_strategy": type_strategy,
            "bond_strategy": bond_strategy,
            "self_condition": self_condition,
            "distill": distill,
            "lr_schedule": lr_schedule,
            "sampling_strategy": sampling_strategy,
            "ode_solver": ode_solver,
            "dpm_solver_order": dpm_solver_order,
            "dpm_flow_shift": dpm_flow_shift,
            "use_ema": use_ema,
            "compile_model": compile_model,
            "warm_up_steps": warm_up_steps,
            **gen.hparams,
            **integrator.hparams,
            **kwargs,
        }
        self.save_hyperparameters(hparams)

        # Sampling-only build: chemistry metrics live outside this reproduction stack.
        self.stability_metrics = None
        self.gen_metrics = None
        self.pair_metrics = None

        self._init_params()

        self.block_cache_enabled = False
        self.block_cache_num_layers = 0

    def set_block_cache(self, enabled: bool, num_layers: int) -> None:
        """
        Enable or disable block-level periodic caching for inference-only evaluation.

        When enabled, :meth:`forward` runs :meth:`SemlaGenerator.compute_prefix_state`
        (wrapped by the cache helper) and then :meth:`SemlaGenerator.complete_forward_from_prefix`
        so only the first ``num_layers`` dynamics layers are subject to caching.

        Parameters
        ----------
        enabled : bool
            Whether to use the prefix/suffix split for caching.
        num_layers : int
            Number of initial :class:`~semlaflow.models.semla.EquiMessagePassingLayer`
            modules to include in the cached prefix.

        Raises
        ------
        ValueError
            If ``num_layers`` is not in ``[1, n_dynamics_layers]`` when ``enabled`` is ``True``.
        """
        from semlaflow.models.semla import SemlaGenerator

        self.block_cache_enabled = bool(enabled)
        self.block_cache_num_layers = int(num_layers)
        if not self.block_cache_enabled:
            return
        gen = self.gen
        if not isinstance(gen, SemlaGenerator):
            raise ValueError("Block caching is only supported for SemlaGenerator architectures.")
        n_dyn = len(gen.dynamics.layers)
        if self.block_cache_num_layers < 1 or self.block_cache_num_layers > n_dyn:
            raise ValueError(
                f"block_cache num_layers must be in [1, {n_dyn}], got {self.block_cache_num_layers}."
            )

    def _inference_generator(self):
        """
        Return the underlying generator module used at inference (EMA weights if enabled).

        Returns
        -------
        MolecularGenerator
            ``ema_gen.module`` when EMA is used, otherwise ``gen``.
        """
        if not self.training and self.ema_gen is not None:
            return self.ema_gen.module
        return self.gen

    def forward(self, batch, t, training=False, cond_batch=None):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """

        coords = batch["coords"]
        atom_types = batch["atomics"]
        bonds = batch["bonds"]
        mask = batch["mask"]

        # Prepare invariant atom features
        times = t.view(-1, 1, 1).expand(-1, coords.size(1), -1)
        features = torch.cat((times, atom_types), dim=2)

        if getattr(self, "block_cache_enabled", False) and not training:
            gen = self._inference_generator()
            cond_kwargs = {}
            if cond_batch is not None:
                cond_kwargs = {
                    "cond_coords": cond_batch["coords"],
                    "cond_atomics": cond_batch["atomics"],
                    "cond_bonds": cond_batch["bonds"],
                }
            prefix_state = gen.compute_prefix_state(
                coords,
                features,
                edge_feats=bonds,
                atom_mask=mask,
                num_prefix_layers=self.block_cache_num_layers,
                **cond_kwargs,
            )
            return gen.complete_forward_from_prefix(
                prefix_state,
                self.block_cache_num_layers,
                mask,
            )

        # Whether to use the EMA version of the model or not
        if not training and self.ema_gen is not None:
            model = self.ema_gen
        else:
            model = self.gen

        if cond_batch is not None:
            out = model(
                coords,
                features,
                edge_feats=bonds,
                cond_coords=cond_batch["coords"],
                cond_atomics=cond_batch["atomics"],
                cond_bonds=cond_batch["bonds"],
                atom_mask=mask,
            )

        else:
            out = model(coords, features, edge_feats=bonds, atom_mask=mask)

        return out

    def training_step(self, batch, b_idx):
        _, data, interpolated, times = batch

        if self.distill:
            return self._distill_training_step(batch)

        cond_batch = None

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:
            cond_batch = {
                "coords": torch.zeros_like(interpolated["coords"]),
                "atomics": torch.zeros_like(interpolated["atomics"]),
                "bonds": torch.zeros_like(interpolated["bonds"]),
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_coords, cond_types, cond_bonds, _ = self(
                        interpolated, times, training=True, cond_batch=cond_batch
                    )
                    cond_batch = {
                        "coords": cond_coords,
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1),
                    }

        coords, types, bonds, charges = self(
            interpolated, times, training=True, cond_batch=cond_batch
        )
        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges,
        }

        losses = self._loss(data, interpolated, predicted)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(f"train-{name}", loss_val, on_step=True, logger=True)

        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True)

        return loss

    def on_train_batch_end(self, outputs, batch, b_idx):
        if self.ema_gen is not None:
            self.ema_gen.update_parameters(self.gen)

    def validation_step(self, batch, b_idx):
        # Training / metric validation is out of scope for this sampling-only tree.
        return None

    def on_validation_epoch_end(self):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def on_test_epoch_end(self):
        return None

    def predict_step(self, batch, batch_idx):
        prior, _, _, _ = batch
        gen_batch = self._generate(
            prior, self.integrator.steps, strategy=self.sampling_strategy, solver=self.ode_solver
        )
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=0.0,
        )

        if self.lr_schedule == "constant":
            warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
            scheduler = LinearLR(opt, start_factor=1e-2, total_iters=warm_up_steps)

        # TODO could use warm_up_steps to shift peak of one cycle
        elif self.lr_schedule == "one-cycle":
            scheduler = OneCycleLR(
                opt, max_lr=self.lr, total_steps=self.total_steps, pct_start=0.3
            )

        else:
            raise ValueError(
                "Only `constant` or `one-cycle` LR schedules are supported."
            )

        config = {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
        return config

    def _distill_training_step(self, batch):
        prior, data, interpolated, times = batch

        input_batch = prior
        cond_batch = None
        input_times = torch.zeros_like(times)

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:
            cond_batch = {
                "coords": torch.zeros_like(interpolated["coords"]),
                "atomics": torch.zeros_like(interpolated["atomics"]),
                "bonds": torch.zeros_like(interpolated["bonds"]),
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_coords, cond_types, cond_bonds, _ = self(
                        input_batch, input_times, training=True, cond_batch=cond_batch
                    )
                    cond_batch = {
                        "coords": cond_coords,
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1),
                    }

                input_batch = interpolated
                input_times = times

        coords, types, bonds, charges = self(
            input_batch, input_times, training=True, cond_batch=cond_batch
        )
        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges,
        }

        losses = self._distill_loss(data, predicted, times)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(f"train-{name}", loss_val, on_step=True, logger=True)

        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True)

        return loss

    def _compile_model(self, model):
        return model
        return torch.compile(
            model, dynamic=False, fullgraph=True, mode="reduce-overhead"
        )

    def _loss(self, data, interpolated, predicted):
        pred_coords = predicted["coords"]
        coords = data["coords"]
        mask = data["mask"].unsqueeze(2)

        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        coord_loss = (coord_loss * mask).mean(dim=(1, 2))

        type_loss = self._type_loss(data, interpolated, predicted)
        bond_loss = self._bond_loss(data, interpolated, predicted)
        charge_loss = self._charge_loss(data, predicted)

        coord_loss = coord_loss.mean()
        type_loss = type_loss.mean() * self.type_loss_weight
        bond_loss = bond_loss.mean() * self.bond_loss_weight
        charge_loss = charge_loss.mean() * self.charge_loss_weight

        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss,
        }
        return losses

    def _distill_loss(self, data, predicted, eps=1e-3):
        coords = data["coords"]
        atomics = data["atomics"]
        bonds = data["bonds"]
        mask = data["mask"].unsqueeze(2)

        pred_coords = predicted["coords"]
        pred_atomic_logits = predicted["atomics"]
        pred_bond_logits = predicted["bonds"]

        pred_atomic_dists = F.log_softmax(pred_atomic_logits, dim=-1)
        pred_bond_dists = F.log_softmax(pred_bond_logits, dim=-1)

        # When distilling data should already be given as a dist so use KL div for categoricals
        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        type_loss = F.kl_div(pred_atomic_dists, atomics, reduction="none")
        bond_loss = F.kl_div(pred_bond_dists, bonds, reduction="none")

        adj_matrix = smolF.adj_from_node_mask(mask.squeeze(-1), self_connect=True)
        n_atoms = mask.sum(dim=(1, 2)) + eps
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        coord_loss = (coord_loss * mask).mean(dim=(1, 2))
        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        bond_loss = (bond_loss * adj_matrix.unsqueeze(-1)).sum(dim=(1, 2, 3)) / n_bonds
        charge_loss = self._charge_loss(data, predicted)

        coord_loss = coord_loss.mean()
        type_loss = type_loss.mean() * self.type_loss_weight
        bond_loss = bond_loss.mean() * self.bond_loss_weight
        charge_loss = charge_loss.mean() * self.charge_loss_weight

        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss,
        }
        return losses

    def _type_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["atomics"]
        atomics_dist = data["atomics"]
        mask = data["mask"].unsqueeze(2)
        batch_size, num_atoms, _ = pred_logits.size()

        if self.type_strategy == "mse":
            type_loss = F.mse_loss(pred_logits, atomics_dist, reduction="none")
        else:
            atomics = torch.argmax(atomics_dist, dim=-1).flatten(0, 1)
            type_loss = F.cross_entropy(
                pred_logits.flatten(0, 1), atomics, reduction="none"
            )
            type_loss = type_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)

        n_atoms = mask.sum(dim=(1, 2)) + eps

        # If we are training with masking, only compute the loss on masked types
        if self.type_strategy == "mask":
            masked_types = (
                torch.argmax(interpolated["atomics"], dim=-1) == self.type_mask_index
            )
            n_atoms = masked_types.sum(dim=-1) + eps
            type_loss = type_loss * masked_types.float().unsqueeze(-1)

        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        return type_loss

    def _bond_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["bonds"]
        mask = data["mask"]
        bonds = torch.argmax(data["bonds"], dim=-1)
        batch_size, num_atoms, _, _ = pred_logits.size()

        bond_loss = F.cross_entropy(
            pred_logits.flatten(0, 2), bonds.flatten(0, 2), reduction="none"
        )
        bond_loss = bond_loss.unflatten(0, (batch_size, num_atoms, num_atoms))

        adj_matrix = smolF.adj_from_node_mask(mask, self_connect=True)
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        # Only compute loss on masked bonds if we are training with masking strategy
        if self.bond_strategy == "mask":
            masked_bonds = (
                torch.argmax(interpolated["bonds"], dim=-1) == self.bond_mask_index
            )
            n_bonds = masked_bonds.sum(dim=(1, 2)) + eps
            bond_loss = bond_loss * masked_bonds.float()

        bond_loss = (bond_loss * adj_matrix).sum(dim=(1, 2)) / n_bonds
        return bond_loss

    def _charge_loss(self, data, predicted, eps=1e-3):
        pred_logits = predicted["charges"]
        charges = data["charges"]
        mask = data["mask"]
        batch_size, num_atoms, _ = pred_logits.size()

        charges = torch.argmax(charges, dim=-1).flatten(0, 1)
        charge_loss = F.cross_entropy(
            pred_logits.flatten(0, 1), charges, reduction="none"
        )
        charge_loss = charge_loss.unflatten(0, (batch_size, num_atoms))

        n_atoms = mask.sum(dim=1) + eps
        charge_loss = (charge_loss * mask).sum(dim=1) / n_atoms
        return charge_loss

    def _softmax_prediction_dict(
        self,
        curr: _BatchT,
        coords: _T,
        type_logits: _T,
        bond_logits: _T,
        charge_logits: _T,
    ) -> _BatchT:
        """
        Package raw logits into a prediction dict used by the integrator.

        Parameters
        ----------
        curr : dict[str, Tensor]
            Current batch; ``mask`` is taken from ``curr["mask"]``.
        coords : Tensor
            Predicted coordinates from the generator.
        type_logits : Tensor
            Atom type logits.
        bond_logits : Tensor
            Bond logits.
        charge_logits : Tensor
            Charge logits.

        Returns
        -------
        dict[str, Tensor]
            Keys ``coords``, ``atomics``, ``bonds``, ``charges``, ``mask``.

        Examples
        --------
        >>> pred = model._softmax_prediction_dict(
        ...     curr, coords, tl, bl, cl
        ... )  # doctest: +SKIP
        """
        type_probs = F.softmax(type_logits, dim=-1)
        bond_probs = F.softmax(bond_logits, dim=-1)
        charge_probs = F.softmax(charge_logits, dim=-1)
        return {
            "coords": coords,
            "atomics": type_probs,
            "bonds": bond_probs,
            "charges": charge_probs,
            "mask": curr["mask"],
        }

    def _generate_dpmpp(self, prior: _BatchT, steps: int) -> _BatchT:
        """
        Sample with DPM-Solver++ on coordinates and Euler-style discrete updates.

        Requires the ``diffusers`` package. Uses a flow-sigma schedule from
        ``DPMSolverMultistepScheduler`` (``use_flow_sigmas=True``,
        ``prediction_type="sample"``). The Hydra ``ode_sampling_strategy`` (linear/log) does
        not apply; step count is ``steps``.

        Parameters
        ----------
        prior : dict[str, Tensor]
            Prior noise batch.
        steps : int
            Number of scheduler steps (same meaning as ``integration_steps``).

        Returns
        -------
        dict[str, Tensor]
            Final softmax predictions at ``t = 1`` (coordinates scaled by ``coord_scale``).

        Raises
        ------
        ImportError
            If ``diffusers`` is not installed.

        Examples
        --------
        >>> out = model._generate_dpmpp(prior, 50)  # doctest: +SKIP
        """
        from semlaflow.util.dpm_solver_fm import (
            build_flow_dpm_scheduler,
            discrete_step_size_from_sigmas,
            dpmpp_coords_step,
        )

        scheduler = build_flow_dpm_scheduler(
            steps,
            solver_order=self.dpm_solver_order,
            flow_shift=self.dpm_flow_shift,
        )
        scheduler.set_timesteps(steps)
        scheduler.sigmas = scheduler.sigmas.to(device=self.device)

        batch_size = prior["coords"].size(0)
        curr = {k: v.clone() for k, v in prior.items()}
        cond_batch = {
            "coords": torch.zeros_like(prior["coords"]),
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"]),
        }

        start_time = time.time()
        with torch.inference_mode():
            for i in range(len(scheduler.timesteps)):
                cond = cond_batch if self.self_condition else None
                idx = (
                    scheduler._step_index
                    if scheduler._step_index is not None
                    else 0
                )
                sigma = scheduler.sigmas[idx]
                t_fm = torch.full(
                    (batch_size,),
                    float((1.0 - sigma).item()),
                    device=self.device,
                    dtype=torch.float32,
                )

                coords, tl, bl, cl = self(
                    curr, t_fm, training=False, cond_batch=cond
                )
                predicted = self._softmax_prediction_dict(
                    curr, coords, tl, bl, cl
                )
                del tl, bl, cl
                cond_batch = {
                    "coords": coords,
                    "atomics": predicted["atomics"],
                    "bonds": predicted["bonds"],
                }

                dt = discrete_step_size_from_sigmas(scheduler.sigmas, idx)
                sample_coords = curr["coords"]
                curr = self.integrator.step_discrete_only(
                    curr, predicted, prior, t_fm, dt
                )
                timestep = scheduler.timesteps[i]
                if isinstance(timestep, torch.Tensor):
                    timestep = timestep.to(device=self.device)
                coords_next = dpmpp_coords_step(
                    scheduler, coords, sample_coords, timestep
                )
                curr["coords"] = coords_next
                del predicted

            times = torch.ones(batch_size, device=self.device, dtype=torch.float32)
            cond = cond_batch if self.self_condition else None
            coords, type_logits, bond_logits, charge_logits = self(
                curr, times, training=False, cond_batch=cond
            )
            predicted = self._softmax_prediction_dict(
                curr, coords, type_logits, bond_logits, charge_logits
            )
            del type_logits, bond_logits, charge_logits

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        predicted["coords"] = predicted["coords"] * self.coord_scale
        return predicted

    def _generate(
        self,
        prior: _BatchT,
        steps: int = 100,
        strategy: str = "linear",
        solver: Optional[str] = None,
    ) -> _BatchT:
        """
        Sample molecules by integrating the flow from prior noise to ``t = 1``.

        Parameters
        ----------
        prior : dict[str, Tensor]
            Prior noise batch.
        steps : int, optional
            Number of ODE steps (intervals).
        strategy : str, optional
            Time grid: ``"linear"`` or ``"log"``. Ignored when ``solver`` is ``"dpmpp"``.
        solver : str, optional
            ``"euler"``, ``"heun"``, or ``"dpmpp"``. Defaults to ``self.ode_solver``.

        Returns
        -------
        dict[str, Tensor]
            Final softmax predictions at ``t = 1`` (coordinates scaled by ``coord_scale``).

        Examples
        --------
        >>> out = model._generate(prior, 100, strategy="log", solver="euler")  # doctest: +SKIP
        """
        if self.distill:
            return self._distill_generate(prior)

        solver = solver if solver is not None else self.ode_solver
        if solver not in ("euler", "heun", "dpmpp"):
            raise ValueError(
                f"Unknown ODE solver '{solver}'. Supported are 'euler', 'heun', and 'dpmpp'."
            )

        if solver == "dpmpp":
            return self._generate_dpmpp(prior, steps)

        time_points = build_ode_time_grid(strategy, steps)
        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        curr = {k: v.clone() for k, v in prior.items()}

        cond_batch = {
            "coords": torch.zeros_like(prior["coords"]),
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"]),
        }

        start_time = time.time()
        with torch.inference_mode():
            for step_size in step_sizes:
                cond = cond_batch if self.self_condition else None

                if solver == "euler":
                    coords, type_logits, bond_logits, charge_logits = self(
                        curr, times, training=False, cond_batch=cond
                    )
                    predicted = self._softmax_prediction_dict(
                        curr, coords, type_logits, bond_logits, charge_logits
                    )
                    del type_logits, bond_logits, charge_logits
                    cond_batch = {
                        "coords": coords,
                        "atomics": predicted["atomics"],
                        "bonds": predicted["bonds"],
                    }
                    curr = self.integrator.step(
                        curr, predicted, prior, times, step_size
                    )
                    times = times + step_size

                else:
                    coords1, tl1, bl1, cl1 = self(
                        curr, times, training=False, cond_batch=cond
                    )
                    pred1 = self._softmax_prediction_dict(
                        curr, coords1, tl1, bl1, cl1
                    )
                    del tl1, bl1, cl1
                    cond_batch = {
                        "coords": coords1,
                        "atomics": pred1["atomics"],
                        "bonds": pred1["bonds"],
                    }
                    trial = self.integrator.step(
                        curr, pred1, prior, times, step_size
                    )
                    times_next = times + step_size
                    if os.environ.get("SEMLAFLOW_HEUN_EMPTY_CACHE", "0") == "1":
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    cond2 = cond_batch if self.self_condition else None
                    coords2, tl2, bl2, cl2 = self(
                        trial, times_next, training=False, cond_batch=cond2
                    )
                    pred2 = self._softmax_prediction_dict(
                        trial, coords2, tl2, bl2, cl2
                    )
                    del tl2, bl2, cl2
                    cond_batch = {
                        "coords": coords2,
                        "atomics": pred2["atomics"],
                        "bonds": pred2["bonds"],
                    }
                    curr = self.integrator.heun_step(
                        curr,
                        pred1,
                        pred2,
                        prior,
                        times,
                        step_size,
                        trial=trial,
                    )
                    times = times_next
                    predicted = pred2
                    del coords1, trial
                    del pred1, pred2

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        predicted["coords"] = predicted["coords"] * self.coord_scale
        return predicted

    def _distill_generate(self, prior):
        cond_batch = {
            "coords": torch.zeros_like(prior["coords"]),
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"]),
        }

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        curr = {k: v.clone() for k, v in prior.items()}
        cond = cond_batch if self.self_condition else None

        coords, type_logits, bond_logits, charge_logits = self(
            curr, times, training=False, cond_batch=cond
        )

        type_probs = F.softmax(type_logits, dim=-1)
        bond_probs = F.softmax(bond_logits, dim=-1)
        charge_probs = F.softmax(charge_logits, dim=-1)

        predicted = {
            "coords": coords,
            "atomics": type_probs,
            "bonds": bond_probs,
            "charges": charge_probs,
            "mask": curr["mask"],
        }

        if self.self_condition:
            curr = self.integrator.step(curr, predicted, prior, times, 0.5)
            times = times + 0.5
            cond_batch = {"coords": coords, "atomics": type_probs, "bonds": bond_probs}
            coords, type_logits, bond_logits, charge_logits = self(
                curr, times, training=False, cond_batch=cond
            )

            type_probs = F.softmax(type_logits, dim=-1)
            bond_probs = F.softmax(bond_logits, dim=-1)
            charge_probs = F.softmax(charge_logits, dim=-1)

            predicted = {
                "coords": coords,
                "atomics": type_probs,
                "bonds": bond_probs,
                "charges": charge_probs,
                "mask": curr["mask"],
            }

        predicted["coords"] = predicted["coords"] * self.coord_scale
        return predicted

    def _generate_mols(self, generated, sanitise=True):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]

        mols = self.builder.mols_from_tensors(
            coords,
            atom_dists,
            masks,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise,
        )
        return mols

    def _generate_stabilities(self, generated):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]
        stabilities = self.builder.mol_stabilities(
            coords, atom_dists, masks, bond_dists, charge_dists
        )
        return stabilities

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
