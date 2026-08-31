"""Util file for Equinv scripts"""

import resource
from pathlib import Path

import numpy as np
import torch
from openbabel import pybel
from rdkit import RDLogger
from tqdm import tqdm

import flowr_model.util.functional as smolF
import flowr_model.util.rdkit as smolRD
from flowr_model.util.tokeniser import (
    Vocabulary,
    pocket_atom_names_plinder,
    pocket_residue_names_plinder,
)

# Declarations to be used in scripts
QM9_COORDS_STD_DEV = 1.723299503326416
GEOM_COORDS_STD_DEV = 2.407038688659668
PLINDER_COORDS_STD_DEV = 3.5152788162231445  # 2.8421707153320312
CROSSDOCKED_COORDS_STD_DEV = 2.882481107711792
KINODATA_COORDS_STD_DEV = 3.2349061965942383
BINDINGMOAD_COORDS_STD_DEV = 2.882481107711792


QM9_BUCKET_LIMITS = [12, 16, 18, 20, 22, 24, 30]
GEOM_DRUGS_BUCKET_LIMITS = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 96, 192]
PLINDER_BUCKET_LIMITS = [
    100,
    140,
    180,
    200,
    220,
    230,
    240,
    250,
    260,
    280,
    320,
    360,
    400,
    440,
    1000,
]
KINODATA_BUCKET_LIMITS = [
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    320,
    600,
]
CROSSDOCKED_BUCKET_LIMITS = [
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    320,
    400,
    600,
    1500,
]
BINDINGMOAD_BUCKET_LIMITS = [
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    320,
    600,
]

PROJECT_PREFIX = "equinv"
BOND_MASK_INDEX = 5
COMPILER_CACHE_SIZE = 128


def disable_lib_stdout():
    pybel.ob.obErrorLog.StopLogging()
    RDLogger.DisableLog("rdApp.*")


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Need to ensure the limits are large enough when using OT since lots of preprocessing needs to be done on the batches
# OT seems to cause a problem when there are not enough allowed open FDs
def configure_fs(limit=4096):
    """
    Try to increase the limit on open file descriptors
    If not possible use a different strategy for sharing files in torch
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            print("Limit changed successfully!")

        except:
            print(
                "Limit change unsuccessful. Using torch file_system file sharing strategy instead."
            )

            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        print("Open file limit already sufficiently large.")


# Applies the following transformations to a molecule:
# 1. Scales coordinate values by 1 / coord_std (so that they are standard normal)
# 2. Applies a random rotation to the coordinates
# 3. Removes the centre of mass of the molecule
# 4. Creates a one-hot vector for the atomic numbers of each atom
# 5. Creates a one-hot vector for the bond type for every possible bond
# 6. Encodes charges as non-negative numbers according to encoding map
def mol_transform(molecule, vocab, n_bonds, coord_std, rotate=True, zero_com=True):
    if coord_std != 1.0:
        molecule = molecule.scale(1.0 / coord_std)
    if rotate:
        rotation = tuple(np.random.rand(3) * np.pi * 2)
        molecule = molecule.rotate(rotation)
    if zero_com:
        molecule = molecule.zero_com()

    atomic_nums = [int(atomic) for atomic in molecule.atomics.tolist()]
    tokens = [smolRD.PT.symbol_from_atomic(atomic) for atomic in atomic_nums]
    one_hot_atomics = torch.tensor(vocab.indices_from_tokens(tokens, one_hot=True))

    bond_types = smolF.one_hot_encode_tensor(molecule.bond_types, n_bonds)

    charge_idxs = [
        smolRD.CHARGE_IDX_MAP[charge] for charge in molecule.charges.tolist()
    ]
    charge_idxs = torch.tensor(charge_idxs)

    transformed = molecule._copy_with(
        atomics=one_hot_atomics, bond_types=bond_types, charges=charge_idxs
    )
    return transformed


def complex_transform(
    pocket_complex,
    vocab,
    n_bonds,
    coord_std,
    remove_hs=False,
    pocket_noise="apo",
    use_interactions=False,
):
    assert coord_std == 1.0, (
        "coord_std must be 1.0 for complex transform for now (e.g., full_pocket scale is not supported yet"
    )

    holo_pocket = pocket_complex.holo
    apo_pocket = pocket_complex.apo
    ligand = pocket_complex.ligand

    # *** Transform LIGAND *** #
    lig_trans = mol_transform(
        ligand, vocab, n_bonds, coord_std, rotate=False, zero_com=False
    )
    # *** Transform HOLO *** #
    if holo_pocket is not None:
        holo_mol_trans = mol_transform(
            holo_pocket.mol, vocab, n_bonds, coord_std, rotate=False, zero_com=False
        )
        holo_pocket = holo_pocket._copy_with(mol=holo_mol_trans)
    # *** Transform APO *** #
    if apo_pocket is not None:
        apo_mol_trans = mol_transform(
            apo_pocket.mol, vocab, n_bonds, coord_std, rotate=False, zero_com=False
        )
        apo_pocket = apo_pocket._copy_with(mol=apo_mol_trans)

    # *** Transform COMPLEX *** #
    if pocket_noise in ["fix", "random"]:
        assert holo_pocket is not None, (
            "Holo must be provided for rigid or random pocket flow matching"
        )
        trans_complex = pocket_complex._copy_with(lig_trans, holo=holo_pocket)
        if pocket_noise == "random":
            trans_complex = trans_complex.move_holo_and_lig_to_holo_lig_com()
        else:
            trans_complex = trans_complex.move_holo_and_lig_to_holo_com()
    elif pocket_noise == "apo":
        assert apo_pocket is not None, "apo must be provided for apo-holo flow matching"
        if holo_pocket is None:
            # NOTE: This should only happen at inference when just a apo structure is provided
            holo_pocket = apo_pocket._copy_with()
        trans_complex = pocket_complex._copy_with(
            lig_trans, holo=holo_pocket, apo=apo_pocket
        )
        trans_complex = trans_complex.move_apo_and_holo_and_lig_to_apo_com()
    else:
        raise ValueError(
            f"Invalid pocket noise type {pocket_noise}. Must be one of ['fix', 'apo', 'random']"
        )
    if use_interactions:
        # *** Transform INTERACTIONS *** #
        # Add a one-hot vector for the interaction type (N_pocket, N_lig, num_interactions + 1)
        # where no interaction is encoded as the first index
        interactions = trans_complex.interactions
        n_pocket, n_lig, n_interactions = interactions.shape
        interactions_arr = np.zeros((n_pocket, n_lig, n_interactions + 1))
        interactions_arr[:, :, 1:] = interactions
        interactions_flat = interactions_arr.reshape(
            n_pocket * n_lig, n_interactions + 1
        )
        interactions_flat = np.argmax(
            interactions_flat, axis=-1
        )  # to get no interaction class at index 0
        interactions_arr = smolF.one_hot_encode_tensor(
            torch.from_numpy(interactions_flat), n_interactions + 1
        )
        interactions_arr = interactions_arr.reshape(
            n_pocket, n_lig, -1
        )  # (N_pocket, N_lig, num_interactions + 1)
        trans_complex.interactions = interactions_arr

    return trans_complex


def get_n_bond_types(cat_strategy):
    n_bond_types = len(smolRD.BOND_IDX_MAP.keys()) + 1
    n_bond_types = n_bond_types + 1 if cat_strategy == "mask" else n_bond_types
    return n_bond_types


def build_vocab(remove_hs=False):
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>"]
    core_atoms = [
        "H",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Si",
        "P",
        "S",
        "Cl",
        "Se",
        "Br",
        "I",
    ]
    if remove_hs:
        core_atoms = core_atoms[1:]
    tokens = special_tokens + core_atoms
    return Vocabulary(tokens)


def _build_vocab():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>", "<MASK>"]
    core_atoms = [
        "H",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "As",
        "Se",
        "Br",
        "I",
        "Hg",
        "Bi",
    ]
    tokens = special_tokens + core_atoms
    return Vocabulary(tokens)


def _build_vocab_pocket_atoms():
    special_token = ["<PAD>"]
    ligand = ["LIG"]
    tokens = special_token + ligand + pocket_atom_names_plinder
    return Vocabulary(tokens)


def _build_vocab_pocket_res(pocket_noise="apo"):
    special_token = ["<PAD>"]
    ligand = ["LIG"]
    tokens = special_token + ligand + pocket_residue_names_plinder
    return Vocabulary(tokens)


# TODO support multi gpus
def generate_molecules(model, dm, steps, strategy, stabilities=False):
    test_dl = dm.test_dataloader()
    model.eval()
    cuda_model = model.to("cuda")

    outputs = []
    for i, batch in tqdm(enumerate(test_dl)):
        batch = {k: v.cuda() for k, v in batch[0].items()}
        output = cuda_model._generate(batch, steps, strategy, iter=i, save_traj=True)
        outputs.append(output)

    molecules = [cuda_model._generate_mols(output) for output in outputs]
    molecules = [mol for mol_list in molecules for mol in mol_list]

    if not stabilities:
        return molecules, outputs

    stabilities = [cuda_model._generate_stabilities(output) for output in outputs]
    stabilities = [mol_stab for mol_stabs in stabilities for mol_stab in mol_stabs]
    return molecules, outputs, stabilities


def generate_ligands_per_target(args, hparams, model, batch, save_traj=False, iter=""):
    prior, data, interpolated, _ = batch

    assert len(set([cmpl.metadata["system_id"] for cmpl in data["complex"]])) == 1, (
        "Sampling N ligands per target, but found mixed targets"
    )

    # get ligand and pocket data
    lig_prior = model.builder.extract_ligand_from_complex(prior)
    lig_prior["interactions"] = prior["interactions"]
    lig_prior["fragment_mask"] = prior["fragment_mask"]
    lig_prior = {k: v.cuda() if torch.is_tensor(v) else v for k, v in lig_prior.items()}
    if args.arch == "pocket_flex" and hparams["pocket_noise"] == "apo":
        pocket_prior = model.builder.extract_pocket_from_complex(prior)
    else:
        pocket_prior = model.builder.extract_pocket_from_complex(data)
    pocket_prior["interactions"] = data["interactions"]
    pocket_prior["complex"] = data["complex"]
    pocket_prior = {
        k: v.cuda() if torch.is_tensor(v) else v for k, v in pocket_prior.items()
    }

    # specify times
    lig_times = torch.zeros(prior["coords"].size(0), device="cuda")
    pocket_times = torch.zeros(pocket_prior["coords"].size(0), device="cuda")
    interaction_times = torch.zeros(prior["coords"].size(0), device="cuda")
    prior_times = [lig_times, pocket_times, interaction_times]

    # run generation N times
    output = model._generate(
        lig_prior,
        pocket_prior,
        times=prior_times,
        steps=args.integration_steps,
        strategy=args.ode_sampling_strategy,
        iter=iter,
        corr_iters=args.corrector_iters,
        save_traj=save_traj,
    )
    # generate molecules
    gen_ligs = model._generate_mols(output)
    if args.arch == "pocket_flex":
        pocket_prior = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in pocket_prior.items()
        }
        gen_pdbs = model.retrieve_pdbs(
            pocket_prior,
            coords=output["pocket_coords"],
            save_dir=Path(args.save_dir) / "gen_pdbs",
            iter=iter,
        )
        return gen_ligs, gen_pdbs
    return gen_ligs


