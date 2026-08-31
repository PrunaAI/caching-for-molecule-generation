"""Main util file for all scripts"""

import copy
import resource
from typing import Optional

import numpy as np
import rdkit
import torch
from openbabel import openbabel as ob
from rdkit import RDLogger

import flowr_root_model.constants as constants
import flowr_root_model.util.functional as smolF
import flowr_root_model.util.rdkit as smolRD
from flowr_root_model.models.integrator import Integrator
from flowr_root_model.util.pocket import PROLIF_INTERACTIONS
from flowr_root_model.util.tokeniser import (
    Vocabulary,
    pocket_atom_names,
    pocket_residue_names,
)

# from flowr_root_model.models.fm_mol import LigandCFM
# from flowr_root_model.models.fm_pocket_flex import LigandPocketFlexCFM
LigandCFM = LigandPocketFlexCFM = None


BOND_MASK_INDEX = 5
COMPILER_CACHE_SIZE = 128


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ******************************* UTILS ***************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


def disable_lib_stdout():
    ob.obErrorLog.StopLogging()
    RDLogger.DisableLog("rdApp.*")


# bfloat16 training produced significantly worse models than full so use default 16-bit instead
    # return "16-mixed" if args.mixed_precision else "32"


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

        except Exception:
            print(
                "Limit change unsuccessful. Using torch file_system file sharing strategy instead."
            )

            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        print("Open file limit already sufficiently large.")


def mol_transform(
    molecule,
    vocab,
    vocab_charges,
    n_bonds,
    vocab_hybridization=None,
    vocab_aromatic=None,
    coord_std=1.0,
    rotate=False,
    zero_com=False,
):
    """
    Transform a molecule into a format suitable for model input.
    # Applies the following optional transformations to a molecule:
        # 1. Scales coordinate values by 1 / coord_std (so that they are standard normal)
        # 2. Applies a random rotation to the coordinates
        # 3. Removes the centre of mass of the molecule
        # 4. Creates a one-hot vector for the atomic numbers of each atom
        # 5. Creates a one-hot vector for the bond type for every possible bond
        # 6. Encodes charges as non-negative numbers according to encoding map
    """
    if zero_com:
        molecule = molecule.zero_com()
    if coord_std != 1.0:
        molecule = molecule.scale(1.0 / coord_std)
    if rotate:
        from scipy.spatial.transform import Rotation

        rotation = Rotation.random()
        molecule = molecule.rotate(rotation)

    atomic_nums = [int(atomic) for atomic in molecule.atomics.tolist()]
    tokens = [smolRD.PT.symbol_from_atomic(atomic) for atomic in atomic_nums]
    atomics = torch.tensor(vocab.indices_from_tokens(tokens, one_hot=True))

    bond_types = smolF.one_hot_encode_tensor(molecule.bond_types, n_bonds)

    charges = [int(charge) for charge in molecule.charges.tolist()]
    charges = torch.tensor(vocab_charges.indices_from_tokens(charges, one_hot=True))

    if vocab_hybridization is not None:
        hybridization = [
            smolRD.IDX_ADD_FEAT_MAP["hybridization"][int(hybrid)]
            for hybrid in molecule.hybridization.tolist()
        ]
        hybridization = torch.tensor(
            vocab_hybridization.indices_from_tokens(hybridization, one_hot=True)
        )
    else:
        hybridization = None
    if vocab_aromatic is not None:
        is_aromatic = [
            smolRD.IDX_ADD_FEAT_MAP["is_aromatic"][int(aromatic)]
            for aromatic in molecule.aromaticity.tolist()
        ]
        is_aromatic = torch.tensor(
            vocab_aromatic.indices_from_tokens(is_aromatic, one_hot=True)
        )
    else:
        is_aromatic = None

    transformed = molecule._copy_with(
        atomics=atomics, bond_types=bond_types, charges=charges
    )
    if hybridization is not None:
        transformed = transformed._copy_with(hybridization=hybridization)
    else:
        transformed = transformed._copy_with(hybridization=None)
    if vocab_aromatic is not None:
        transformed = transformed._copy_with(is_aromatic=is_aromatic)
    else:
        transformed = transformed._copy_with(is_aromatic=None)

    return transformed


def complex_transform(
    pocket_complex,
    vocab,
    vocab_charges,
    n_bonds,
    vocab_hybridization=None,
    vocab_aromatic=None,
    coord_std: float = 1.0,
    pocket_noise: str = "apo",
    pocket_noise_std: float = 0.02,
    use_interactions: bool = False,
    rotate_complex: bool = False,
):
    assert coord_std == 1.0, "coord_std must be 1.0 for complex transform for now"

    holo_pocket = pocket_complex.holo
    apo_pocket = pocket_complex.apo
    ligand = pocket_complex.ligand

    # *** Transform LIGAND *** #
    lig_trans = mol_transform(
        ligand,
        vocab,
        vocab_charges,
        n_bonds,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        coord_std=coord_std,
        rotate=False,
        zero_com=False,
    )

    # *** Transform HOLO *** #
    if holo_pocket is not None:
        holo_mol_trans = mol_transform(
            holo_pocket.mol,
            vocab,
            vocab_charges,
            n_bonds,
            coord_std=coord_std,
            rotate=False,
            zero_com=False,
        )
        holo_pocket = holo_pocket._copy_with(mol=holo_mol_trans)

    # *** Transform APO *** #
    if apo_pocket is not None:
        apo_mol_trans = mol_transform(
            apo_pocket.mol,
            vocab,
            vocab_charges,
            n_bonds,
            coord_std=coord_std,
            rotate=False,
            zero_com=False,
        )
        apo_pocket = apo_pocket._copy_with(mol=apo_mol_trans)

    # *** Transform COMPLEX *** #
    if pocket_noise == "fix":
        assert holo_pocket is not None, "Holo must be provided for rigid flow matching"
        trans_complex = pocket_complex._copy_with(lig_trans, holo=holo_pocket)
        trans_complex = trans_complex.move_holo_and_lig_to_holo_com()
    elif pocket_noise == "random":
        assert (
            holo_pocket is not None
        ), "Holo must be provided for random pocket flow matching"
        coords = (
            holo_pocket.mol.coords
            + torch.randn_like(holo_pocket.mol.coords) * pocket_noise_std
        )
        apo_pocket_mol = holo_pocket.mol._copy_with(coords=coords)
        apo_pocket = holo_pocket._copy_with(mol=apo_pocket_mol)
        trans_complex = pocket_complex._copy_with(
            lig_trans, holo=holo_pocket, apo=apo_pocket
        )
        trans_complex = trans_complex.move_apo_and_holo_and_lig_to_apo_com()
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

    # *** Transform INTERACTIONS *** #
    if use_interactions:
        # Add a one-hot vector for the interaction type (N_pocket, N_lig, num_interactions + 1)
        # where no interaction is encoded as the first index
        interactions = trans_complex.interactions
        if interactions is not None:
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

    if rotate_complex:
        trans_complex = trans_complex.rotate()

    return trans_complex


def get_n_bond_types(cat_strategy):
    n_bond_types = len(smolRD.BOND_IDX_MAP.keys()) + 1
    n_bond_types = n_bond_types + 1 if cat_strategy == "mask" else n_bond_types
    return n_bond_types


def _build_vocab():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>"]
    tokens = special_tokens + constants.CORE_ATOMS
    return Vocabulary(tokens)


def _build_vocab_charges():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>"]
    charge_tokens = [0, 1, 2, 3, -1, -2, -3]
    tokens = special_tokens + charge_tokens
    return Vocabulary(tokens)


def _build_vocab_hybridization():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>"]
    hybridization_tokens = [
        rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        rdkit.Chem.rdchem.HybridizationType.S,
        rdkit.Chem.rdchem.HybridizationType.SP,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        rdkit.Chem.rdchem.HybridizationType.SP3,
        rdkit.Chem.rdchem.HybridizationType.SP2D,
        rdkit.Chem.rdchem.HybridizationType.SP3D,
        rdkit.Chem.rdchem.HybridizationType.SP3D2,
        rdkit.Chem.rdchem.HybridizationType.OTHER,
    ]
    tokens = special_tokens + hybridization_tokens
    return Vocabulary(tokens)


def _build_vocab_pocket_atoms():
    special_token = ["<PAD>"]
    ligand = ["LIG"]
    tokens = special_token + ligand + pocket_atom_names
    return Vocabulary(tokens)


def _build_vocab_pocket_res(pocket_noise="apo"):
    special_token = ["<PAD>"]
    ligand = ["LIG"]
    tokens = special_token + ligand + pocket_residue_names
    return Vocabulary(tokens)


# Function to recursively inject LoRA layers
def _inject_lora(lora_rank: int, lora_alpha: float, mod: torch.nn.Module):
    from flowr_root_model.models.lora import LinearWithLoRA

    for name, child in mod.named_children():
        # wrap any pure Linear
        if isinstance(child, torch.nn.Linear):
            setattr(
                mod,
                name,
                LinearWithLoRA(child, rank=lora_rank, alpha=lora_alpha),
            )
        else:
            _inject_lora(lora_rank, lora_alpha, child)


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# **************************** BUILD TRAINER **********************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ******************************* BUILD MODEL *********************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


def load_model(
    args,
    ckpt_path: str = None,
    return_info: bool = True,
    dataset_info: Optional[dict] = None,
):
    checkpoint = torch.load(
        args.ckpt_path if ckpt_path is None else ckpt_path, map_location="cpu"
    )
    hparams = dotdict(checkpoint["hyper_parameters"])
    hparams["compile_model"] = False
    # Set sampling hyperparameters
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy
    hparams["use_inpaint_mode_embed"] = (
        hparams.get("scaffold_hopping", False)
        or hparams.get("scaffold_elaboration", False)
        or hparams.get("interaction_conditional", False)
        or hparams.get("core_growing", False)
        or hparams.get("linker_inpainting", False)
        or hparams.get("fragment_inpainting", False)
        or hparams.get("fragment_growing", False)
        or hparams.get("substructure_inpainting", False)
    )
    hparams["interaction_conditional"] = args.interaction_conditional
    hparams["scaffold_hopping"] = args.scaffold_hopping
    hparams["scaffold_elaboration"] = args.scaffold_elaboration
    hparams["linker_inpainting"] = args.linker_inpainting
    hparams["core_growing"] = args.core_growing
    hparams["fragment_inpainting"] = args.fragment_inpainting
    hparams["fragment_growing"] = args.fragment_growing
    hparams["substructure_inpainting"] = args.substructure_inpainting
    hparams["substructure"] = args.substructure
    hparams["data_path"] = args.data_path
    hparams["save_dir"] = args.save_dir
    # Set optimizer hyperparameters
    hparams["lr"] = args.lr if getattr(args, "lr", None) else hparams.get("lr", 1e-4)
    hparams["lr_schedule"] = (
        args.lr_schedule
        if getattr(args, "lr_schedule", None) is not None
        else hparams.get("lr_schedule", "exponential")
    )
    hparams["cosine_decay_fraction"] = (
        args.cosine_decay_fraction
        if getattr(args, "cosine_decay_fraction", None) is not None
        else hparams.get("cosine_decay_fraction", 1.0)
    )
    hparams["lr_gamma"] = (
        args.lr_gamma
        if getattr(args, "lr_gamma", None) is not None
        else hparams.get("lr_gamma", 0.995)
    )
    hparams["warm_up_steps"] = 0
    hparams["weight_decay"] = (
        args.weight_decay
        if getattr(args, "weight_decay", None) is not None
        else hparams.get("weight_decay", 1e-12)
    )
    hparams["beta1"] = (
        args.beta1
        if getattr(args, "beta1", None) is not None
        else hparams.get("beta1", 0.9)
    )
    hparams["beta2"] = (
        args.beta2
        if getattr(args, "beta2", None) is not None
        else hparams.get("beta2", 0.95)
    )
    # Set loss weights
    hparams["coord_loss_weight"] = getattr(args, "coord_loss_weight", None)
    hparams["type_loss_weight"] = getattr(args, "type_loss_weight", None)
    hparams["bond_loss_weight"] = getattr(args, "bond_loss_weight", None)
    hparams["charge_loss_weight"] = getattr(args, "charge_loss_weight", None)
    hparams["hybridization_loss_weight"] = getattr(
        args, "hybridization_loss_weight", None
    )
    hparams["distance_loss_weight_lig"] = getattr(
        args, "distance_loss_weight_lig", None
    )
    hparams["distance_loss_weight_lig_pocket"] = getattr(
        args, "distance_loss_weight_lig_pocket", None
    )
    hparams["bond_angle_loss_weight"] = getattr(args, "bond_angle_loss_weight", None)
    hparams["bond_angle_huber_delta"] = getattr(args, "bond_angle_huber_delta", None)
    hparams["dihedral_loss_weight"] = getattr(args, "dihedral_loss_weight", None)
    hparams["dihedral_huber_delta"] = getattr(args, "dihedral_huber_delta", None)
    hparams["bond_length_loss_weight"] = getattr(args, "bond_length_loss_weight", None)
    hparams["affinity_loss_weight"] = getattr(args, "affinity_loss_weight", None)
    hparams["docking_loss_weight"] = getattr(args, "docking_loss_weight", None)
    hparams["energy_loss_weight"] = getattr(args, "energy_loss_weight", None)
    hparams["energy_loss_weighting"] = getattr(args, "energy_loss_weighting", None)
    hparams["energy_loss_decay_rate"] = getattr(args, "energy_loss_decay_rate", None)

    # Number of corrector iterations
    if args.corrector_iters > 0:
        assert (
            args.categorical_strategy == "velocity-sample"
        ), "Only velocity sampling supported for corrector iterations."
        hparams["corrector_iters"] = args.corrector_iters

    print("Building model vocabs...")
    vocab = _build_vocab()
    vocab_charges = _build_vocab_charges()
    vocab_pocket_atoms = _build_vocab_pocket_atoms()
    vocab_pocket_res = _build_vocab_pocket_res()
    if hparams["add_feats"]:
        print("Including hybridization features...")
        vocab_hybridization = _build_vocab_hybridization()
        vocab_aromatic = None  # _build_vocab_aromatic()
    else:
        vocab_hybridization = None
        vocab_aromatic = None
    print("Vocabs complete.")

    if hparams["pocket_noise"] in ["fix", "random"]:
        assert (
            args.arch == "pocket"
        ), "Model trained on rigid pocket flow matching. Change arch to pocket."
        assert (
            args.pocket_type == "holo"
        ), "Model trained on rigid pocket flow matching. Change pocket_type to holo."
    if hparams["pocket_noise"] == "apo":
        assert (
            args.arch == "pocket_flex"
        ), "Model trained on apo pocket flow matching. Change arch to pocket_flex."
        assert (
            args.pocket_type == "apo"
        ), "Model trained on apo pocket flow matching. Change pocket_type to apo."

    n_atom_types = vocab.size
    n_bond_types = get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else None
    )
    # n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else None
    n_interaction_types = (
        len(PROLIF_INTERACTIONS) + 1
        if hparams["flow_interactions"] or hparams["predict_interactions"]
        else None
    )

    # Build the EGNN generator
    if args.arch == "pocket":
        from flowr_root_model.models.fm_pocket import LigandPocketCFM
        from flowr_root_model.models.pocket import LigandGenerator, PocketEncoder

        fixed_equi = hparams["pocket-fixed_equi"]
        pocket_enc = PocketEncoder(
            hparams["pocket-d_equi"],
            hparams["pocket-d_inv"],
            hparams["d_message"],
            hparams["pocket-n_layers"],
            hparams["n_attn_heads"],
            hparams["d_message_ff"],
            hparams["d_edge"],
            vocab_pocket_atoms.size,
            n_bond_types,
            vocab_pocket_res.size,
            fixed_equi=fixed_equi,
            emb_size=hparams["emb_size"],
            use_rbf=hparams["use_rbf"],
            use_distances=hparams["use_distances"],
            use_crossproducts=hparams["use_crossproducts"],
        )
        egnn_gen = LigandGenerator(
            hparams["d_equi"],
            hparams["d_inv"],
            hparams["d_message"],
            hparams["n_layers"],
            hparams["n_attn_heads"],
            hparams["d_message_ff"],
            hparams["d_edge"],
            emb_size=hparams["emb_size"],
            n_atom_types=n_atom_types,
            n_charge_types=n_charge_types,
            n_bond_types=n_bond_types,
            n_extra_atom_feats=(
                n_hybridization_types  # + n_aromatic_types
                if hparams["add_feats"]
                else None
            ),
            predict_interactions=hparams["predict_interactions"],
            flow_interactions=hparams["flow_interactions"],
            n_interaction_types=n_interaction_types,
            predict_affinity=hparams["predict_affinity"],
            predict_docking_score=hparams["predict_docking_score"],
            use_rbf=hparams["use_rbf"],
            use_sphcs=hparams["use_sphcs"],
            use_distances=hparams["use_distances"],
            use_crossproducts=hparams["use_crossproducts"],
            use_fourier_time_embed=hparams["use_fourier_time_embed"],
            use_lig_pocket_rbf=hparams["use_lig_pocket_rbf"],
            use_inpaint_mode_embed=hparams["use_inpaint_mode_embed"],
            self_cond=hparams["self_cond"],
            coord_skip_connect=hparams["coord_skip_connect"],
            coord_update_every_n=hparams.get("coord_update_every_n", None),
            pocket_enc=pocket_enc,
        )
    elif args.arch == "pocket_flex":
        from flowr_root_model.models.complex import SemlaEncoder, SemlaLayer
        from flowr_root_model.models.fm_complex import LigandPocketCFM

        n_res_types = vocab_pocket_res.size
        layer = SemlaLayer(
            d_equi=hparams["d_equi"],
            d_inv=hparams["d_inv"],
            d_message=hparams["d_message"],
            n_heads=hparams["n_attn_heads"],
            d_attn_ff=hparams["d_attn_ff"],
            d_edge=hparams["d_edge"],
        )
        egnn_gen = SemlaEncoder(
            layer=layer,
            n_layers=hparams["n_layers"],
            n_atom_names=n_atom_types,
            n_res_types=n_res_types,
            n_charge_types=n_charge_types,
            n_bond_types=n_bond_types,
            self_cond=hparams["self_cond"],
            n_rbf=hparams["num_rbf"],
            emb_size=hparams["size_emb"],
            equi_diff=True,
        )
    else:
        raise ValueError(f"Unknown architecture {args.arch}")

    # Check if the model has been LoRA finetuned
    if hparams.get("lora_finetuning", False):
        # Apply LoRA to ligand decoder
        _inject_lora(
            lora_rank=hparams["lora_rank"],
            lora_alpha=hparams["lora_alpha"],
            mod=egnn_gen.ligand_dec,
        )
        # Apply LoRA to pocket encoder if exists
        if egnn_gen.pocket_enc is not None:
            _inject_lora(
                lora_rank=hparams["lora_rank"],
                lora_alpha=hparams["lora_alpha"],
                mod=egnn_gen.pocket_enc,
            )

    _ckpt_display = ckpt_path if ckpt_path is not None else args.ckpt_path
    print(f"Loading pretrained checkpoint from {_ckpt_display}...")
    # Initialize the ligand-pocket conditional flow model
    CFM = LigandPocketCFM
    type_mask_index = None
    bond_mask_index = None
    integrator = Integrator(
        args.integration_steps,
        use_sde_simulation=args.use_sde_simulation,
        type_strategy=args.categorical_strategy,
        bond_strategy=args.categorical_strategy,
        coord_strategy="continuous",
        pocket_noise=hparams["pocket_noise"],
        cat_noise_level=args.cat_sampling_noise_level,
        coord_noise_std=args.coord_noise_scale,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        use_cosine_scheduler=args.use_cosine_scheduler,
    )
    _ckpt = ckpt_path if ckpt_path is not None else args.ckpt_path
    fm_model = CFM.load_from_checkpoint(
        _ckpt,
        gen=egnn_gen,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        dataset_info=dataset_info,
        graph_inpainting=args.graph_inpainting is not None,
        **hparams,
    )

    if getattr(args, "lora_finetuning", None):
        print("Applying LoRA finetuning...")
        _hparams = fm_model.hparams
        _hparams["lora_finetuning"] = True
        _hparams["lora_rank"] = args.lora_rank
        _hparams["lora_alpha"] = args.lora_alpha

        # Load the pretrained weights
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items()}
        egnn = copy.deepcopy(egnn_gen)
        egnn.load_state_dict(state_dict, strict=True)
        assert (
            not args.affinity_finetuning
        ), "Cannot use both LoRA and affinity_finetune."
        assert not args.freeze_layers, "Cannot use both LoRA and freeze_layers."

        # Apply LoRA to ligand decoder
        _inject_lora(
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, mod=egnn.ligand_dec
        )
        # Apply LoRA to pocket encoder if exists
        if egnn.pocket_enc is not None:
            _inject_lora(
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                mod=egnn.pocket_enc,
            )

        # Freeze all parameters except LoRA
        trainable_params = 0
        total_params = 0

        for n, p in egnn.ligand_dec.named_parameters():
            total_params += p.numel()
            if "lora" in n:
                p.requires_grad = True
                trainable_params += p.numel()
            else:
                p.requires_grad = False

        # Keep pocket encoder trainable if exists
        if egnn.pocket_enc is not None:
            for n, p in egnn.pocket_enc.named_parameters():
                total_params += p.numel()
                if "lora" in n:
                    p.requires_grad = True
                    trainable_params += p.numel()
                else:
                    p.requires_grad = False

        print(
            f"LoRA: {trainable_params}/{total_params} parameters trainable ({100*trainable_params/total_params:.2f}%)"
        )
        # Set the modified generator back to the model
        fm_model.gen = egnn
        fm_model.save_hyperparameters(_hparams)

    elif getattr(args, "freeze_layers", None):
        print("Applying freeze_layers finetuning...")
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items()}
        egnn = copy.deepcopy(egnn_gen)
        egnn.load_state_dict(state_dict, strict=True)
        assert (
            not args.affinity_finetuning
        ), "Cannot use both freeze_layers and affinity_finetune."

        def _freeze_bottom_layers(args, egnn_gen):
            """Freeze bottom layers for gentle fine-tuning"""

            n_layers_to_train = getattr(args, "n_top_layers_to_retrain", 3)
            n_layers_to_train_pocket = getattr(
                args, "n_top_layers_to_retrain_pocket", 2
            )

            trainable_params = 0
            total_params = 0

            # For LigandGenerator with pocket conditioning (args.arch == "pocket")
            if hasattr(egnn_gen, "ligand_dec") and hasattr(
                egnn_gen.ligand_dec, "layers"
            ):
                layers = egnn_gen.ligand_dec.layers
                n_layers = len(layers)

                print(
                    f"Found {n_layers} ligand decoder layers, training top {n_layers_to_train}"
                )

                for i, layer in enumerate(layers):
                    layer_name = f"ligand_dec.layers.{i}"
                    if i < n_layers - n_layers_to_train:
                        # Freeze bottom layers
                        for name, param in layer.named_parameters():
                            param.requires_grad = False
                            total_params += param.numel()
                            print(f"  Frozen: {layer_name}.{name}")
                    else:
                        # Train top layers
                        for name, param in layer.named_parameters():
                            param.requires_grad = True
                            trainable_params += param.numel()
                            total_params += param.numel()
                            print(f"  Trainable: {layer_name}.{name}")
            else:
                raise ValueError("Ligand decoder missing in initiated model!")

            # Here in this func, always keep pocket encoder trainable (if it exists)
            if hasattr(egnn_gen, "pocket_enc") and egnn_gen.pocket_enc is not None:
                if n_layers_to_train_pocket is not None:
                    egnn_gen.pocket_enc.freeze_bottom_layers(n_layers_to_train_pocket)
                    print(
                        f"  Pocket encoder: training top {n_layers_to_train_pocket} layers"
                    )
                else:
                    for name, param in egnn_gen.pocket_enc.named_parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                        total_params += param.numel()
                    print("  Pocket encoder: kept trainable")
            else:
                raise ValueError("Pocket encoder missing in initiated model!")

            # Keep output projections trainable - these are direct attributes of ligand_dec
            output_module_names = [
                "coord_out_proj",
                "atom_type_proj",
                "atom_charge_proj",
                "bond_proj",
                "bond_refine",
            ]
            if hparams["predict_affinity"]:
                output_module_names += [
                    "pic50_head",
                    "pkd_head",
                    "pki_head",
                    "pec50_head",
                ]
            if hparams["predict_docking_score"]:
                output_module_names += ["vina_head", "gnina_head"]

            for module_name in output_module_names:
                if hasattr(egnn_gen.ligand_dec, module_name):
                    module = getattr(egnn_gen.ligand_dec, module_name)
                    for name, param in module.named_parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                        total_params += param.numel()
                    print(f"  Output module ligand_dec.{module_name}: kept trainable")

            # Handle final normalization layers
            norm_modules = ["final_coord_norm", "final_inv_norm", "final_bond_norm"]
            for module_name in norm_modules:
                if hasattr(egnn_gen.ligand_dec, module_name):
                    module = getattr(egnn_gen.ligand_dec, module_name)
                    for name, param in module.named_parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                        total_params += param.numel()
                    print(f"  Norm module ligand_dec.{module_name}: kept trainable")

            print(
                f"Layer freezing: {trainable_params}/{total_params} parameters trainable ({100*trainable_params/total_params:.2f}%)"
            )
            return egnn_gen

        egnn = _freeze_bottom_layers(args, egnn)
        fm_model.gen = egnn

    elif getattr(args, "affinity_finetuning", None):
        print("Applying affinity_finetuning...")
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items()}
        egnn = copy.deepcopy(egnn_gen)
        egnn.load_state_dict(state_dict, strict=True)
        assert (
            not args.freeze_layers
        ), "Cannot use both freeze_layers and affinity_finetune."

        def _freeze_all_except_affinity_heads(args, egnn_gen):
            """
            Freeze all parameters except the selected affinity head(s).
            """
            affinity_heads = args.affinity_finetuning
            if isinstance(affinity_heads, str):
                affinity_heads = [affinity_heads]
            trainable_params = 0
            total_params = 0

            # Freeze all parameters
            for name, param in egnn_gen.named_parameters():
                param.requires_grad = False
                total_params += param.numel()

            # Unfreeze only the selected affinity head(s)
            for head in affinity_heads:
                head_name = f"{head}_head"
                if hasattr(egnn_gen.ligand_dec, head_name):
                    module = getattr(egnn_gen.ligand_dec, head_name)
                    for n, p in module.named_parameters():
                        p.requires_grad = True
                        trainable_params += p.numel()
                        print(f"  Affinity head ligand_dec.{head_name}.{n}: trainable")
                else:
                    print(
                        f"  Warning: Affinity head ligand_dec.{head_name} not found in model."
                    )

            print(
                f"Affinity finetuning: {trainable_params}/{total_params} parameters trainable ({100*trainable_params/total_params:.2f}%)"
            )
            return egnn_gen

        egnn = _freeze_all_except_affinity_heads(args, egnn)
        fm_model.gen = egnn

    print("Done.")
    if return_info:
        return (
            fm_model,
            hparams,
            vocab,
            vocab_charges,
            vocab_hybridization,
            vocab_aromatic,
            vocab_pocket_atoms,
            vocab_pocket_res,
        )
    return fm_model


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ******************************* LOAD DATA ***********************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


