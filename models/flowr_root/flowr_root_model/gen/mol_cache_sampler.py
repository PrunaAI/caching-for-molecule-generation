"""FLOWR.root mol-cache adapter: load SPINDR/CrossDocked checkpoints and sample."""

from __future__ import annotations

import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from mol_cache.assets import checkpoint_path, data_path
from mol_cache.model import (
    MolCacheModel,
    SampleResult,
    resolve_device,
    synchronize_cuda,
    unwrap_pruna,
    wall_clock_seconds,
    write_molecules_sdf,
)


def _flowr_root_load_util(args, hparams, vocab, vocab_charges, vocab_hybridization=None, vocab_aromatic=None):
    """Build FLOWR.root transform and interpolant (cosine-scheduler safe)."""
    import flowr_root_model.scriptutil as util
    from flowr_root_model.data.interpolate import ComplexInterpolant, GeometricNoiseSampler
    from flowr_root_model.util.pocket import PROLIF_INTERACTIONS
    from flowr_root_model.util.rdkit import ConformerGenerator

    coord_std = 1.0 if hparams["coord_scale"] == 1.0 else hparams["coord_scale"]
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    transform = partial(
        util.complex_transform,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        n_bonds=n_bond_types,
        coord_std=coord_std,
        pocket_noise=args.pocket_noise,
        pocket_noise_std=args.pocket_coord_noise_std,
    )
    conformer_generator = (
        ConformerGenerator(
            cache_dir=Path(args.data_path) / "conformers",
            max_conformers=10,
            max_iters=200,
            enable_caching=True,
            vocab=vocab,
        )
        if args.graph_inpainting is not None and args.graph_inpainting == "conformer"
        else None
    )
    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        vocab_charges.size,
        n_hybridization_types=vocab_hybridization.size if vocab_hybridization is not None else None,
        n_aromatic_types=vocab_aromatic.size if vocab_aromatic is not None else None,
        coord_noise="gaussian",
        type_noise=hparams["val-ligand-prior-type-noise"],
        bond_noise=hparams["val-ligand-prior-bond-noise"],
        zero_com=True,
        type_mask_index=None,
        bond_mask_index=None,
        conformer_generator=conformer_generator,
    )
    if args.categorical_strategy == "mask":
        categorical_interpolation = "unmask"
    elif args.categorical_strategy in {"uniform-sample", "prior-sample"}:
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "velocity-sample":
        categorical_interpolation = "sample"
    else:
        raise ValueError(f"Interpolation '{args.categorical_strategy}' is not supported.")

    coord_interpolation = "cosine" if args.use_cosine_scheduler else "linear"
    eval_interpolant = ComplexInterpolant(
        prior_sampler,
        ligand_coord_interpolation=coord_interpolation,
        ligand_type_interpolation=categorical_interpolation,
        ligand_bond_interpolation=categorical_interpolation,
        pocket_noise=args.pocket_noise,
        separate_pocket_interpolation=args.separate_pocket_interpolation,
        separate_interaction_interpolation=args.separate_interaction_interpolation,
        n_interaction_types=(
            len(PROLIF_INTERACTIONS)
            if hparams["flow_interactions"]
            or hparams["predict_interactions"]
            or hparams["interaction_conditional"]
            else None
        ),
        flow_interactions=hparams["flow_interactions"],
        interaction_conditional=args.interaction_conditional,
        scaffold_hopping=args.scaffold_hopping,
        scaffold_elaboration=args.scaffold_elaboration,
        linker_inpainting=args.linker_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        fragment_growing=getattr(args, "fragment_growing", False),
        max_fragment_cuts=args.max_fragment_cuts,
        substructure_inpainting=args.substructure_inpainting,
        substructure=args.substructure,
        graph_inpainting=args.graph_inpainting,
        batch_ot=False,
        dataset=args.dataset,
        sample_mol_sizes=False,
        anisotropic_prior=getattr(args, "anisotropic_prior", False),
        inference=True,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
    )
    return transform, eval_interpolant


class FlowrRootModel(MolCacheModel):
    """FLOWR.root adapter for mol-cache sampling on SPINDR / CrossDocked."""

    def __init__(self, cfg: DictConfig, **_: Any) -> None:
        super().__init__(cfg)
        self.hparams: Any = None
        self.vocab: Any = None
        self.vocab_charges: Any = None
        self.vocab_hybridization: Any = None
        self.vocab_aromatic: Any = None
        self.args: SimpleNamespace | None = None

    def _build_args(self, out_dir: Path) -> SimpleNamespace:
        """Build FLOWR.root sampling args from the Hydra model config."""
        model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)
        assert isinstance(model_cfg, dict)
        data_dir = data_path(str(self.cfg.route.model), str(self.cfg.route.dataset))
        ckpt = checkpoint_path(str(self.cfg.route.model), str(self.cfg.route.dataset))
        batch_cost = model_cfg.get("batch_cost", 100)
        if self.cfg.sampling.batch_cost is not None:
            batch_cost = int(self.cfg.sampling.batch_cost)
        return SimpleNamespace(
            seed=int(self.cfg.seed),
            mp_index=0,
            gpus=1,
            num_workers=int(self.cfg.hardware.num_workers),
            arch=str(model_cfg["arch"]),
            pocket_noise=str(model_cfg["pocket_noise"]),
            pocket_coord_noise_std=float(model_cfg["pocket_coord_noise_std"]),
            ckpt_path=str(ckpt),
            data_path=str(data_dir),
            dataset=str(self.cfg.route.dataset),
            save_dir=str(out_dir),
            save_file=None,
            coord_noise_scale=float(model_cfg["coord_noise_scale"]),
            max_sample_iter=int(model_cfg["max_sample_iter"]),
            sample_n_molecules_per_target=int(self.cfg.sampling.n_samples),
            sample_mol_sizes=bool(model_cfg["sample_mol_sizes"]),
            corrector_iters=int(model_cfg["corrector_iters"]),
            filter_valid_unique=bool(model_cfg["filter_valid_unique"]),
            batch_cost=int(batch_cost),
            dataset_split=str(model_cfg["dataset_split"]),
            max_systems=int(model_cfg["max_systems"]),
            ligand_time=model_cfg.get("ligand_time"),
            pocket_time=model_cfg.get("pocket_time"),
            interaction_time=model_cfg.get("interaction_time"),
            resampling_steps=model_cfg.get("resampling_steps"),
            interaction_conditional=bool(model_cfg["interaction_conditional"]),
            scaffold_hopping=bool(model_cfg["scaffold_hopping"]),
            scaffold_elaboration=bool(model_cfg["scaffold_elaboration"]),
            linker_inpainting=bool(model_cfg["linker_inpainting"]),
            fragment_inpainting=bool(model_cfg["fragment_inpainting"]),
            fragment_growing=bool(model_cfg["fragment_growing"]),
            substructure_inpainting=bool(model_cfg["substructure_inpainting"]),
            substructure=model_cfg.get("substructure"),
            graph_inpainting=model_cfg.get("graph_inpainting"),
            max_fragment_cuts=int(model_cfg["max_fragment_cuts"]),
            separate_pocket_interpolation=bool(model_cfg["separate_pocket_interpolation"]),
            separate_interaction_interpolation=bool(model_cfg["separate_interaction_interpolation"]),
            integration_steps=int(model_cfg["integration_steps"]),
            cat_sampling_noise_level=int(model_cfg["cat_sampling_noise_level"]),
            ode_sampling_strategy=str(model_cfg["ode_sampling_strategy"]),
            solver=str(model_cfg["solver"]),
            categorical_strategy=str(model_cfg["categorical_strategy"]),
            bucket_cost_scale=str(model_cfg["bucket_cost_scale"]),
            use_sde_simulation=bool(model_cfg["use_sde_simulation"]),
            use_cosine_scheduler=bool(model_cfg["use_cosine_scheduler"]),
            lora_finetuned=bool(model_cfg["lora_finetuned"]),
            anisotropic_prior=bool(model_cfg["anisotropic_prior"]),
            core_growing=bool(model_cfg["core_growing"]),
            pocket_type=str(model_cfg["pocket_type"]),
        )

    def load(self) -> None:
        """Load FLOWR.root model for SPINDR or CrossDocked sampling."""
        import flowr_root_model.scriptutil as util
        import pytorch_lightning as L
        from flowr_root_model.scriptutil import load_model

        L.seed_everything(int(self.cfg.seed))
        util.disable_lib_stdout()
        util.configure_fs()

        args = self._build_args(Path("."))
        (
            model,
            hparams,
            vocab,
            vocab_charges,
            vocab_hybridization,
            vocab_aromatic,
            _,
            _,
        ) = load_model(args)
        model = model.to(resolve_device(self.cfg))
        model.eval()
        self.model = model
        self.hparams = hparams
        self.vocab = vocab
        self.vocab_charges = vocab_charges
        self.vocab_hybridization = vocab_hybridization
        self.vocab_aromatic = vocab_aromatic
        self.args = args

    def configure_cache(self) -> None:
        """Configure FLOWR.root cache hooks after smash."""
        smashed = self.model
        model = unwrap_pruna(smashed)
        helper = getattr(model, "cache_helper", None) or getattr(smashed, "cache_helper", None)
        if helper is None:
            raise AttributeError("No cache_helper found; smash with mol_periodic first.")
        helper.configure(
            pipe=model,
            backbone=model.gen,
            pipe_call_method="_generate",
            step_argument="steps",
            backbone_call_method="forward",
        )

    def sample(self, out_dir: Path) -> SampleResult:
        """Sample FLOWR.root ligands and write predictions + SDF."""
        from tqdm import tqdm

        from flowr_root_model.data.dataset import GeometricDataset
        from flowr_root_model.gen.generate import generate_ligands_per_target
        from flowr_root_model.gen.generate_from_smol import get_dataloader, split_list
        from flowr_root_model.util.pocket import PocketComplexBatch

        self.args = self._build_args(out_dir)
        model = self.model
        hparams = self.hparams
        args = self.args
        transform, interpolant = _flowr_root_load_util(
            args,
            hparams,
            self.vocab,
            self.vocab_charges,
            self.vocab_hybridization,
            self.vocab_aromatic,
        )

        data_file = Path(args.data_path) / f"{args.dataset_split}.smol"
        systems = PocketComplexBatch.from_bytes(data_file.read_bytes(), remove_hs=hparams["remove_hs"])
        systems = split_list(systems, 1)[0]
        if args.max_systems is not None:
            systems = systems[: args.max_systems]

        out_dict: dict[str, list] = defaultdict(list)
        all_flat: list[Any] = []
        synchronize_cuda()
        global_start = time.perf_counter()

        for system in tqdm(systems, desc="FLOWR.root sampling"):
            system = PocketComplexBatch([system])
            dataset = GeometricDataset(system, data_cls=PocketComplexBatch, transform=transform)
            k = 0
            num_ligands = 0
            all_gen_ligs: list[Any] = []
            data_batch = None
            while num_ligands < args.sample_n_molecules_per_target and k <= args.max_sample_iter:
                need = args.sample_n_molecules_per_target - num_ligands
                data = dataset.sample_n_molecules_per_target(need)
                dataloader = get_dataloader(args, data, interpolant, iter=k)
                for batch in dataloader:
                    prior, data_batch, _, _ = batch
                    gen_ligs = generate_ligands_per_target(
                        args,
                        model,
                        prior=prior,
                        posterior=data_batch,
                        pocket_noise=args.pocket_noise,
                    )
                    all_gen_ligs.extend(gen_ligs)
                    num_ligands += len(gen_ligs)
                k += 1
            if num_ligands == 0:
                raise RuntimeError("FLOWR.root sampling produced no ligands.")
            if num_ligands > args.sample_n_molecules_per_target:
                all_gen_ligs = all_gen_ligs[: args.sample_n_molecules_per_target]

            assert data_batch is not None
            ref_ligs = model._generate_ligs(
                data_batch, lig_mask=data_batch["lig_mask"].bool(), scale=model.coord_scale
            )[0]
            out_dict["gen_ligs"].append(all_gen_ligs)
            out_dict["ref_ligs"].append(ref_ligs)
            out_dict["ref_ligs_with_hs"].append(model.retrieve_ligs_with_hs(data_batch, save_idx=0))
            out_dict["ref_pdbs"].append(
                model.retrieve_pdbs(data_batch, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0)
            )
            out_dict["ref_pdbs_with_hs"].append(
                model.retrieve_pdbs_with_hs(data_batch, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0)
            )
            all_flat.extend(all_gen_ligs)

        timing = wall_clock_seconds(global_start)
        pred_path = out_dir / "predictions.pt"
        torch.save(dict(out_dict), str(pred_path))
        sdf_path = write_molecules_sdf(all_flat, out_dir / "molecules.sdf")
        return SampleResult(
            timing_seconds=timing,
            n_generated=len(all_flat),
            artifacts={"predictions.pt": str(pred_path), "molecules.sdf": str(sdf_path)},
            model=model,
        )
