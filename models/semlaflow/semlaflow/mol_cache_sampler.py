"""SemlaFlow mol-cache adapter: load GEOM checkpoint and sample molecules."""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

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


class SemlaFlowModel(MolCacheModel):
    """SemlaFlow adapter for mol-cache sampling on GEOM."""

    def __init__(self, cfg: DictConfig, **_: Any) -> None:
        super().__init__(cfg)
        self.dm: Any = None
        self.solver: str = str(cfg.model.solver)
        self.integration_steps: int = int(cfg.model.integration_steps)
        self.strategy: str = str(cfg.model.ode_sampling_strategy)

    def load(self) -> None:
        """Load SemlaFlow checkpoint and GEOM datamodule."""
        import lightning as L
        import semlaflow.scriptutil as util
        from semlaflow.data.datamodules import GeometricInterpolantDM
        from semlaflow.data.datasets import GeometricDataset
        from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
        from semlaflow.models.fm import Integrator, MolecularCFM
        from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator

        L.seed_everything(int(self.cfg.seed))
        util.disable_lib_stdout()
        util.configure_fs()

        model_name = str(self.cfg.route.model)
        dataset_name = str(self.cfg.route.dataset)
        ckpt = checkpoint_path(model_name, dataset_name)
        data_dir = data_path(model_name, dataset_name)
        vocab = util.build_vocab()

        checkpoint = torch.load(str(ckpt), map_location="cpu")
        hparams = checkpoint["hyper_parameters"]
        hparams["compile_model"] = False
        hparams["integration-steps"] = self.integration_steps
        hparams["sampling_strategy"] = self.strategy
        hparams["ode_solver"] = self.solver
        hparams["dpm_solver_order"] = int(self.cfg.model.dpm_solver_order)
        hparams["dpm_flow_shift"] = float(self.cfg.model.dpm_flow_shift)

        n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])
        if hparams.get("architecture") is None:
            hparams["architecture"] = "semla"
        if hparams["architecture"] != "semla":
            raise ValueError(f"Unsupported SemlaFlow architecture: {hparams['architecture']}")

        dynamics = EquiInvDynamics(
            hparams["d_model"],
            hparams["d_message"],
            hparams["n_coord_sets"],
            hparams["n_layers"],
            n_attn_heads=hparams["n_attn_heads"],
            d_message_hidden=hparams["d_message_hidden"],
            d_edge=hparams["d_edge"],
            self_cond=hparams["self_cond"],
            coord_norm=hparams["coord_norm"],
        )
        egnn_gen = SemlaGenerator(
            hparams["d_model"],
            dynamics,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            self_cond=hparams["self_cond"],
            size_emb=hparams["size_emb"],
            max_atoms=hparams["max_atoms"],
        )
        type_mask_index = (
            vocab.indices_from_tokens(["<MASK>"])[0]
            if hparams["train-type-interpolation"] == "mask"
            else None
        )
        integrator = Integrator(
            self.integration_steps,
            type_strategy=hparams["integration-type-strategy"],
            bond_strategy=hparams["integration-bond-strategy"],
            type_mask_index=type_mask_index,
            bond_mask_index=None,
            cat_noise_level=int(self.cfg.model.cat_sampling_noise_level),
        )
        model = MolecularCFM.load_from_checkpoint(
            str(ckpt),
            gen=egnn_gen,
            vocab=vocab,
            integrator=integrator,
            type_mask_index=type_mask_index,
            bond_mask_index=None,
            **hparams,
        )

        transform = partial(
            util.mol_transform,
            vocab=vocab,
            n_bonds=5,
            coord_std=util.GEOM_COORDS_STD_DEV,
        )
        dataset = GeometricDataset.load(Path(data_dir) / "test.smol", transform=transform)
        dataset = dataset.sample(int(self.cfg.sampling.n_samples), replacement=True)
        val_type_mask = (
            vocab.indices_from_tokens(["<MASK>"])[0]
            if hparams["val-type-interpolation"] == "mask"
            else None
        )
        prior_sampler = GeometricNoiseSampler(
            vocab.size,
            5,
            coord_noise="gaussian",
            type_noise=hparams["val-prior-type-noise"],
            bond_noise=hparams["val-prior-bond-noise"],
            scale_ot=hparams["val-prior-noise-scale-ot"],
            zero_com=True,
            type_mask_index=val_type_mask,
            bond_mask_index=None,
        )
        eval_interpolant = GeometricInterpolant(
            prior_sampler,
            coord_interpolation="linear",
            type_interpolation=hparams["val-type-interpolation"],
            bond_interpolation=hparams["val-bond-interpolation"],
            equivariant_ot=False,
            batch_ot=False,
        )
        batch_cost = self.cfg.model.batch_cost
        if self.cfg.sampling.batch_cost is not None:
            batch_cost = self.cfg.sampling.batch_cost
        self.dm = GeometricInterpolantDM(
            None,
            None,
            dataset,
            int(batch_cost),
            test_interpolant=eval_interpolant,
            bucket_limits=util.GEOM_DRUGS_BUCKET_LIMITS,
            bucket_cost_scale=str(self.cfg.model.bucket_cost_scale),
            pad_to_bucket=False,
        )
        self.model = model

    def configure_cache(self) -> None:
        """Configure SemlaFlow cache hooks after smash."""
        num_backbone_calls = 2 if self.solver == "heun" else 1
        smashed = self.model
        model = unwrap_pruna(smashed)
        helper = getattr(model, "cache_helper", None) or getattr(smashed, "cache_helper", None)
        if helper is None:
            raise AttributeError("No cache_helper found; smash with mol_periodic first.")
        if hasattr(model, "set_block_cache"):
            model.set_block_cache(False, 0)
        helper.configure(
            pipe=model,
            backbone=model,
            pipe_call_method="_generate",
            step_argument="steps",
            backbone_call_method="forward",
            num_backbone_calls_per_step=num_backbone_calls,
        )

    def sample(self, out_dir: Path) -> SampleResult:
        """Generate SemlaFlow molecules and write SDF output."""
        import semlaflow.scriptutil as util

        synchronize_cuda()
        start = time.perf_counter()
        molecules, _ = util.generate_molecules(
            self.model,
            self.dm,
            self.integration_steps,
            self.strategy,
            solver=self.solver,
            stabilities=False,
            gpu_id=int(self.cfg.hardware.device),
        )
        timing = wall_clock_seconds(start)
        sdf_path = write_molecules_sdf(molecules, out_dir / "molecules.sdf")
        return SampleResult(
            timing_seconds=timing,
            n_generated=len(molecules),
            artifacts={"molecules.sdf": str(sdf_path)},
            model=self.model,
        )
