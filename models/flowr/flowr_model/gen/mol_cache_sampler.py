"""FLOWR mol-cache adapter: load SPINDR checkpoint and sample ligands."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


class FlowrModel(MolCacheModel):
    """FLOWR adapter for mol-cache sampling on SPINDR."""

    def __init__(self, cfg: DictConfig, **_: Any) -> None:
        super().__init__(cfg)
        self.hparams: Any = None
        self.vocab: Any = None
        self.args: SimpleNamespace | None = None

    def _build_args(self, out_dir: Path) -> SimpleNamespace:
        """Build FLOWR sampling args from the Hydra model config."""
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
            ckpt_path=str(ckpt),
            data_path=str(data_dir),
            dataset=str(self.cfg.route.dataset),
            save_dir=str(out_dir),
            save_file=None,
            coord_noise_std=float(model_cfg["coord_noise_std"]),
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
            interaction_inpainting=bool(model_cfg["interaction_inpainting"]),
            scaffold_inpainting=bool(model_cfg["scaffold_inpainting"]),
            func_group_inpainting=bool(model_cfg["func_group_inpainting"]),
            linker_inpainting=bool(model_cfg["linker_inpainting"]),
            use_equi_ot=bool(model_cfg["use_equi_ot"]),
            separate_pocket_interpolation=bool(model_cfg["separate_pocket_interpolation"]),
            separate_interaction_interpolation=bool(model_cfg["separate_interaction_interpolation"]),
            integration_steps=int(model_cfg["integration_steps"]),
            cat_sampling_noise_level=int(model_cfg["cat_sampling_noise_level"]),
            ode_sampling_strategy=str(model_cfg["ode_sampling_strategy"]),
            categorical_strategy=str(model_cfg["categorical_strategy"]),
            bucket_cost_scale=str(model_cfg["bucket_cost_scale"]),
        )

    def load(self) -> None:
        """Load FLOWR model for SPINDR sampling."""
        import flowr_model.scriptutil as util
        import pytorch_lightning as L
        from flowr_model.gen.generate_from_smol import load_model

        L.seed_everything(int(self.cfg.seed))
        util.disable_lib_stdout()
        util.configure_fs()

        args = self._build_args(Path("."))
        model, hparams, vocab, _, _ = load_model(args)
        model = model.to(resolve_device(self.cfg))
        model.eval()
        self.model = model
        self.hparams = hparams
        self.vocab = vocab
        self.args = args

    def configure_cache(self) -> None:
        """Configure FLOWR cache hooks after smash."""
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
        """Sample FLOWR ligands and write SDF output."""
        from tqdm import tqdm

        from flowr_model.data.datasets import GeometricDataset
        from flowr_model.gen.generate_from_smol import get_dataloader, load_util, split_list
        from flowr_model.scriptutil import generate_ligands_per_target
        from flowr_model.util.pocket import PocketComplexBatch

        assert self.args is not None
        self.args = self._build_args(out_dir)
        self.args.save_dir = str(out_dir)
        model = self.model
        hparams = self.hparams
        vocab = self.vocab
        args = self.args
        transform, interpolant = load_util(args, hparams, vocab)

        data_file = Path(args.data_path) / f"{args.dataset_split}.smol"
        systems = PocketComplexBatch.from_bytes(data_file.read_bytes(), remove_hs=hparams["remove_hs"])
        systems = split_list(systems, 1)[0]
        if args.max_systems is not None:
            systems = systems[: args.max_systems]

        all_flat: list[Any] = []
        synchronize_cuda()
        global_start = time.perf_counter()

        for system in tqdm(systems, desc="FLOWR sampling"):
            system = PocketComplexBatch([system])
            dataset = GeometricDataset(system, data_cls=PocketComplexBatch, transform=transform)
            k = 0
            num_ligands = 0
            all_gen_ligs: list[Any] = []
            while num_ligands < args.sample_n_molecules_per_target and k <= args.max_sample_iter:
                need = args.sample_n_molecules_per_target - num_ligands
                data = dataset.sample_n_molecules_per_target(need)
                dataloader = get_dataloader(data, vocab, interpolant, args, hparams, iter=k)
                for batch in dataloader:
                    gen_ligs = generate_ligands_per_target(args, hparams, model, batch)
                    all_gen_ligs.extend(gen_ligs)
                    num_ligands += len(gen_ligs)
                k += 1
            if num_ligands == 0:
                raise RuntimeError("FLOWR sampling produced no ligands.")
            if num_ligands > args.sample_n_molecules_per_target:
                all_gen_ligs = all_gen_ligs[: args.sample_n_molecules_per_target]
            all_flat.extend(all_gen_ligs)

        timing = wall_clock_seconds(global_start)
        sdf_path = write_molecules_sdf(all_flat, out_dir / "molecules.sdf")
        return SampleResult(
            timing_seconds=timing,
            n_generated=len(all_flat),
            artifacts={"molecules.sdf": str(sdf_path)},
            model=model,
        )
