"""Tabasco mol-cache adapter: load Lightning checkpoint and sample molecules."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from mol_cache.assets import checkpoint_path
from mol_cache.model import (
    MolCacheModel,
    SampleResult,
    resolve_device,
    synchronize_cuda,
    unwrap_pruna,
    wall_clock_seconds,
    write_molecules_sdf,
)


class TabascoModel(MolCacheModel):
    """Tabasco adapter for mol-cache sampling on GEOM."""

    def __init__(self, cfg: DictConfig, **_: Any) -> None:
        self.lightning_module: Any = None
        super().__init__(cfg)

    @property
    def model(self) -> Any:
        """Return the smash target (inner FlowMatchingModel)."""
        if self.lightning_module is None:
            raise RuntimeError("TabascoModel.load() must be called before accessing .model")
        return self.lightning_module.model

    @model.setter
    def model(self, value: Any) -> None:
        """Install a (possibly smashed) inner model into the Lightning wrapper."""
        if self.lightning_module is None:
            return
        object.__setattr__(self.lightning_module, "model", value)

    def load(self) -> None:
        """Load Tabasco Lightning checkpoint (no dataset required)."""
        from tabasco.models.lightning_tabasco import LightningTabasco

        device = resolve_device(self.cfg)
        ckpt = checkpoint_path(str(self.cfg.route.model), str(self.cfg.route.dataset))
        lightning_module = LightningTabasco.load_from_checkpoint(str(ckpt))
        lightning_module.model.net.eval()
        lightning_module.to(device)
        self.lightning_module = lightning_module

    def configure_cache(self) -> None:
        """Configure Tabasco cache hooks targeting ``model.net``."""
        smashed = self.lightning_module.model
        model = unwrap_pruna(smashed)
        helper = getattr(model, "cache_helper", None) or getattr(smashed, "cache_helper", None)
        if helper is None:
            raise AttributeError("No cache_helper found; smash with mol_periodic first.")
        backbone = getattr(model, "net", model)
        helper.configure(
            pipe=model,
            backbone=backbone,
            pipe_call_method="sample",
            step_argument="num_steps",
            backbone_call_method="forward",
        )

    def sample(self, out_dir: Path) -> SampleResult:
        """Sample Tabasco molecules and write SDF output."""
        n_samples = int(self.cfg.sampling.n_samples)
        num_steps = int(self.cfg.model.num_steps)
        max_batch = int(self.cfg.model.max_batch_size)
        max_batch = min(n_samples, max_batch)
        n_batches = math.ceil(n_samples / max_batch)
        all_molecules: list[Any] = []
        total_time = 0.0
        cache_enabled = int(self.cfg.caching.cache_interval) > 1

        for _ in range(n_batches):
            batch_size = min(n_samples - len(all_molecules), max_batch)
            if cache_enabled and hasattr(self.lightning_module.model, "cache_helper"):
                self.lightning_module.model.cache_helper._reset(num_steps)
            synchronize_cuda()
            start = time.perf_counter()
            with torch.no_grad():
                out_batch = self.lightning_module.sample(batch_size=batch_size, num_steps=num_steps)
            total_time += wall_clock_seconds(start)
            all_molecules.extend(self.lightning_module.mol_converter.from_batch(out_batch))

        sdf_path = write_molecules_sdf(all_molecules, out_dir / "molecules.sdf")
        return SampleResult(
            timing_seconds=total_time,
            n_generated=len(all_molecules),
            artifacts={"molecules.sdf": str(sdf_path)},
            model=self.lightning_module.model,
        )
