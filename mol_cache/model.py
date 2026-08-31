"""Abstract model adapter contract and shared sampling helpers."""

from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Chem import KekulizeException

from mol_cache.assets import checkpoint_path, data_path, repo_root

_MODEL_PATHS_INSTALLED = False


@dataclass
class SampleResult:
    """Normalized sampling result returned by adapters and the orchestrator."""

    timing_seconds: float
    n_generated: int
    artifacts: dict[str, str] = field(default_factory=dict)
    model: Any = None
    output_dir: Path | None = None
    n_samples: int = 0


def ensure_model_source_paths() -> None:
    """Insert vendored model package roots onto ``sys.path`` once."""
    global _MODEL_PATHS_INSTALLED
    if _MODEL_PATHS_INSTALLED:
        return
    root = repo_root()
    paths = [
        root,
        root / "models" / "semlaflow",
        root / "models" / "tabasco" / "src",
        root / "models" / "flowr",
        root / "models" / "flowr_root",
    ]
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    _MODEL_PATHS_INSTALLED = True


def unwrap_pruna(model: Any) -> Any:
    """Unwrap a PrunaModel wrapper when present."""
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        return inner
    return model


def write_molecules_sdf(molecules: list[Any], path: Path) -> Path:
    """Write RDKit molecules to an SDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = Chem.Mol()
    dummy.SetProp("_Name", "NONE")
    written = 0
    with Chem.SDWriter(str(path)) as writer:
        for idx, mol in enumerate(molecules):
            try:
                mol_out = mol if mol is not None else Chem.Mol(dummy)
                mol_out.SetIntProp("_Idx", idx)
                writer.write(mol_out)
                written += 1
            except KekulizeException:
                print(f"KekulizeException for molecule {idx}")
                continue
    print(f"Wrote {written} molecules to {path}")
    return path


def prepare_output_dir(cfg: DictConfig) -> Path:
    """Create ``outputs/<model>_<dataset>_<timestamp>/`` and write resolved config."""
    from mol_cache.config import cfg_to_jsonable

    stamp = time.strftime("%Y%m%d_%H%M%S")
    model = str(cfg.route.model)
    dataset = str(cfg.route.dataset)
    out = Path(str(cfg.paths.output_dir)) / f"{model}_{dataset}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    payload = cfg_to_jsonable(cfg)
    payload["experiment_dir"] = str(out)
    payload["checkpoint"] = str(checkpoint_path(model, dataset))
    payload["data_dir"] = str(data_path(model, dataset))
    (out / "resolved_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def effective_cache_steps(model: Any) -> float | None:
    """Read effective cache steps from a smashed model when available."""
    helper = getattr(model, "cache_helper", None)
    if helper is None:
        helper = getattr(unwrap_pruna(model), "cache_helper", None)
    if helper is None or not hasattr(helper, "schedule"):
        return None
    schedule = getattr(helper.schedule, "schedule", None)
    if schedule is None:
        return None
    return float(np.array(schedule).sum())


def print_sample_summary(
    cfg: DictConfig,
    out_dir: Path,
    timing_seconds: float,
    n_generated: int,
    artifacts: dict[str, str],
    effective_steps: float | None,
) -> None:
    """Print the shared sampling summary for every route."""
    print("Sampling summary")
    print(f"  model: {cfg.route.model}")
    print(f"  dataset: {cfg.route.dataset}")
    print(f"  wall_clock_seconds: {round(timing_seconds, 4)}")
    print(f"  n_samples_requested: {int(cfg.sampling.n_samples)}")
    print(f"  n_generated: {n_generated}")
    print(f"  output_dir: {out_dir}")
    print(f"  artifacts: {artifacts}")
    print(f"  cache_interval: {int(cfg.caching.cache_interval)}")
    if effective_steps is not None:
        print(f"  effective_cache_steps: {effective_steps}")


def resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve the torch device from a Hydra config."""
    device_id = int(cfg.hardware.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def synchronize_cuda() -> None:
    """Block until outstanding CUDA kernels finish (no-op without CUDA)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wall_clock_seconds(start: float) -> float:
    """Elapsed wall-clock seconds since ``start``, after a CUDA synchronize."""
    synchronize_cuda()
    return time.perf_counter() - start


class MolCacheModel(ABC):
    """
    Abstract adapter implemented by each vendored model package.

    Concrete subclasses own checkpoint / dataset loading and generation.
    The central orchestrator owns smash application and summary printing.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.model: Any = None
        self.out_dir: Path | None = None

    @abstractmethod
    def load(self) -> None:
        """
        Load checkpoint (and dataset when required) into ``self.model``.

        Returns
        -------
        None
        """

    @abstractmethod
    def sample(self, out_dir: Path) -> SampleResult:
        """
        Generate molecules and write artifacts under ``out_dir``.

        Parameters
        ----------
        out_dir : Path
            Experiment output directory.

        Returns
        -------
        SampleResult
            Timing, counts, artifacts, and reporting model.
        """

    @abstractmethod
    def configure_cache(self) -> None:
        """
        Wire model-specific cache-helper hooks after smash.

        Returns
        -------
        None
        """
