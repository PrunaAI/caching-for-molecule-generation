"""Mol-cache: molecular caching algorithms and reproduction CLI for Pruna."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

_REGISTERED = False


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def register_mol_algorithms(force: bool = False) -> None:
    """Idempotently register molecular cachers with Pruna's AlgorithmRegistry."""
    global _REGISTERED
    if _REGISTERED and not force:
        return

    from pruna.algorithms.base.registry import AlgorithmRegistry

    from mol_cache.cache.periodic import MolPeriodicCacher

    name = MolPeriodicCacher.algorithm_name
    if name not in getattr(AlgorithmRegistry, "_registry", {}):
        AlgorithmRegistry.register_algorithm(MolPeriodicCacher())
    _REGISTERED = True


def get_smash_config(cfg: Any):
    """Build a Pruna SmashConfig from ``cfg.caching``."""
    from pruna import SmashConfig

    register_mol_algorithms()
    caching = cfg.caching
    config = SmashConfig()
    if hasattr(config, "disable_saving"):
        config.disable_saving()
    config.add("mol_periodic")
    config.add(
        {
            "mol_periodic_cache_interval": int(caching.cache_interval),
            "mol_periodic_start_step": int(caching.cache_start_step),
            "mol_periodic_end_step": int(caching.cache_end_step),
            "mol_periodic_cache_mode": str(caching.cache_mode),
            "mol_periodic_cache_order": int(caching.cache_order),
            "mol_periodic_custom_model": True,
        }
    )
    return config


def smash(model: Any, smash_config: Any = None, experimental: bool = True, cfg: Any = None) -> Any:
    """Apply mol_periodic caching through open-source ``pruna.smash``."""
    from pruna import smash as pruna_smash

    register_mol_algorithms()
    if smash_config is None:
        smash_config = get_smash_config(cfg)
    return pruna_smash(model, smash_config, experimental=experimental)
