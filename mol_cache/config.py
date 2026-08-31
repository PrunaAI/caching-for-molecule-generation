"""Hydra composition helpers for centralized mol-cache configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# Map CLI destination names to Hydra override paths.
_CLI_OVERRIDE_MAP = {
    "n_samples": "sampling.n_samples",
    "seed": "seed",
    "device": "hardware.device",
    "output_dir": "paths.output_dir",
    "cache_interval": "caching.cache_interval",
    "cache_mode": "caching.cache_mode",
    "cache_order": "caching.cache_order",
    "start_step": "caching.cache_start_step",
    "end_step": "caching.cache_end_step",
    "max_systems": "sampling.max_systems",
    "num_steps": "sampling.num_steps",
    "integration_steps": "sampling.integration_steps",
}


def conf_dir() -> Path:
    """Return the package-shipped Hydra configuration directory."""
    return Path(__file__).resolve().parent / "conf"


def compose_config(
    model: str,
    dataset: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Compose a fully resolved Hydra config for one model–dataset route.

    Parameters
    ----------
    model : str
        Model family name (``semlaflow``, ``tabasco``, ``flowr``, ``flowr_root``).
    dataset : str
        Dataset name (``geom``, ``spindr``, ``crossdocked``).
    overrides : list of str or None
        Extra Hydra overrides such as ``caching.cache_interval=2``.

    Returns
    -------
    omegaconf.DictConfig
        Resolved configuration.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from mol_cache.assets import validate_route

    validate_route(model, dataset)
    GlobalHydra.instance().clear()
    override_list = [f"model={model}", f"route.dataset={dataset}", *(overrides or [])]
    with initialize_config_dir(config_dir=str(conf_dir()), version_base="1.3"):
        cfg = compose(config_name="config", overrides=override_list)
    OmegaConf.set_struct(cfg, False)
    cfg.route.model = model
    cfg.route.dataset = dataset
    OmegaConf.resolve(cfg)
    return cfg


def overrides_from_cli(args: Any) -> list[str]:
    """Convert non-``None`` argparse values into Hydra overrides."""
    overrides: list[str] = []
    for dest, path in _CLI_OVERRIDE_MAP.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        if isinstance(value, Path):
            value = str(value)
        overrides.append(f"{path}={value}")
    return overrides


def cfg_to_jsonable(cfg: DictConfig) -> dict[str, Any]:
    """Convert a Hydra config to a plain JSON-serializable dictionary."""
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[return-value]
