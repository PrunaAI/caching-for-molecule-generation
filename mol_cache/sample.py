"""Thin in-process sampling orchestrator for Hydra-configured model adapters."""

from __future__ import annotations

from omegaconf import DictConfig

from mol_cache import seed_everything
from mol_cache.assets import SUPPORTED_ROUTES, validate_route
from mol_cache.model import (
    MolCacheModel,
    SampleResult,
    effective_cache_steps,
    ensure_model_source_paths,
    prepare_output_dir,
    print_sample_summary,
)


def list_routes() -> list[tuple[str, str]]:
    """Return the supported model–dataset matrix."""
    return list(SUPPORTED_ROUTES)


def instantiate_adapter(cfg: DictConfig) -> MolCacheModel:
    """Instantiate the model adapter declared by ``cfg.model._target_``."""
    from hydra.utils import get_class

    ensure_model_source_paths()
    cls = get_class(str(cfg.model._target_))
    return cls(cfg)


def sample(cfg: DictConfig) -> SampleResult:
    """
    Run the standardized sampling lifecycle for one route.

    Order: validate → load → smash (optional) → configure_cache → sample → summary.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Fully composed Hydra configuration.

    Returns
    -------
    SampleResult
        Normalized result metadata.
    """
    validate_route(str(cfg.route.model), str(cfg.route.dataset))
    ensure_model_source_paths()
    seed_everything(int(cfg.seed))
    out_dir = prepare_output_dir(cfg)

    adapter = instantiate_adapter(cfg)
    adapter.load()

    if int(cfg.caching.cache_interval) > 1:
        # Imported lazily so adapter/config work while mol_cache.cache is under refactor.
        from mol_cache import get_smash_config, smash

        smash_config = get_smash_config(cfg)
        adapter.model = smash(adapter.model, smash_config, experimental=True)
        adapter.configure_cache()

    result = adapter.sample(out_dir)
    effective = (
        effective_cache_steps(result.model)
        if int(cfg.caching.cache_interval) > 1
        else None
    )
    print_sample_summary(
        cfg,
        out_dir,
        result.timing_seconds,
        result.n_generated,
        result.artifacts,
        effective,
    )
    result.output_dir = out_dir
    result.n_samples = int(cfg.sampling.n_samples)
    return result
