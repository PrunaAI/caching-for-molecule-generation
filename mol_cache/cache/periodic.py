"""Pruna algorithm: periodic Taylor / Adams–Bashforth caching for molecular models."""

from __future__ import annotations

from typing import Any

from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter
from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import Boolean
from pruna.engine.save import SAVE_FUNCTIONS

from mol_cache.cache.factory import CACHE_MODES, create_cache, create_cache_helper
from mol_cache.cache.schedule import PeriodicSchedule


class MolPeriodicCacher(PrunaAlgorithmBase):
    """
    Periodic backbone caching for molecular geometry generators.

    After ``start_step``, the backbone is evaluated every ``cache_interval`` steps.
    Skipped steps are filled by either a Taylor expansion (``taylor``) or an
    Adams–Bashforth combination (``ab``).
    """

    algorithm_name: str = "mol_periodic"
    group_tags: list[str] = [AlgorithmTag.CACHER]
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {}
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = False
    compatible_algorithms: dict = {}

    def get_hyperparameters(self) -> list:
        """Return SmashConfig hyperparameters for mol_periodic."""
        return [
            OrdinalHyperparameter(
                "cache_interval",
                sequence=range(1, 100),
                default_value=2,
                meta=dict(desc="Compute every N steps; higher is faster."),
            ),
            OrdinalHyperparameter(
                "cache_order",
                sequence=range(1, 7),
                default_value=2,
                meta=dict(desc="Taylor max order or Adams–Bashforth order."),
            ),
            OrdinalHyperparameter(
                "start_step",
                sequence=range(300),
                default_value=0,
                meta=dict(desc="Number of initial full-compute steps."),
            ),
            OrdinalHyperparameter(
                "end_step",
                sequence=range(-1, 300),
                default_value=-1,
                meta=dict(desc="After this step, always compute (-1 = until end)."),
            ),
            CategoricalHyperparameter(
                "cache_mode",
                choices=list(CACHE_MODES),
                default_value="taylor",
                meta={
                    "desc": "taylor: Taylor expansion; ab: Adams–Bashforth multistep."
                },
            ),
            Boolean(
                "custom_model",
                default=True,
                meta={"desc": "Always true for molecular models; configure via adapter."},
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """Accept any model; molecular generators are wired via CustomHelper."""
        return True

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """Attach a periodic schedule and Taylor/AB cache helper to ``model``."""
        schedule = PeriodicSchedule(
            cache_interval=smash_config["cache_interval"],
            start_step=smash_config["start_step"],
            end_step=smash_config["end_step"],
        )
        cache = create_cache(smash_config["cache_mode"], smash_config["cache_order"])
        model.cache_helper = create_cache_helper(
            model,
            schedule,
            cache,
            smash_config["custom_model"],
            smash_config["cache_order"],
        )
        # Molecular adapters call configure_cache() after smash; enable happens there.
        return model

    def import_algorithm_packages(self) -> dict[str, Any]:
        """No extra packages required."""
        return {}
