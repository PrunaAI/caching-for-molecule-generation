"""Factories for Taylor / Adams–Bashforth caches and the custom-model helper."""

from __future__ import annotations

from typing import Any

from mol_cache.cache.ab_cache import ABCache
from mol_cache.cache.cache import Cache
from mol_cache.cache.helper import CustomHelper
from mol_cache.cache.schedule import CacheSchedule
from mol_cache.cache.taylor_cache import TaylorCache

CACHE_MODES = ("taylor", "ab")


def create_cache(cache_mode: str, cache_order: int = 2) -> Cache:
    """
    Build a cache for the requested mode.

    Parameters
    ----------
    cache_mode : str
        ``taylor`` or ``ab``.
    cache_order : int
        Taylor max order or Adams–Bashforth order.

    Returns
    -------
    Cache
        Concrete cache instance.
    """
    if cache_mode == "taylor":
        return TaylorCache(max_order=max(1, cache_order))
    if cache_mode == "ab":
        return ABCache(order=max(1, cache_order))
    raise ValueError(f"Invalid cache mode: {cache_mode!r}. Available: {CACHE_MODES}")


def create_cache_helper(
    model: Any,
    schedule: CacheSchedule,
    cache: Cache,
    is_custom_model: bool = True,
    cache_order: int = 0,
) -> CustomHelper:
    """Create the custom-model helper used by molecular generators."""
    return CustomHelper(schedule, cache, cache_order)
