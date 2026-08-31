"""Abstract cache interface for storing and retrieving backbone outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class Cache(ABC):
    """Store and retrieve noise / feature predictions across solver steps."""

    @abstractmethod
    def put(self, noise_pred: Tensor, **kwargs: Any) -> None:
        """Store a backbone prediction."""

    @abstractmethod
    def get(self, **kwargs: Any) -> Tensor:
        """Return a cached or extrapolated prediction."""

    @abstractmethod
    def reset(self) -> None:
        """Clear cached state."""

    def set_params(self, **kwargs: Any) -> None:
        """Update cache-specific parameters (optional)."""
