"""Cache schedules: when to compute vs. reuse cached backbone outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from torch import Tensor


class CacheSchedule(ABC):
    """Binary schedule of compute (1) vs. cache (0) steps."""

    def __init__(self) -> None:
        self.schedule: List[int] = []

    @property
    def copy(self) -> List[int]:
        """Return a shallow copy of the schedule list."""
        return self.schedule.copy()

    @property
    def num_steps(self) -> int:
        """Return the number of steps in the schedule."""
        return len(self.schedule)

    def adapt_before_step(self, step: int, latent: Tensor | None = None, **kwargs) -> None:
        """Optional hook before a compute/cache decision."""

    def adapt_after_step(
        self,
        step: int,
        noise_pred: Tensor,
        latent: Tensor | None = None,
        **kwargs,
    ) -> None:
        """Optional hook after a step completes."""

    def reset(self, num_steps: int) -> None:
        """Rebuild the schedule for ``num_steps``."""
        self.rescale(num_steps)

    @abstractmethod
    def rescale(self, num_steps: int, **kwargs) -> None:
        """Rebuild the schedule for a new step count."""

    def set_params(self, **kwargs) -> None:
        """Update schedule parameters (optional)."""

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, idx: int) -> int:
        if idx < 0:
            raise ValueError("Index cannot be negative")
        if idx >= self.num_steps:
            raise IndexError(f"Index {idx} out of bounds for schedule of length {self.num_steps}")
        return self.schedule[idx]


class PeriodicSchedule(CacheSchedule):
    """Compute every ``cache_interval`` steps after ``start_step``."""

    def __init__(
        self, start_step: int = 0, cache_interval: int = 2, end_step: int = -1
    ) -> None:
        super().__init__()
        self._validate_params(start_step, cache_interval)
        self.start_step = start_step
        self.cache_interval = cache_interval
        self.end_step = end_step

    def _validate_params(self, start_step: int, cache_interval: int) -> None:
        if start_step < 0:
            raise ValueError("`start_step` must be non-negative.")
        if cache_interval < 1:
            raise ValueError("`cache_interval` must be at least 1.")

    def set_params(self, **kwargs) -> None:
        start_step = kwargs.get("start_step", self.start_step)
        cache_interval = kwargs.get("cache_interval", self.cache_interval)
        end_step = kwargs.get("end_step", self.end_step)
        self._validate_params(start_step, cache_interval)
        self.start_step = start_step
        self.cache_interval = cache_interval
        self.end_step = end_step

    def rescale(self, num_steps: int, **kwargs) -> None:
        start = kwargs.get("start_step", self.start_step)
        interval = kwargs.get("cache_interval", self.cache_interval)
        end = kwargs.get("end_step", self.end_step)

        self.schedule = [1] * start + [0] * max(0, num_steps - start)
        for i in range(start, num_steps, interval):
            self.schedule[i] = 1

        if end >= 0:
            for i in range(min(end + 1, num_steps), num_steps):
                self.schedule[i] = 1

        if num_steps > 0:
            self.schedule[-1] = 1
