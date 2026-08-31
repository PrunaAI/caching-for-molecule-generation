"""Taylor-series cache for predicting skipped backbone outputs."""

from __future__ import annotations

from math import factorial
from typing import Any, Dict

import torch
from torch import Tensor

from mol_cache.cache.cache import Cache


def _zeros_like_float(value: Any) -> Any:
    """Allocate zeros matching ``value`` (Tensor or TensorDict-safe)."""
    if value is None:
        return None
    if value.__class__.__name__ == "TensorDict" or type(value).__module__.startswith("tensordict"):
        return torch.zeros_like(value)
    return torch.zeros_like(value, dtype=torch.float32)


class TaylorCache(Cache):
    """Approximate skipped steps with a finite Taylor expansion."""

    def __init__(self, max_order: int = 2) -> None:
        self._validate_max_order(max_order)
        self.max_order = max_order
        self.cache: Dict[int, Any] = {}
        self.last_activated_step: int = 0
        self.dictionary_keys = None

    def _validate_max_order(self, max_order: int) -> None:
        if max_order not in {1, 2, 3, 4}:
            raise ValueError("`max_order` must be an integer between 1 and 4.")

    def put(self, noise_pred: Tensor, **kwargs: Any) -> None:
        step = kwargs.get("step")
        if step is None:
            raise ValueError("Taylor cache requires a `step` argument.")

        step_diff = step - self.last_activated_step
        if isinstance(noise_pred, dict):
            self.dictionary_keys = sorted(noise_pred.keys())
            noise_pred = [noise_pred[k] for k in self.dictionary_keys]
        else:
            self.dictionary_keys = None

        updated: Dict[int, Any] = {
            0: [n.detach().clone() if n is not None else None for n in noise_pred]
        }
        for i in range(self.max_order):
            if i not in self.cache:
                break
            updated[i + 1] = [
                (updated[i][k] - self.cache[i][k]) / step_diff
                if updated[i][k] is not None
                else None
                for k in range(len(updated[i]))
            ]

        self.cache = updated
        self.last_activated_step = step

    def get(self, **kwargs: Any) -> Tensor:
        step = kwargs.get("step")
        if step is None:
            raise ValueError("Taylor cache requires a `step` argument.")
        if not self.cache:
            raise ValueError("Taylor cache is empty.")

        x = step - self.last_activated_step
        first_order = next(iter(self.cache.values()))
        output = [
            _zeros_like_float(first_order[k]) if first_order[k] is not None else None
            for k in range(len(first_order))
        ]
        for i, value in self.cache.items():
            for j in range(len(value)):
                if value[j] is not None:
                    output[j] += (1 / factorial(i)) * value[j] * (x**i)

        if self.dictionary_keys is not None:
            return {k: output[i] for i, k in enumerate(self.dictionary_keys)}
        return output

    def reset(self) -> None:
        self.cache = {}
        self.last_activated_step = 0

    def set_params(self, **kwargs: Any) -> None:
        max_order = kwargs.get("max_order", self.max_order)
        self._validate_max_order(max_order)
        self.max_order = max_order
