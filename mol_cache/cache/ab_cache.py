"""Adams–Bashforth multistep cache for predicting skipped backbone outputs."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Union

import torch
from torch import Tensor

from mol_cache.cache.cache import Cache

# Standard Adams–Bashforth coefficients (most recent first).
_AB_COEFFS = {
    1: [1.0],
    2: [3 / 2, -1 / 2],
    3: [23 / 12, -16 / 12, 5 / 12],
    4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
    5: [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720],
    6: [
        4277 / 1440,
        -7923 / 1440,
        9982 / 1440,
        -7298 / 1440,
        2877 / 1440,
        -475 / 1440,
    ],
}


class ABCache(Cache):
    """Extrapolate skipped steps with an Adams–Bashforth linear combination."""

    def __init__(self, order: int = 2) -> None:
        self._validate_order(order)
        self.order = order
        self.cache: Deque[Union[Tensor, List[Tensor]]] = deque(maxlen=order)
        self.dictionary_keys = None
        self.coefficients = _AB_COEFFS[order]

    def _validate_order(self, order: int) -> None:
        if not isinstance(order, int) or order < 1 or order > 6:
            raise ValueError("`order` must be an integer between 1 and 6.")

    def put(self, noise_pred: Union[Tensor, List[Tensor]], **kwargs: Any) -> None:
        if isinstance(noise_pred, dict):
            self.dictionary_keys = sorted(noise_pred.keys())
            noise_pred = [noise_pred[k] for k in self.dictionary_keys]
            self.cache.append(
                [n.detach().clone() if n is not None else None for n in noise_pred]
            )
        elif isinstance(noise_pred, (list, tuple)):
            self.dictionary_keys = None
            self.cache.append(
                [n.detach().clone() if n is not None else None for n in noise_pred]
            )
        else:
            self.cache.append(noise_pred.detach().clone())

    def get(self, **kwargs: Any) -> Union[Tensor, List[Tensor]]:
        if len(self.cache) == 0:
            raise ValueError("AB cache is empty")

        if len(self.cache) == 1:
            item = self.cache[0]
            if isinstance(item, list):
                result = [t.clone() if t is not None else None for t in item]
                if self.dictionary_keys is not None:
                    return {k: result[i] for i, k in enumerate(self.dictionary_keys)}
                return result
            return item.clone()

        available = min(len(self.cache), self.order)
        coefficients = _AB_COEFFS[available]
        reference = self.cache[-1]

        if isinstance(reference, list):
            results = []
            for item in reference:
                if item is None:
                    results.append(None)
                elif item.__class__.__name__ == "TensorDict" or type(item).__module__.startswith(
                    "tensordict"
                ):
                    results.append(torch.zeros_like(item))
                else:
                    results.append(torch.zeros_like(item, dtype=torch.float32))

            for i in range(available):
                coeff = coefficients[i]
                cached = self.cache[-(i + 1)]
                for j in range(len(results)):
                    if results[j] is not None and cached[j] is not None:
                        results[j] += coeff * cached[j]

            if self.dictionary_keys is not None:
                return {k: results[i] for i, k in enumerate(self.dictionary_keys)}
            return results

        result = torch.zeros_like(reference)
        for i in range(available):
            result += coefficients[i] * self.cache[-(i + 1)]
        return result

    def reset(self) -> None:
        self.cache = deque(maxlen=self.order)
        self.coefficients = _AB_COEFFS[self.order]

    def set_params(self, **kwargs: Any) -> None:
        order = kwargs.get("order")
        if order is None:
            return
        self._validate_order(order)
        self.order = order
        self.cache = deque(list(self.cache), maxlen=order)
        self.coefficients = _AB_COEFFS[order]
