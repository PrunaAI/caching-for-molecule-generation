"""Cache helpers that wrap model forward passes with periodic compute/reuse."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor

from mol_cache.cache.cache import Cache
from mol_cache.cache.schedule import CacheSchedule


class CacheHelper(ABC):
    """Wrap a backbone call so scheduled steps compute and others reuse cache."""

    def __init__(
        self, schedule: CacheSchedule, cache: Cache, backbone: Any, call_method: str
    ) -> None:
        self.schedule = schedule
        self.backbone = backbone
        self.backbone_call = getattr(backbone, call_method)
        self.call_method = call_method
        self.cache_list: list[Cache] = [copy.deepcopy(cache) for _ in range(100)]
        self.cache_index = 0
        self.previous_latent: Tensor | None = None
        self.is_first_step = False
        self.num_backbone_calls_per_step = 1
        self.step = 0
        self.enabled = False

    def set_params(self, **kwargs: Any) -> None:
        for cache in self.cache_list:
            cache.set_params(**kwargs)
        self.schedule.set_params(**kwargs)

    def disable(self) -> None:
        if self.enabled:
            setattr(self.backbone, self.call_method, self.backbone_call)
            self.enabled = False
            self._reset(42)

    def enable(self) -> None:
        if not self.enabled:
            setattr(self.backbone, self.call_method, self._wrapped_backbone_call)
            self.enabled = True

    def _wrapped_backbone_call(
        self, *args: Any, **kwargs: Any
    ) -> Tensor | Tuple[Tensor, ...]:
        num_steps = self._get_num_steps(*args, **kwargs)
        if self.step >= self.schedule.num_steps or num_steps != self.schedule.num_steps:
            self._reset(num_steps)

        latent = self._extract_latent(*args, **kwargs)
        is_first = self._is_first_backbone_call_in_step(latent)
        self._update_state(latent, is_first)

        if is_first:
            self.schedule.adapt_before_step(step=self.step, latent=latent)

        if self.schedule[self.step]:
            noise_pred = self._compute_step(latent, *args, **kwargs)
        else:
            noise_pred = self.cache_list[self.cache_index].get(
                step=self.step,
                latent=latent,
                schedule=self.schedule.copy,
            )

        if is_first:
            self.schedule.adapt_after_step(
                step=self.step, latent=latent, noise_pred=noise_pred
            )

        packed = self._pack_noise_pred(noise_pred)
        if self.is_first_step:
            self.cache_index += 1
            return packed
        if self.cache_index == self.num_backbone_calls_per_step - 1:
            self.step += 1
        self.cache_index += 1
        return packed

    def _is_first_backbone_call_in_step(self, latent: Tensor) -> bool:
        if self.previous_latent is None:
            return True
        return not torch.equal(latent, self.previous_latent)

    def _update_state(self, latent: Tensor, is_first: bool) -> None:
        if self.previous_latent is None:
            self.is_first_step = True
            self.previous_latent = latent
            return
        if is_first:
            self.cache_index = 0
            self.previous_latent = latent
            if self.is_first_step:
                self.step += 1
                self.is_first_step = False
            return
        if self.is_first_step:
            self.num_backbone_calls_per_step += 1

    def _reset(self, num_steps: int) -> None:
        for cache in self.cache_list:
            cache.reset()
        self.cache_index = 0
        self.step = 0
        self.num_backbone_calls_per_step = 1
        self.previous_latent = None
        self.is_first_step = False
        self.schedule.reset(num_steps)

    @abstractmethod
    def _get_num_steps(self, *args: Any, **kwargs: Any) -> int:
        """Return the total number of solver steps for this generation."""

    def _extract_latent(self, *args: Any, **kwargs: Any) -> Tensor:
        if len(args) > 0:
            return args[0]
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        raise ValueError("Model does not support caching.")

    def _unpack_noise_pred(self, noise_pred: Tensor | Tuple[Tensor, ...]) -> Tensor:
        if not isinstance(noise_pred, Tensor):
            raise ValueError("Noise prediction is not a tensor.")
        return noise_pred

    def _pack_noise_pred(self, noise_pred: Tensor) -> Tensor | Tuple[Tensor, ...]:
        return noise_pred

    def _compute_step(self, latent: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        noise_pred = self.backbone_call(*args, **kwargs)
        noise_pred = self._unpack_noise_pred(noise_pred)
        self.cache_list[self.cache_index].put(
            noise_pred=noise_pred,
            step=self.step,
            schedule=self.schedule.copy,
            latent=latent,
        )
        return noise_pred


class CustomHelper(CacheHelper):
    """Cache helper for molecular models configured after smash."""

    def __init__(
        self, schedule: CacheSchedule, cache: Cache, cache_order: int = 0
    ) -> None:
        self.schedule = schedule
        self.cache = cache
        self.configured = False
        self.enabled = False
        self.pipe_num_steps = 1

    def configure(
        self,
        backbone: Any,
        backbone_call_method: str,
        pipe: Any,
        pipe_call_method: str,
        step_argument: str,
        num_backbone_calls_per_step: int = 1,
    ) -> None:
        """Attach pipe/backbone hooks after smash."""
        self.pipe = pipe
        self.pipe_call_method = pipe_call_method
        self.pipe_call = getattr(self.pipe, pipe_call_method)
        self.step_argument = step_argument
        self.num_backbone_calls = num_backbone_calls_per_step
        super().__init__(self.schedule, self.cache, backbone, backbone_call_method)
        self.configured = True
        self.enable()

    def enable(self) -> None:
        if not self.configured:
            return
        if not self.enabled:
            setattr(self.pipe, self.pipe_call_method, self._wrapped_pipe_call)
        super().enable()

    def disable(self) -> None:
        if self.enabled:
            setattr(self.pipe, self.pipe_call_method, self.pipe_call)
        super().disable()

    def _wrapped_backbone_call(
        self, *args: Any, **kwargs: Any
    ) -> Tensor | Tuple[Tensor, ...]:
        num_steps = self._get_num_steps(*args, **kwargs)
        if self.step >= self.schedule.num_steps or num_steps != self.schedule.num_steps:
            self._reset(num_steps)

        latent = self._extract_latent(*args, **kwargs)

        if self.cache_index == 0:
            self.schedule.adapt_before_step(step=self.step, latent=latent)

        if self.schedule[self.step]:
            noise_pred = self.backbone_call(*args, **kwargs)
            noise_pred = self._unpack_noise_pred(noise_pred)
            self.cache_list[self.cache_index].put(
                noise_pred=noise_pred,
                step=self.step,
                schedule=self.schedule.copy,
                latent=latent,
            )
        else:
            noise_pred = self.cache_list[self.cache_index].get(
                step=self.step,
                latent=latent,
                schedule=self.schedule.copy,
            )

        if self.cache_index == 0:
            self.schedule.adapt_after_step(
                step=self.step, latent=latent, noise_pred=noise_pred
            )

        noise_pred = self._pack_noise_pred(noise_pred)
        self.cache_index = (self.cache_index + 1) % self.num_backbone_calls
        if self.cache_index == 0:
            self.step += 1
        return noise_pred

    def _wrapped_pipe_call(self, *args: Any, **kwargs: Any) -> Any:
        super()._reset(1)
        if self.step_argument not in kwargs:
            raise ValueError(
                f"The argument {self.step_argument} was not found in the kwargs of the pipe call."
            )
        self.pipe_num_steps = kwargs[self.step_argument]
        return self.pipe_call(*args, **kwargs)

    def _get_num_steps(self, *args: Any, **kwargs: Any) -> int:
        return self.pipe_num_steps

    def _unpack_noise_pred(self, noise_pred: Tensor | Tuple[Tensor, ...]) -> Tensor:
        self.return_tuple = False
        return noise_pred

    def _pack_noise_pred(self, noise_pred: Tensor) -> Tensor | Tuple[Tensor, ...]:
        if getattr(self, "return_tuple", False):
            return (noise_pred,)
        return noise_pred

    def _extract_latent(self, *args: Any, **kwargs: Any) -> Tensor:
        if len(args) > 0:
            return args[0]
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        return Tensor()
