"""
DPM-Solver++ integration for SemlaFlow flow matching (coordinates).

Uses Hugging Face ``diffusers.DPMSolverMultistepScheduler`` with ``use_flow_sigmas`` and
``prediction_type="sample"`` so the network's coordinate output is treated as data prediction
(``x_1`` / endpoint), consistent with the existing Euler integrator velocity field.

Discrete atomics and bonds are still advanced with :meth:`Integrator.step_discrete_only` using
an Euler-style step size derived from the flow ``sigma`` grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from semlaflow.models.fm import Integrator


def _require_diffusers():
    try:
        from diffusers import DPMSolverMultistepScheduler
    except ImportError as e:
        raise ImportError(
            "DPM-Solver++ requires the `diffusers` package. Install it with: pip install diffusers"
        ) from e
    return DPMSolverMultistepScheduler


def build_flow_dpm_scheduler(
    num_inference_steps: int,
    *,
    num_train_timesteps: int = 1000,
    solver_order: int = 2,
    flow_shift: float = 1.0,
) -> Any:
    """
    Build a ``DPMSolverMultistepScheduler`` configured for flow-style sigmas (DPM-Solver++).

    Parameters
    ----------
    num_inference_steps : int
        Number of sampling steps (matches ``integration_steps``).
    num_train_timesteps : int, optional
        Training-time discretization (diffusers default 1000).
    solver_order : int, optional
        Multistep order (1, 2, or 3). ``2`` matches common DPM-Solver++ usage.
    flow_shift : float, optional
        Flow sigma shift (see diffusers ``flow_shift``).

    Returns
    -------
    DPMSolverMultistepScheduler
        Scheduler with ``set_timesteps`` not yet called.

    Examples
    --------
    >>> sched = build_flow_dpm_scheduler(50)  # doctest: +SKIP
    >>> sched.set_timesteps(50)  # doctest: +SKIP
    """
    DPMSolverMultistepScheduler = _require_diffusers()
    return DPMSolverMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        solver_order=solver_order,
        prediction_type="sample",
        use_flow_sigmas=True,
        flow_shift=flow_shift,
        final_sigmas_type="zero",
    )


def discrete_step_size_from_sigmas(
    sigmas: torch.Tensor, step_index: int
) -> float:
    """
    Euler step size in ``t``-space between current and next sigma knot.

    Parameters
    ----------
    sigmas : Tensor
        1D sigma schedule from the scheduler (CPU or device).
    step_index : int
        Current ``scheduler.step_index`` before ``step``.

    Returns
    -------
    float
        Positive step size ``|t_next - t_curr|`` for discrete categorical updates.

    Examples
    --------
    >>> s = torch.tensor([1.0, 0.5, 0.0])
    >>> discrete_step_size_from_sigmas(s, 0)
    0.5...
    """
    s0 = sigmas[step_index].float()
    s1 = sigmas[step_index + 1].float()
    t0 = 1.0 - s0
    t1 = 1.0 - s1
    return float(abs(t1 - t0).item())


def dpmpp_coords_step(
    scheduler: Any,
    coords_pred: torch.Tensor,
    curr_coords: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """
    Run one DPM-Solver++ coordinate update.

    Parameters
    ----------
    scheduler : DPMSolverMultistepScheduler
        Scheduler with ``set_timesteps`` already called.
    coords_pred : Tensor
        Model coordinate output (data / endpoint prediction), same as Euler path.
    curr_coords : Tensor
        Current coordinates ``x_t``.
    timestep : Tensor
        Scalar or batch timestep index matching the pipeline contract (integer).

    Returns
    -------
    Tensor
        ``prev_sample`` coordinates for the next sigma.

    Examples
    --------
    >>> # Used from MolecularCFM._generate when solver is dpmpp.  # doctest: +SKIP
    """
    out = scheduler.step(
        model_output=coords_pred,
        timestep=timestep,
        sample=curr_coords,
    )
    return out.prev_sample
