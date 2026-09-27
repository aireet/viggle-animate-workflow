"""MiniMax-H3 sampling schedules.

The upstream ComfyUI workflow ships one manual schedule: ``1.0, 0.8571428571428571, 0.6, 0.0``
(four sigma points -> three Euler updates). It is not arbitrary: it is a uniform grid pushed
through Comfy's sigma shift with ``shift=3``::

    sigma' = shift * sigma / (1 + (shift - 1) * sigma)
    grid [1, 2/3, 1/3, 0] -> [1.0, 0.857142857142857, 0.6, 0.0]

which reproduces both middle values digit for digit. So a different step count is the same shift
applied to a finer or coarser grid -- no scheduler node required.

3 steps is what the finetune and the DMD LoRA are distilled for; 6 steps doubles the sampling
time and is a comparison knob, not a quality upgrade.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_SHIFT = 3.0
DEFAULT_STEPS = 3
SUPPORTED_STEPS = (3, 4, 6)

#: The list the vendor workflow hard-codes; kept for the equality test.
UPSTREAM_SIGMAS = (1.0, 0.8571428571428571, 0.6, 0.0)


def h3_sigmas(steps: int = DEFAULT_STEPS, shift: float = DEFAULT_SHIFT) -> list[float]:
    """``steps + 1`` sigma values (last one zero) for a MiniMax-H3 render."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if shift <= 0:
        raise ValueError(f"shift must be > 0, got {shift}")
    return [
        shift * (i / steps) / (1.0 + (shift - 1.0) * (i / steps))
        for i in range(steps, -1, -1)
    ]


def format_sigmas(sigmas: Sequence[float]) -> str:
    return ", ".join(f"{value:.6g}" for value in sigmas)
