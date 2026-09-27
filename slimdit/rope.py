"""Rotary frequency table.

The official MiniMax-H3 checkpoint does not ship ``rope.inv_freq`` (the reference implementation
derives it), but ComfyUI's ``MiniMaxH3Model`` registers it as an ``nn.Buffer`` filled with
``torch.empty`` and expects the checkpoint to carry the values. The reference checkpoint's
stored table is reproduced *bit for bit* by evaluating

    inv_freq[k] = 1 / (10000 ** (k / 16)),  k in [0, 16)

entirely in float32 (other spellings of the same formula round differently in the last ulps).
It is consumed as ``pos[..., None] * inv_freq`` per (t, h, w) axis, concatenated into a 48-wide
half, so the rope rotates 96 of the 128 head channels.
"""

from __future__ import annotations

import numpy as np

INV_FREQ_LEN = 16
THETA = 10000.0


def rope_inv_freq(length: int = INV_FREQ_LEN, theta: float = THETA) -> np.ndarray:
    """``[length]`` float32 rotary frequencies, index 0 == 1.0."""
    k = np.arange(length, dtype=np.float32)
    exponent = k / np.float32(length)
    return np.float32(1.0) / (np.float32(theta) ** exponent)
