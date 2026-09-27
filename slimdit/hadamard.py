"""ConvRot Hadamard rotations, bit-compatible with comfy-kitchen.

ComfyUI's runtime un-rotates INT8 weights with its own matrix (``comfy_kitchen.tensor.int8_utils``),
so anything we write has to use exactly that construction or the loaded model computes garbage.
The reference builds a *regular* Hadamard from the 4x4 seed below (Kronecker powers) and
normalizes by ``sqrt(size)``; size must therefore be a power of 4 (256 = 4**4).

The matrix is symmetric (the seed is, and Kronecker products of symmetric matrices are), which
is why the same helper rotates and un-rotates.
"""

from __future__ import annotations

import math

import numpy as np

H4 = np.array(
    [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
    dtype=np.float64,
)

_CACHE: dict[tuple[int, str], np.ndarray] = {}


def build_hadamard(size: int = 256, dtype=np.float32) -> np.ndarray:
    """Normalized regular orthogonal Hadamard matrix of ``size`` (a power of 4)."""
    key = (size, np.dtype(dtype).str)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"regular Hadamard size must be a power of 4, got {size}")

    h = H4
    current = 4
    while current < size:
        h = np.kron(h, H4)
        current *= 4
    normalized = (h / math.sqrt(size)).astype(dtype)
    _CACHE[key] = normalized
    return normalized


def rotate(weight: np.ndarray, h: np.ndarray, group_size: int = 256) -> np.ndarray:
    """Group-wise rotation ``W @ H^T`` over blocks of ``group_size`` input features.

    Mirrors ``comfy_kitchen.tensor.int8_utils._rotate_weight``. Applying it twice is the
    identity (``H`` is orthogonal and symmetric), so the same call de-rotates.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {weight.shape}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    rotated = np.matmul(grouped, h.T.astype(weight.dtype, copy=False))
    return rotated.reshape(out_features, in_features)
