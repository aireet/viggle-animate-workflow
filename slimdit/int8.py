"""INT8 ConvRot weight quantization, matching comfy-kitchen's on-disk format.

ComfyUI stores an INT8 ConvRot linear as three tensors:

* ``<name>.weight``       -> int8  [N, K], already rotated (``W @ H^T``)
* ``<name>.weight_scale`` -> float32 [N, 1], per-row absmax scale
* ``<name>.comfy_quant``  -> uint8 [72], the descriptor blob below

and re-derives the math as ``x_rot @ (q * scale)^T`` with the activation rotated online by the
same Hadamard. Because the rotation is orthogonal it changes nothing mathematically; it only
spreads outliers so the per-row INT8 grid fits better.
"""

from __future__ import annotations

import json

import numpy as np

from .hadamard import build_hadamard, rotate

#: 72-byte tensor of this shape carries the descriptor in every quantized checkpoint.
BLOB_LEN = 72
#: Field order/separators matter: ComfyUI parses this with plain ``json.loads``.
BLOB = '{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'


def comfy_quant_blob(group_size: int = 256) -> np.ndarray:
    """The ``comfy_quant`` descriptor tensor (uint8 [72]), null padded."""
    if group_size != 256:
        raise ValueError("this project only writes convrot_groupsize 256")
    blob = BLOB
    if len(blob) > BLOB_LEN:
        raise AssertionError(f"descriptor longer than {BLOB_LEN} bytes: {len(blob)}")
    raw = blob.encode("ascii") + b"\x00" * (BLOB_LEN - len(blob))
    return np.frombuffer(raw, dtype=np.uint8).copy()


def quantize_convrot_weight(
    weight: np.ndarray, group_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate then row-wise symmetric INT8 quantize. Returns ``(int8 [N, K], scale [N, 1])``."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {weight.shape}")
    src = weight.astype(np.float32, copy=False)
    h = build_hadamard(group_size, np.float32)
    rotated = rotate(src, h, group_size)

    abs_max = np.abs(rotated).max(axis=1, keepdims=True)
    scale = np.clip(abs_max.astype(np.float32) / 127.0, 1e-30, None)
    q = np.rint(rotated / scale, out=None).clip(-128.0, 127.0).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_convrot_weight(
    q: np.ndarray, scale: np.ndarray, group_size: int = 256
) -> np.ndarray:
    """Inverse of :func:`quantize_convrot_weight`: ``(q * scale) @ H^T``."""
    rotated = q.astype(np.float32) * scale.reshape(-1, 1)
    h = build_hadamard(group_size, np.float32)
    return rotate(rotated, h, group_size)


def descriptor_fields(blob: np.ndarray) -> dict:
    """Parse a ``comfy_quant`` blob back into its JSON fields (used by the verifier)."""
    text = bytes(np.asarray(blob, dtype=np.uint8)).rstrip(b"\x00").decode("ascii")
    return json.loads(text)
