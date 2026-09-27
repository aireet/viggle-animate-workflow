"""NVFP4 weight quantization for ComfyUI checkpoints.

The on-disk shape of an NVFP4 linear (read by ``comfy/ops.py``) is

    <base>.weight          uint8   [N, K//2]   two E2M1 values per byte, high nibble first
    <base>.weight_scale    uint8   [N, K//16]  E4M3 block scales in the cuBLAS tiled layout
    <base>.weight_scale_2  float32 scalar      per-tensor scale (amax / (448 * 6))
    <base>.comfy_quant     uint8   [19]        {"format": "nvfp4"}

The quantization itself is *not* reimplemented here: it must be bit-identical to what the runtime
dequantizes, and comfy-kitchen already owns that (fp4 round-to-nearest-even via its magic-adder
conversion, fp8-rounded block scales, cuBLAS swizzle). So this module is a thin, explicit bridge:
if ``comfy_kitchen`` is importable we use its layout class; if not, we refuse to guess.

Complements the INT8 ConvRot path in :mod:`slimdit.int8`: half the weight bytes, ~10x the
quantization error, so it is a size/quality trade the caller opts into.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from .int8 import BLOB_LEN

#: Block size of the NVFP4 grid and of the cuBLAS tiled scale layout.
NVFP4_BLOCK = 16
TILE_ROWS = 128
TILE_COLS = 4

NVFP4_DESCRIPTOR = '{"format": "nvfp4"}'


class ComfyKitchenMissing(RuntimeError):
    """Raised when NVFP4 is requested without comfy-kitchen available."""


def _layout():
    try:
        from comfy_kitchen.tensor import TensorCoreNVFP4Layout
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ComfyKitchenMissing(
            "NVFP4 conversion requires comfy_kitchen (it ships with ComfyUI). "
            "Run the converter inside the ComfyUI environment, or use --quant int8_convrot."
        ) from err
    return TensorCoreNVFP4Layout


def nvfp4_available() -> bool:
    try:
        _layout()
    except ComfyKitchenMissing:
        return False
    return True


def comfy_quant_blob() -> np.ndarray:
    """``comfy_quant`` descriptor tensor for an NVFP4 layer."""
    raw = NVFP4_DESCRIPTOR.encode("ascii")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def quantized_shapes(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    """Shapes of ``(weight, weight_scale)`` for an ``[rows, cols]`` NVFP4 linear."""
    if rows % TILE_ROWS or (cols // NVFP4_BLOCK) % TILE_COLS:
        raise ValueError(f"NVFP4 tiling expects rows%128==0 and (cols/16)%4==0, got {rows}x{cols}")
    return (rows, cols // 2), (rows, cols // NVFP4_BLOCK)


def quantize_nvfp4_weight(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize ``[N, K]`` float32 weights into ``(qdata uint8, block_scale uint8, scale f32)``."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {weight.shape}")
    layout = _layout()
    tensor = torch.from_numpy(np.ascontiguousarray(weight)).to(torch.float32)
    qdata, params = layout.quantize(tensor)

    block_scale = params.block_scale
    if block_scale.dtype != torch.uint8:
        block_scale = block_scale.view(torch.uint8)
    return (
        qdata.numpy(),
        np.ascontiguousarray(block_scale.numpy()),
        np.asarray(params.scale.detach().cpu().numpy(), dtype=np.float32),
    )


def quantize_nvfp4_with_error(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """As :func:`quantize_nvfp4_weight`, plus the round-trip error against the input."""
    layout = _layout()
    tensor = torch.from_numpy(np.ascontiguousarray(weight)).to(torch.float32)
    qdata, params = layout.quantize(tensor)
    restored = layout.dequantize(qdata, params).to(torch.float32)
    error = float((restored - tensor).norm() / tensor.norm())
    block_scale = params.block_scale
    if block_scale.dtype != torch.uint8:
        block_scale = block_scale.view(torch.uint8)
    return (
        qdata.numpy(),
        np.ascontiguousarray(block_scale.numpy()),
        np.asarray(params.scale.detach().cpu().numpy(), dtype=np.float32),
        error,
    )


def descriptor_fields(blob: np.ndarray) -> dict:
    text = bytes(np.asarray(blob, dtype=np.uint8)).rstrip(b"\x00").decode("ascii")
    return json.loads(text)


__all__ = [
    "BLOB_LEN",
    "ComfyKitchenMissing",
    "NVFP4_BLOCK",
    "comfy_quant_blob",
    "descriptor_fields",
    "nvfp4_available",
    "quantize_nvfp4_weight",
    "quantized_shapes",
]
