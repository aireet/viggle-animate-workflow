"""Minimal safetensors reader/writer that streams tensor by tensor.

A 21 GB checkpoint cannot be assembled in RAM, so we precompute the output manifest (names,
dtypes, shapes are all known up front), write the header first, then append each tensor's bytes
as it is produced. Reading goes through ``safetensors.safe_open`` with numpy-backed torch
tensors, which memory-maps the shards instead of loading them.
"""

from __future__ import annotations

import json
import struct
from functools import reduce
from operator import mul
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1}
TORCH_TO_ST = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int8: "I8",
    torch.uint8: "U8",
}


def numel(shape) -> int:
    return reduce(mul, shape, 1)


def build_header(manifest: list[tuple[str, str, tuple[int, ...]]], metadata: dict | None) -> bytes:
    """Serialize the safetensors header for ``manifest`` (ordered, name -> dtype/shape)."""
    offset = 0
    entries: dict[str, dict] = {}
    for name, dtype, shape in manifest:
        if dtype not in DTYPE_SIZE:
            raise ValueError(f"unsupported storage dtype {dtype!r} for {name}")
        size = DTYPE_SIZE[dtype] * numel(shape)
        entries[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    payload: dict = {}
    if metadata:
        payload["__metadata__"] = metadata
    payload.update(entries)
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def to_bytes(tensor: torch.Tensor, dtype: str) -> bytes:
    """Convert ``tensor`` to the storage dtype and return its raw little-endian bytes."""
    if dtype == "F32":
        out = tensor.to(torch.float32)
    elif dtype == "F16":
        out = tensor.to(torch.float16)
    elif dtype == "BF16":
        out = tensor.to(torch.bfloat16)
    elif dtype == "I8":
        out = tensor.to(torch.int8)
    elif dtype == "U8":
        out = tensor.to(torch.uint8)
    else:
        raise ValueError(f"unsupported storage dtype {dtype!r}")
    if out.dtype == torch.bfloat16:
        # numpy has no bfloat16; keep the raw 16-bit pattern.
        return out.view(torch.uint16).numpy().tobytes()
    return out.contiguous().numpy().tobytes()


class Writer:
    """Streaming safetensors writer: header first, then tensors in manifest order."""

    def __init__(self, path: str | Path, manifest: list[tuple[str, str, tuple[int, ...]]], metadata: dict | None = None):
        self.path = Path(path)
        self.manifest = manifest
        self._expected = [name for name, _, _ in manifest]
        self._written: list[str] = []
        self._fh = self.path.open("wb")
        self._fh.write(build_header(manifest, metadata))

    def append(self, name: str, payload: bytes) -> None:
        expected = self._expected[len(self._written)]
        if name != expected:
            raise RuntimeError(f"out-of-order write: expected {expected!r}, got {name!r}")
        self._fh.write(payload)
        self._written.append(name)

    def close(self) -> None:
        if len(self._written) != len(self._expected):
            missing = self._expected[len(self._written):][:4]
            self._fh.close()
            raise RuntimeError(f"incomplete write: {len(self._written)}/{len(self._expected)} tensors ({missing}...)")
        self._fh.close()


class ShardSet:
    """The official shards, memory-mapped, keyed by tensor name."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._readers: dict[str, object] = {}
        self.manifest: dict[str, tuple[str, tuple[int, ...]]] = {}
        self.index: dict[str, str] = {}
        for shard in sorted(self.directory.glob("*.safetensors")):
            reader = safe_open(str(shard), framework="pt", device="cpu")
            self._readers[shard.name] = reader
            for key in reader.keys():
                slice_ = reader.get_slice(key)
                raw_dtype = slice_.get_dtype()
                # safetensors >= 0.6 reports the storage name ("BF16"); older ones a torch dtype.
                stored = raw_dtype if isinstance(raw_dtype, str) else TORCH_TO_ST[raw_dtype]
                if stored not in DTYPE_SIZE:
                    raise ValueError(f"unsupported on-disk dtype {stored!r} for {key}")
                self.manifest[key] = (stored, tuple(slice_.get_shape()))
                self.index[key] = shard.name

    def get(self, name: str) -> torch.Tensor:
        reader = self._readers[self.index[name]]
        return reader.get_tensor(name)
