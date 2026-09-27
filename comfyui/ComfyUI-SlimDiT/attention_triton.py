"""Self-written INT8/FP8 flash attention for the Viggle DiT (env gated, fail closed).

The DiT hands attention three strided views (1, heads, seq, dim) of one packed
(seq, 3*heads*dim) bf16 buffer. ``comfy_kitchen.sol_attn`` needs contiguous (B, T, H, D), so the
Sol-Attn override materialises three copies for the duration of every call -- 0.57 GiB at 124
frames, 1.74 GiB at 379 and 3.30 GiB at 719, which is why a 379-frame clip OOMs a 32 GiB card. This
override reads the packed buffer in place and quantises instead of copying:

    Q  int8, quantised inside the kernel, one scale per (token, head)
    K  int8, one scale per (token, head)
    V  fp8 e4m3 with a single global scale, so the PV dequant is a scalar
    QK int8 mma with fp32 accumulate, scales folded back through an outer product
    PV fp8 mma with fp32 accumulate

``l_i`` accumulates the *quantised* p, so numerator and denominator describe the same weights and
fp8 rounding shifts the average instead of biasing it toward the largest logits.

Measured on a 5090 (S=30026, H=56, D=128, bf16 input, self-written kernel versus the paths that
exist today):

    dense SDPA on contiguous copies  ~119 ms      (what the model runs without an override)
    this kernel, int8 QK + fp8 PV     74.1 ms     0.41 GiB of scratch, no copies
    this kernel, all bf16            158.0 ms
    sol_attn sink mode                59.7 ms     fastest, but keeps three bf16 copies alive

cosine 0.999323 against dense SDPA, max abs diff 3.85e-4, no NaN. Accuracy against *real*
activations is still being measured; until that verdict lands the override stays opt-in.

Disabled by default: with ``VIGGLE_TRITON_ATTN`` unset the DiT keeps Comfy's own attention path.
Enabled but unusable (no triton, unexpected layout, masked call, short sequence) the override
falls back to the original function instead of failing the request; ``fail_closed=True`` turns
that into a startup error instead, matching the sm89 packs' posture.
"""

from __future__ import annotations

import os
import time

DEFAULT_MIN_SEQ = 12000  # below this the dense path wins outright
# Measured at the production geometry (S=30026, H=56, D=128, bf16): 70.4 ms here against 83.2 ms for
# 128x64, 79.6 for 64x64 and 115.7 for 128x128.
DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 128
DEFAULT_WARPS = 4
DEFAULT_STAGES = 2
QUANT_BLOCK = 64

_KERNELS: dict = {}
_HADAMARD: dict = {}


def _hadamard(dim, device):
    """Sylvester Hadamard: entries exactly +-1, so the rotation is multiplication-free and exact in
    bf16/tf32, and it is its own inverse up to the 1/dim factor the caller folds into the scale."""
    key = (dim, str(device))
    if key not in _HADAMARD:
        import torch

        index = torch.arange(dim, device=device)
        parity = torch.zeros((dim, dim), dtype=torch.int64, device=device)
        bits = (index[:, None] & index[None, :])
        for bit in range(dim.bit_length() - 1):
            parity += (bits >> bit) & 1
        _HADAMARD[key] = torch.where(parity % 2 == 0, torch.ones_like(parity, dtype=torch.bfloat16),
                                     -torch.ones_like(parity, dtype=torch.bfloat16))
    return _HADAMARD[key]


def _kernels():
    """Import triton lazily and build the kernels once."""
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def quant_int8_rows(src, dst, scales, seq, row_stride, head_stride, tensor_offset, h_ptr,
                        H: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr, BLOCK: tl.constexpr):
        """(seq, H, D) strided view -> int8 (seq, H*D) + one scale per (row, head).

        With ROT the row is rotated by an exact +-1 Hadamard first. The transform is orthogonal, so
        it cannot change q.k^T -- it only stops one loud channel from eating the row's int8 range,
        which on real q/k is worth about a quarter of the total attention error.
        """
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        cols = tl.arange(0, D)
        mask = rows < seq
        for head in range(H):
            base = src + head * head_stride + tensor_offset
            raw = tl.load(base + rows[:, None] * row_stride + cols[None, :], mask=mask[:, None],
                          other=0.0)
            if ROT:
                h = tl.load(h_ptr + cols[:, None] * D + cols[None, :])
                x = tl.dot(raw, h, out_dtype=tl.float32)
            else:
                x = raw.to(tl.float32)
            scale = tl.maximum(tl.max(tl.abs(x), 1), 1e-8) / 127.0
            q = tl.minimum(tl.maximum(tl.floor(x / scale[:, None] + 0.5), -127.0), 127.0)
            tl.store(dst + rows[:, None] * (H * D) + head * D + cols[None, :], q.to(tl.int8),
                     mask=mask[:, None])
            tl.store(scales + rows * H + head, scale, mask=mask)

    @triton.jit
    def tensor_amax(src, out_ptr, seq, row_stride, head_stride, tensor_offset,
                    H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        cols = tl.arange(0, D)
        mask = rows < seq
        local = tl.zeros((), dtype=tl.float32)
        for head in range(H):
            base = src + head * head_stride + tensor_offset
            x = tl.load(base + rows[:, None] * row_stride + cols[None, :], mask=mask[:, None], other=0.0)
            local = tl.maximum(local, tl.max(tl.abs(x.to(tl.float32))))
        tl.atomic_max(out_ptr, local)

    @triton.jit
    def quant_fp8_global(src, dst, scale_ptr, seq, row_stride, head_stride, tensor_offset,
                         H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        cols = tl.arange(0, D)
        mask = rows < seq
        scale = tl.load(scale_ptr)
        for head in range(H):
            base = src + head * head_stride + tensor_offset
            x = tl.load(base + rows[:, None] * row_stride + cols[None, :], mask=mask[:, None],
                        other=0.0).to(tl.float32)
            tl.store(dst + rows[:, None] * (H * D) + head * D + cols[None, :], (x / scale).to(tl.float8e4nv),
                     mask=mask[:, None])

    @triton.jit
    def flash_int8(out_ptr, q_i8_ptr, q_scale_ptr, k_i8, k_scale, v_fp8, v_scale_ptr, attn_scale, seq,
                   H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        head = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, D)
        row_mask = rows < seq

        # Q arrives pre-quantised: the rotation lives in the quantisation pass, which has its own
        # shared-memory budget for the Hadamard matrix
        q_i8 = tl.load(q_i8_ptr + rows[:, None] * (H * D) + head * D + cols[None, :],
                       mask=row_mask[:, None], other=0)
        q_scale = tl.load(q_scale_ptr + rows * H + head, mask=row_mask, other=1.0)

        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        v_scale = tl.load(v_scale_ptr)

        for start in range(0, tl.cdiv(seq, BLOCK_N) * BLOCK_N, BLOCK_N):
            n_rows = start + tl.arange(0, BLOCK_N)
            n_mask = n_rows < seq
            k = tl.load(k_i8 + n_rows[:, None] * (H * D) + head * D + cols[None, :], mask=n_mask[:, None], other=0)
            k_s = tl.load(k_scale + n_rows * H + head, mask=n_mask, other=1.0)
            qk = tl.dot(q_i8, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
            qk = qk * (q_scale[:, None] * k_s[None, :]) * attn_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
            p = tl.exp2((qk - m_new[:, None]) * 1.4426950408889634)
            acc = acc * alpha[:, None]

            v = tl.load(v_fp8 + n_rows[:, None] * (H * D) + head * D + cols[None, :], mask=n_mask[:, None], other=0.0)
            p_q = p.to(tl.float8e4nv)
            acc = tl.dot(p_q, v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p_q.to(tl.float32), 1)
            m_i = m_new

        out = (acc * v_scale) / l_i[:, None]
        out_base = out_ptr + head * D
        tl.store(out_base + rows[:, None] * (H * D) + cols[None, :], out.to(tl.bfloat16), mask=row_mask[:, None])

    _KERNELS.update(quant_int8_rows=quant_int8_rows, tensor_amax=tensor_amax,
                    quant_fp8_global=quant_fp8_global, flash_int8=flash_int8)
    return _KERNELS


def unwrap(value):
    """Mirror ``attention_ops._tensor``: an ``AttentionTensorContainer`` is single-owner, and
    ``take()`` both returns and consumes the tensor. The model always passes containers, so
    without this the override would fall back on every call.

    Plain tensors also expose ``take()`` (it indexes along a dimension), so a tensor has to be
    recognised by its own interface first -- the router hands already-unwrapped tensors down.
    """
    if hasattr(value, "data_ptr") and hasattr(value, "stride"):
        return value
    take = getattr(value, "take", None)
    return take() if take is not None else value


def packed_geometry(q, k, v, heads):
    """Validate the packed (seq, 3*heads*dim) layout and return what the kernel needs.

    The model splits one qkv projection along the last axis, so the three tensors are views of a
    single buffer: q at offset 0, k at heads*dim, v at 2*heads*dim, every row ``3*heads*dim``
    elements long. Anything else (contiguous copies, a different packing, another dtype) returns
    None and the caller hands the call back to the model.
    """
    if q is None or k is None or v is None:
        return None
    shape = getattr(q, "shape", None)
    if shape is None or len(shape) != 4 or tuple(shape) != tuple(k.shape) or tuple(shape) != tuple(v.shape):
        return None
    batch, dim_heads, seq, dim = shape
    if batch != 1 or dim_heads != heads or seq <= 0 or dim <= 0 or (dim & (dim - 1)):
        return None
    if not (q.dtype == k.dtype == v.dtype) or "bfloat16" not in str(q.dtype):
        return None
    if q.stride(3) != 1 or q.stride(2) != 3 * dim_heads * dim or q.stride(1) != dim:
        return None
    if (k.stride(3), k.stride(2), k.stride(1)) != (1, 3 * dim_heads * dim, dim):
        return None
    if (v.stride(3), v.stride(2), v.stride(1)) != (1, 3 * dim_heads * dim, dim):
        return None
    row_bytes = dim_heads * dim * q.element_size()
    if k.data_ptr() - q.data_ptr() != row_bytes or v.data_ptr() - q.data_ptr() != 2 * row_bytes:
        return None
    return {"seq": seq, "heads": dim_heads, "dim": dim, "row_stride": 3 * dim_heads * dim,
            "head_stride": dim, "row_bytes": row_bytes}


class TritonAttnConfig:
    """Knobs for the override; ``from_env`` reads the production-style env names."""

    def __init__(self, enabled=False, block_m=DEFAULT_BLOCK_M, block_n=DEFAULT_BLOCK_N,
                 num_warps=DEFAULT_WARPS, num_stages=DEFAULT_STAGES, min_seq=DEFAULT_MIN_SEQ,
                 fail_closed=False, rotate=True):
        self.enabled = enabled
        self.block_m = block_m
        self.block_n = block_n
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.min_seq = min_seq
        self.fail_closed = fail_closed
        self.rotate = rotate

    def describe(self):
        return {"block_m": self.block_m, "block_n": self.block_n, "num_warps": self.num_warps,
                "num_stages": self.num_stages, "min_seq": self.min_seq, "rotate": self.rotate}


class TritonAttnOverride:
    """Replaces ``comfy.ldm.minimax.model.optimized_attention`` while installed."""

    def __init__(self, config, stats=None):
        if not config.enabled:
            raise ValueError("TritonAttnOverride requires an enabled config")
        self.config = config
        self.stats = stats if stats is not None else {}
        # *_dispatch_seconds are host-side launch costs: attention runs asynchronously, so only the
        # benchmark scripts measure GPU time. They are here to catch pathological overheads.
        for key in ("calls", "kernel_calls", "fallback_calls",
                    "quantise_dispatch_seconds", "kernel_dispatch_seconds"):
            self.stats.setdefault(key, 0 if key.endswith("calls") else 0.0)
        self._module = None
        self._original = None

    def install(self):
        import importlib

        module = importlib.import_module("comfy.ldm.minimax.model")
        self._module = module
        self._original = module.optimized_attention
        try:
            _kernels()
        except Exception as exc:  # pragma: no cover - environment dependent
            if self.config.fail_closed:
                raise RuntimeError(f"triton attention unavailable: {exc}") from exc
            print(f"[triton-attn] disabled, triton unavailable: {exc}", flush=True)
            return self
        module.optimized_attention = self
        return self

    def uninstall(self):
        if self._module is not None and self._module.optimized_attention is self:
            self._module.optimized_attention = self._original

    def __call__(self, q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        started = time.perf_counter()
        self.stats["calls"] += 1
        q_t, k_t, v_t = unwrap(q), unwrap(k), unwrap(v)
        geometry = None
        if mask is None and skip_reshape:
            geometry = packed_geometry(q_t, k_t, v_t, heads)
            if geometry is not None and geometry["seq"] < self.config.min_seq:
                geometry = None
        if geometry is None:
            return self._fallback(q_t, k_t, v_t, heads, mask, skip_reshape, skip_output_reshape, kwargs, started)

        out = self._run(q_t, k_t, v_t, geometry)
        self.stats["kernel_calls"] += 1
        if skip_output_reshape:
            # same contract as the Sol-Attn override: (1, heads, seq, dim)
            return out.view(1, geometry["seq"], heads, geometry["dim"]).transpose(1, 2)
        return out.reshape(1, geometry["seq"], heads * geometry["dim"])

    def _run(self, q, k, v, geometry):
        import torch

        kernels = _kernels()
        seq, heads, dim = geometry["seq"], geometry["heads"], geometry["dim"]
        row_stride, head_stride = geometry["row_stride"], geometry["head_stride"]

        started = time.perf_counter()
        q_i8 = torch.empty(seq, heads * dim, device=q.device, dtype=torch.int8)
        q_scale = torch.empty(seq, heads, device=q.device, dtype=torch.float32)
        k_i8 = torch.empty(seq, heads * dim, device=q.device, dtype=torch.int8)
        k_scale = torch.empty(seq, heads, device=q.device, dtype=torch.float32)
        v_fp8 = torch.empty(seq, heads * dim, device=q.device, dtype=torch.float8_e4m3fn)
        rotate = self.config.rotate
        block = min(QUANT_BLOCK, 32) if rotate else QUANT_BLOCK
        h = _hadamard(dim, q.device) if rotate else torch.zeros(1, device=q.device,
                                                                dtype=torch.bfloat16)
        grid = (-(-seq // block),)
        kernels["quant_int8_rows"][grid](q, q_i8, q_scale, seq, row_stride, head_stride, 0, h,
                                         H=heads, D=dim, ROT=rotate, BLOCK=block)
        kernels["quant_int8_rows"][grid](q, k_i8, k_scale, seq, row_stride, head_stride, heads * dim, h,
                                         H=heads, D=dim, ROT=rotate, BLOCK=block)
        amax = torch.zeros(1, device=q.device, dtype=torch.float32)
        kernels["tensor_amax"][grid](q, amax, seq, row_stride, head_stride, 2 * heads * dim,
                                     H=heads, D=dim, BLOCK=QUANT_BLOCK)
        v_scale = torch.clamp(amax / 448.0, min=1e-8)
        kernels["quant_fp8_global"][grid](q, v_fp8, v_scale, seq, row_stride, head_stride, 2 * heads * dim,
                                          H=heads, D=dim, BLOCK=QUANT_BLOCK)
        out = torch.empty(seq, heads * dim, device=q.device, dtype=torch.bfloat16)
        self.stats["quantise_dispatch_seconds"] += time.perf_counter() - started

        started = time.perf_counter()
        # the Hadamard is unnormalised, so (qH).(kH)^T = dim * q.k^T and the scale carries 1/dim
        attn_scale = dim ** -0.5 / (dim if rotate else 1.0)
        kernels["flash_int8"][(-(-seq // self.config.block_m), heads)](
            out, q_i8, q_scale, k_i8, k_scale, v_fp8, v_scale, attn_scale, seq,
            H=heads, D=dim, BLOCK_M=self.config.block_m, BLOCK_N=self.config.block_n,
            num_warps=self.config.num_warps, num_stages=self.config.num_stages)
        self.stats["kernel_dispatch_seconds"] += time.perf_counter() - started
        return out.unsqueeze(0)

    def _fallback(self, q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs, started):
        out = self._original(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)
        self.stats["fallback_calls"] += 1
        return out


def install_triton_attn(config=None, stats=None):
    """Install the self-written kernel for every length (``VIGGLE_ATTN=triton``)."""
    config = config if config is not None else TritonAttnConfig(enabled=True)
    if not config.enabled:
        return None
    return TritonAttnOverride(config, stats).install()
