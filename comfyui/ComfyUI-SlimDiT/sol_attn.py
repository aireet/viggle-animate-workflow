"""Sol-Attn override for the MiniMax-H3 DiT attention, shipped with the SlimDiT node pack.

Why this exists: the DiT's bf16 attention is the single largest cost in a render -- on a 5090 at
480x832x124f it is ~36 s of a ~56 s six-step loop (67 % of sampling, already ~216 TFLOPS = the
card's dense bf16 peak). The int8 GEMMs by contrast are only ~22 % and have no headroom. So the
only lever left in sampling is doing *less* attention work, and the packed sequence gives the
structure to do it safely:

    text | cond / cond_audio / ref_img / ref_audio | target audio | target video

Every conditioning row sits in one contiguous span at the front, so it can be pinned as an exact
key block for every query (``sink_blocks``). ComfyUI's MiniMax-H3 implementation already publishes
that layout -- ``comfy/ldm/minimax/model.py`` stores ``transformer_options["minimax_h3_layout"]``
for exactly this purpose -- so this module only has to read it.

Measured on the service path (same model, same weights, 10 seeds): with sinks pinned the output
is 32.13 dB from the dense baseline, above the 30.85 dB run-to-run numeric floor, i.e. visually
identical, while sampling drops 51.85 s -> 36.26 s (1.42x) and end-to-end 63.7 s -> 46.1 s.

Disabled by default (``SLIMDIT_SOL_ATTN=1`` to enable). When enabled but unusable -- no
comfy_kitchen, no layout in transformer_options, a masked call, or a short sequence -- the
override calls the original attention instead of failing the render.
"""

from __future__ import annotations

import importlib
import os
import time

DEFAULT_MIN_SEQ = 12000  # below this, dense attention wins; matches comfy-kitchen's own guidance
# 0 = no upper bound. The override keeps three contiguous (1, S, H, D) copies of q/k/v alive for
# the kernel call; that grows with the packed sequence, so set a ceiling if you render long clips
# on a small card and would rather fall back than risk an OOM.
DEFAULT_MAX_SEQ = 0

STATS: dict[str, float] = {
    "calls": 0,
    "sparse_calls": 0,
    "fallback_calls": 0,
    "sparse_seconds": 0.0,
    "fallback_seconds": 0.0,
}


def _tensor(value):
    """Unwrap ``AttentionTensorContainer`` (single-owner) or pass a plain tensor through."""
    if hasattr(value, "data_ptr") and hasattr(value, "stride"):
        return value
    take = getattr(value, "take", None)
    if take is not None:
        return take()
    return value


class SolAttnConfig:
    """Knobs for the override; ``from_env`` reads the ``SLIMDIT_SOL_ATTN_*`` names."""

    def __init__(
        self,
        enabled: bool = False,
        topk_ratio: float = 0.0,
        tau: float = 1.0,
        sink_keys: bool = True,
        sink_queries: bool = False,
        tail: bool = True,
        min_seq: int = DEFAULT_MIN_SEQ,
        max_seq: int = DEFAULT_MAX_SEQ,
        fail_closed: bool = False,
    ):
        self.enabled = enabled
        self.topk_ratio = topk_ratio
        self.tau = tau
        self.sink_keys = sink_keys
        self.sink_queries = sink_queries
        self.tail = tail
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.fail_closed = fail_closed

    @classmethod
    def from_env(cls) -> "SolAttnConfig":
        env = os.environ
        return cls(
            enabled=env.get("SLIMDIT_SOL_ATTN", "0") not in ("", "0", "false", "False"),
            topk_ratio=float(env.get("SLIMDIT_SOL_ATTN_TOPK", 0.0)),
            tau=float(env.get("SLIMDIT_SOL_ATTN_TAU", 1.0)),
            sink_keys=env.get("SLIMDIT_SOL_ATTN_SINKS", "1") not in ("0", "false", "False"),
            sink_queries=env.get("SLIMDIT_SOL_ATTN_SINK_Q", "0") not in ("", "0", "false", "False"),
            tail=env.get("SLIMDIT_SOL_ATTN_TAIL", "1") not in ("0", "false", "False"),
            min_seq=int(env.get("SLIMDIT_SOL_ATTN_MIN_SEQ", DEFAULT_MIN_SEQ)),
            max_seq=int(env.get("SLIMDIT_SOL_ATTN_MAX_SEQ", DEFAULT_MAX_SEQ)),
            fail_closed=env.get("SLIMDIT_SOL_ATTN_FAIL_CLOSED", "0") not in ("", "0", "false", "False"),
        )


class SolAttnOverride:
    """Replaces ``comfy.ldm.minimax.model.optimized_attention`` while installed."""

    def __init__(self, config: SolAttnConfig, stats: dict | None = None):
        if not config.enabled:
            raise ValueError("SolAttnOverride requires an enabled config")
        self.config = config
        self.stats = stats if stats is not None else STATS
        self._module = None
        self._original = None
        self._sol_attn = None

    def install(self) -> "SolAttnOverride":
        module = importlib.import_module("comfy.ldm.minimax.model")
        self._module = module
        self._original = module.optimized_attention

        try:
            kitchen = importlib.import_module("comfy_kitchen")
            self._sol_attn = kitchen.sol_attn
            available = kitchen.sol_attn_is_available()
            error = None
        except Exception as exc:  # noqa: BLE001 - environment dependent
            available, error = False, exc

        if not available:
            message = f"comfy_kitchen.sol_attn unavailable: {error}"
            if self.config.fail_closed:
                raise RuntimeError(message)
            print(f"[slimdit/sol-attn] disabled, {message}", flush=True)
            return self
        module.optimized_attention = self
        print(
            f"[slimdit/sol-attn] installed (min_seq={self.config.min_seq}, sinks={self.config.sink_keys}, "
            f"topk={self.config.topk_ratio}, tau={self.config.tau})",
            flush=True,
        )
        return self

    def uninstall(self) -> None:
        if self._module is not None and self._module.optimized_attention is self:
            self._module.optimized_attention = self._original

    def _sink_span(self, kwargs):
        layout = (kwargs.get("transformer_options") or {}).get("minimax_h3_layout")
        if layout is None:
            return None
        first_target = None
        for start, stop, kind in layout.segments:
            if kind in ("audio", "video"):
                first_target = start
                break
        if first_target is None or first_target <= 0:
            return None
        return [0, first_target]

    def __call__(self, q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        started = time.perf_counter()
        self.stats["calls"] += 1
        q_t, k_t, v_t = _tensor(q), _tensor(k), _tensor(v)

        if mask is not None or not skip_reshape:
            return self._fallback(q_t, k_t, v_t, heads, mask, skip_reshape, skip_output_reshape, kwargs, started)

        seq = q_t.shape[2]
        if seq < self.config.min_seq or (0 < self.config.max_seq < seq):
            return self._fallback(q_t, k_t, v_t, heads, mask, skip_reshape, skip_output_reshape, kwargs, started)

        span = self._sink_span(kwargs) if self.config.sink_keys else None
        if self.config.sink_keys and span is None:
            # The layout is what guarantees conditioning rows stay exact; without it the sparse
            # path would approximate them, so fall back rather than degrade silently.
            return self._fallback(q_t, k_t, v_t, heads, mask, skip_reshape, skip_output_reshape, kwargs, started)

        tokens = [value.transpose(1, 2).contiguous() for value in (q_t, k_t, v_t)]
        out = self._sol_attn(
            tokens[0],
            tokens[1],
            tokens[2],
            tau=self.config.tau,
            scale=None,
            sink_blocks=span if self.config.sink_keys else None,
            sink_q=span if (self.config.sink_keys and self.config.sink_queries) else None,
            topk_ratio=self.config.topk_ratio,
            tail=self.config.tail,
        )
        self.stats["sparse_calls"] += 1
        self.stats["sparse_seconds"] += time.perf_counter() - started
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(out.shape[0], out.shape[1], heads * q_t.shape[3])

    def _fallback(self, q_t, k_t, v_t, heads, mask, skip_reshape, skip_output_reshape, kwargs, started):
        out = self._original(
            q_t, k_t, v_t, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs
        )
        self.stats["fallback_calls"] += 1
        self.stats["fallback_seconds"] += time.perf_counter() - started
        return out


def install_sol_attn(config: SolAttnConfig | None = None, stats: dict | None = None) -> SolAttnOverride | None:
    """Install the override (no-op unless enabled) and return the handle."""
    config = SolAttnConfig.from_env() if config is None else config
    if not config.enabled:
        return None
    return SolAttnOverride(config, stats).install()
