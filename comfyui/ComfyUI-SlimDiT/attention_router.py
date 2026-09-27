"""Pick the attention implementation per call instead of per process (env gated, fail closed).

The measurements say no single choice wins at every clip length on a 32 GiB card:

    124 frames   sol 33.2 s / 26.6 GiB   this kernel 38.3 s / 25.4 GiB
    243 frames   sol 103.0 s / 28.2 GiB  this kernel 116.4 s / 25.9 GiB
    379 frames   sol OOM (>31.36 GiB)    this kernel 313.7 s / 29.39 GiB

sol_attn is the faster, more accurate kernel but it needs three contiguous bf16 copies of q/k/v
alive for the duration of the call; this kernel quantises in place and therefore runs where sol
cannot. That makes the choice a function of *free memory at call time*, not of a configuration
file, since the same service shares a card with other work:

    sol      when the copies provably fit (measured free memory, plus a reserve)
    triton   otherwise, and as the automatic fallback if sol still raises out-of-memory
    dense    the model's own path for anything this router does not recognise

Every branch keeps the same output contract, so the DiT cannot tell which one ran.

Disabled by default: with ``SLIMDIT_ATTN`` unset the engine keeps the model's own attention path.
"""

from __future__ import annotations

import os
import time

DEFAULT_MIN_SEQ = 12000
# Up to this clip length sol_attn is the better kernel and its three contiguous copies still fit a
# 32 GiB card (measured: 26.6 GiB peak at 124 frames, 28.2 at 243, out-of-memory at 379). Above it
# the self-written kernel is the only accelerated path. Routing on the request length keeps a
# request reproducible; the memory guard below only exists so a shared card cannot fail the request.
DEFAULT_MAX_SOL_FRAMES = 243
DEFAULT_RESERVE_GIB = 1.5  # leave room for the rest of the step, not just for the copies
DEFAULT_SAFETY = 1.15  # sol also builds its own workspace on top of the copies


def _empty_cache():
    import torch

    torch.cuda.empty_cache()


MODE_ENV = "SLIMDIT_ATTN"
LEGACY_MODE_ENV = "VIGGLE_ATTN"


def attention_mode(environ=None):
    """The one switch an operator needs.

    ``SLIMDIT_ATTN=auto``   pick per request: sol up to MAX_SOL_FRAMES, the kernel above (recommended)
    ``SLIMDIT_ATTN=off``    leave the model's own attention alone (default)
    ``SLIMDIT_ATTN=sol``    force sol for every length (debug/compare)
    ``SLIMDIT_ATTN=triton`` force the self-written kernel for every length

    The older single-kernel gates (VIGGLE_SOL_ATTN / VIGGLE_TRITON_ATTN) still work as aliases.
    """
    environ = os.environ if environ is None else environ
    mode = environ.get(MODE_ENV, "").strip().lower() or environ.get(LEGACY_MODE_ENV, "").strip().lower()
    if mode in ("auto", "off", "sol", "triton"):
        return mode
    if environ.get("VIGGLE_SOL_ATTN", "0") not in ("", "0", "false", "False"):
        return "sol"
    if environ.get("VIGGLE_TRITON_ATTN", "0") not in ("", "0", "false", "False"):
        return "triton"
    return "off"


class RouterConfig:
    """Thresholds, not knobs: every number here comes from a measurement (see RESULTS.md).

    MIN_SEQ         below this the dense path wins anyway; overhead is not worth it
    MAX_SOL_FRAMES  sol peaks at 26.6 GiB for 124 frames and 28.2 for 243, and runs out of memory
                    at 379 -- 243 is the last length that fits a 32 GiB card with headroom
    RESERVE_GIB     sol also allocates its own workspace on top of the copies
    SAFETY          margin on the three contiguous bf16 copies sol needs
    """

    def __init__(self, min_seq=12000, max_sol_frames=243, reserve_gib=1.5, safety=1.15,
                 memory_guard=True, over_budget_choice="dense", short_seq_choice="dense"):
        self.min_seq = min_seq
        self.max_sol_frames = max_sol_frames
        self.reserve_gib = reserve_gib
        self.safety = safety
        self.memory_guard = memory_guard
        # Measured at 372 frames on a 32 GiB card: dense 26.1 GiB / clean, the in-place int8
        # kernel 31.8 GiB / ghosting. So when sol does not fit, dense is both the safest and
        # the cleaner choice; the kernel stays reachable via SLIMDIT_ATTN=triton.
        self.over_budget_choice = over_budget_choice
        self.short_seq_choice = short_seq_choice

    def describe(self):
        return {"min_seq": self.min_seq, "max_sol_frames": self.max_sol_frames,
                "over_budget": self.over_budget_choice, "short_seq": self.short_seq_choice}


class AttentionRouter:
    """Replaces ``comfy.ldm.minimax.model.optimized_attention`` while installed."""

    def __init__(self, config, triton_config=None, sol_config=None, stats=None):
        from .attention_triton import TritonAttnConfig, TritonAttnOverride

        self.config = config
        self.stats = stats if stats is not None else {}
        for key in ("sol_calls", "triton_calls", "dense_calls", "sol_oom"):
            self.stats.setdefault(key, 0)
        # the router's own gate decides when this runs, so the per-kernel gate is forced on here
        triton_config = triton_config or TritonAttnConfig()
        triton_config.enabled = True
        self._triton = TritonAttnOverride(triton_config)
        self._sol = None
        self._sol_config = sol_config
        self._module = None
        self._original = None
        self._free_bytes = None
        self._oom = ()
        self._empty_cache = _empty_cache
        self._frames = None

    def install(self):
        import importlib

        import torch

        module = importlib.import_module("comfy.ldm.minimax.model")
        self._module = module
        self._original = module.optimized_attention
        self._free_bytes = torch.cuda.mem_get_info
        self._oom = (torch.OutOfMemoryError,)
        # sol is the faster branch of "auto", so probe it once; if it is missing the router simply
        # runs the self-written kernel everywhere
        from .sol_attn import SolAttnConfig, SolAttnOverride

        try:
            handle = SolAttnOverride(self._sol_config or SolAttnConfig(enabled=True))
            handle.install()
            handle.uninstall()  # probe only: the router drives it directly
            self._sol = handle
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[attention-router] sol_attn unavailable, kernel only: {exc}", flush=True)
        module.optimized_attention = self
        return self

    def uninstall(self):
        if self._module is not None and self._module.optimized_attention is self:
            self._module.optimized_attention = self._original

    def set_request(self, frames):
        """The engine declares the clip it is about to render; routing follows the request."""
        self._frames = int(frames) if frames else None

    def choose(self, geometry, heads):
        """The kernel the *request length* calls for, unless memory says sol cannot run."""
        if geometry is None:
            return "dense"
        if self._sol is None or geometry["seq"] < self.config.min_seq:
            return self.config.short_seq_choice
        if self._frames is not None and self._frames > self.config.max_sol_frames:
            return self.config.over_budget_choice  # this length is known not to fit; do not even try
        if not self.config.memory_guard:
            return "sol"
        copies = 3 * geometry["seq"] * heads * geometry["dim"] * 2
        need = copies * self.config.safety + self.config.reserve_gib * 2 ** 30
        free, _ = self._free_bytes()
        return "sol" if free >= need else self.config.over_budget_choice

    def __call__(self, q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        started = time.perf_counter()
        if mask is not None or not skip_reshape:
            return self._dense(q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs, started)

        from .attention_triton import packed_geometry, unwrap

        q_t, k_t, v_t = unwrap(q), unwrap(k), unwrap(v)
        geometry = packed_geometry(q_t, k_t, v_t, heads)
        choice = self.choose(geometry, heads)

        if choice == "sol":
            try:
                out = self._sol(q_t, k_t, v_t, heads, mask=None, skip_reshape=True,
                                skip_output_reshape=skip_output_reshape, **kwargs)
                self.stats["sol_calls"] += 1
                return out
            except self._oom:
                # measured memory said it fits and it still did not: drop to the kernel that
                # needs no copies at all rather than failing the request
                self.stats["sol_oom"] += 1
                self._empty_cache()
                choice = self.config.over_budget_choice

        if choice == "triton":
            out = self._triton(q_t, k_t, v_t, heads, mask=None, skip_reshape=True,
                               skip_output_reshape=skip_output_reshape, **kwargs)
            self.stats["triton_calls"] += 1
            return out

        return self._dense(q_t, k_t, v_t, heads, mask, skip_reshape, skip_output_reshape, kwargs, started)

    def _dense(self, q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs, started):
        self.stats["dense_calls"] += 1
        return self._original(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                              skip_output_reshape=skip_output_reshape, **kwargs)


def install_attention(config=None, stats=None):
    """Install whatever ``VIGGLE_ATTN`` asks for, or nothing when it is unset/off."""
    from .attention_triton import TritonAttnConfig, install_triton_attn

    mode = attention_mode()
    if mode == "auto":
        return AttentionRouter(config or RouterConfig(), TritonAttnConfig(), stats=stats).install()
    if mode == "triton":
        return install_triton_attn(TritonAttnConfig(), stats=stats)
    if mode == "sol":
        from .sol_attn import SolAttnConfig, install_sol_attn

        return install_sol_attn(SolAttnConfig(enabled=True), stats=stats)
    return None
