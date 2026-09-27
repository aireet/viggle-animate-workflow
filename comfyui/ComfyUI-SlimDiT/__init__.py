"""ComfyUI entry point for the SlimDiT nodes.

Besides the nodes this package can install an attention override for MiniMax-H3. Attention is the
largest single cost in a render, and no single kernel wins at every clip length on a 32 GiB card:
sol_attn is faster but needs three contiguous q/k/v copies (out of memory at 379 frames), while
the self-written Triton kernel quantizes in place and therefore runs where sol cannot. The router
picks per call from the clip length and live free memory, falling back sol -> kernel -> dense.

Opt-in through ``SLIMDIT_ATTN=auto|sol|triton|off`` (unset = off). It only ever patches the
MiniMax-H3 model module, so other models in the same ComfyUI process are untouched.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

ATTENTION_STATS: dict = {}

try:
    from .attention_router import attention_mode, install_attention

    _MODE = attention_mode()
    if _MODE != "off":
        print(f"[slimdit] attention router mode = {_MODE} (sol / in-place triton / dense)", flush=True)
    _ATTENTION_HANDLE = install_attention(stats=ATTENTION_STATS)
except Exception as exc:  # noqa: BLE001 - never break node loading over an optional fast path
    _ATTENTION_HANDLE = None
    print(f"[slimdit] attention override not installed: {type(exc).__name__}: {exc}", flush=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
