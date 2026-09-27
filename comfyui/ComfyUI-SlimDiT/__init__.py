"""ComfyUI entry point for the SlimDiT nodes.

Besides the two nodes this package can install an attention override for MiniMax-H3 (the DiT's
attention is the largest single cost in a render; see ``sol_attn.py``). It is opt-in through
``SLIMDIT_SOL_ATTN=1`` and only ever patches the MiniMax-H3 model module, so other models in the
same ComfyUI process are untouched.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    from . import sol_attn as _sol_attn

    _SOL_ATTN_HANDLE = _sol_attn.install_sol_attn()
except Exception as exc:  # noqa: BLE001 - never break node loading over an optional fast path
    _SOL_ATTN_HANDLE = None
    print(f"[slimdit] sol-attn override not installed: {type(exc).__name__}: {exc}", flush=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
