"""SlimDiT nodes for ComfyUI: convert and inspect slimmed checkpoints without leaving the graph.

The heavy lifting lives in the ``slimdit`` package (same repository, one directory up); these
nodes are thin wrappers that pick sane paths, report progress and never touch graph tensors.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for candidate in (_HERE.parent, _HERE.parent.parent, _HERE.parent.parent.parent):
    if (candidate / "slimdit").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

try:
    import folder_paths  # type: ignore

    _DIFFUSION_DIR = Path(folder_paths.models_dir) / "diffusion_models"
except Exception:  # noqa: BLE001 - outside ComfyUI (unit tests, CLI use)
    _DIFFUSION_DIR = Path.cwd()


DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1}


def diffusion_dir() -> Path:
    _DIFFUSION_DIR.mkdir(parents=True, exist_ok=True)
    return _DIFFUSION_DIR


class SlimDiTConvert:
    """Download the official shards (if needed) and write a curve + INT8 checkpoint."""

    CATEGORY = "slimdit"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "Viggle/Viggle-Animate"}),
                "output_name": ("STRING", {"default": "minimax_h3_ref2va_slimdit_int8_convrot.safetensors"}),
                "curve_rank": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1}),
                "curve_grid": ("INT", {"default": 1025, "min": 65, "max": 4097, "step": 1}),
                "quantize": (["int8_convrot", "nvfp4"], {"default": "int8_convrot"}),
            },
            "optional": {
                "source_dir": ("STRING", {"default": "", "multiline": False}),
                "verify_blocks": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
            },
        }

    def run(self, repo_id, output_name, curve_rank, curve_grid, quantize, source_dir="", verify_blocks=0):
        from slimdit import convert as convert_mod

        source = Path(source_dir).expanduser() if source_dir else Path.home() / ".cache" / "slimdit" / "official"
        if not list(source.glob("*.safetensors")):
            from huggingface_hub import snapshot_download

            print(f"[slimdit] fetching {repo_id} transformer shards into {source}")
            snapshot_download(repo_id=repo_id, allow_patterns=["transformer/*"], local_dir=str(source))
            source = source / "transformer"

        out_path = diffusion_dir() / output_name
        report = convert_mod.convert(
            source,
            out_path,
            rank=curve_rank,
            grid=curve_grid,
            verify_blocks=verify_blocks,
            quant=quantize,
        )
        text = f"slimdit: wrote {out_path}\n{report.summary()}"
        print(text)
        return {"ui": {"text": [text]}, "result": (text,)}


class SlimDiTInspect:
    """Read a checkpoint header and describe its slim structure."""

    CATEGORY = "slimdit"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"checkpoint": ("STRING", {"default": "minimax_h3_ref2va_slimdit_int8_convrot.safetensors"})},
            "optional": {"path": ("STRING", {"default": "", "multiline": False})},
        }

    def run(self, checkpoint, path=""):
        target = Path(path) if path else diffusion_dir() / checkpoint
        if not target.exists():
            raise FileNotFoundError(f"checkpoint not found: {target}")

        with target.open("rb") as fh:
            (length,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(length))
        header.pop("__metadata__", None)

        total = 0
        for entry in header.values():
            size = DTYPE_SIZE.get(entry["dtype"], 1)
            for dim in entry["shape"]:
                size *= dim
            total += size

        quantized = sum(1 for key in header if key.endswith(".comfy_quant"))
        curves = [k for k in header if k.endswith("adaln_proj.linear.weight")]
        table = header.get("adaln_t_table")
        info = {
            "path": str(target),
            "size_gib": round(total / 2**30, 2),
            "tensors": len(header),
            "quantized_linears": quantized,
            "curve": None if table is None else {"grid": table["shape"][0], "rank": table["shape"][1]},
            "curve_linears": len(curves),
            "time_embedder_present": "time_embedder.proj_in.weight" in header,
        }
        text = json.dumps(info, indent=2)
        print(text)
        return {"ui": {"text": [text]}, "result": (text,)}


class SlimDiTSolAttnStats:
    """Report what the MiniMax-H3 attention override has done this session."""

    CATEGORY = "slimdit"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def run(self):
        from . import ATTENTION_STATS, _ATTENTION_HANDLE

        stats = ATTENTION_STATS or getattr(_ATTENTION_HANDLE, "stats", {}) or {}
        mode = os.environ.get("SLIMDIT_ATTN") or os.environ.get("VIGGLE_ATTN") or "off"
        info = {
            "mode": mode,
            "install_hint": "set SLIMDIT_ATTN=auto before starting ComfyUI (sol for short clips, "
                            "the in-place Triton kernel past the frame threshold)",
            "calls": int(stats.get("calls", 0)),
            "sol_calls": int(stats.get("sol_calls", 0)),
            "triton_calls": int(stats.get("triton_calls", 0)),
            "dense_calls": int(stats.get("dense_calls", 0)),
            "sol_oom_fallbacks": int(stats.get("sol_oom", 0)),
        }
        text = json.dumps(info, indent=2)
        print(text)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "SlimDiTConvert": SlimDiTConvert,
    "SlimDiTInspect": SlimDiTInspect,
    "SlimDiTSolAttnStats": SlimDiTSolAttnStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SlimDiTConvert": "SlimDiT Convert (INT8 + curves)",
    "SlimDiTInspect": "SlimDiT Inspect Checkpoint",
    "SlimDiTSolAttnStats": "SlimDiT Attention Override Stats",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
