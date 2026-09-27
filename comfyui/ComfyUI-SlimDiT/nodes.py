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


class SlimDiTSigmas:
    """MiniMax-H3 sigma schedule with a step-count choice.

    The upstream four-point list ``1.0, 0.8571428571428571, 0.6, 0.0`` is a uniform grid pushed
    through Comfy's sigma shift (``shift=3``): ``2/3 -> 0.857142857`` and ``1/3 -> 0.6`` to the
    digit, so a different step count is just a different grid length through the same shift -- no
    other scheduler node needed.

    3 steps is what the finetune ships with (four sigma points = three Euler updates). 6 steps
    doubles the sampling time and is outside what the DMD LoRA was distilled for.
    """

    CATEGORY = "slimdit"
    FUNCTION = "run"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    OUTPUT_NODE = True
    DESCRIPTION = (
        "3 = upstream default (four-point schedule, three Euler updates). 6 = twice the sampling "
        "time, for comparison -- the DMD LoRA was distilled for the few-step regime."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": (["3", "4", "6"], {"default": "3"}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            }
        }

    def run(self, steps, shift):
        import torch

        try:
            from slimdit.sigmas import format_sigmas, h3_sigmas
        except ImportError:  # standalone/vendored install: same formula, kept in sync by tests

            def h3_sigmas(count, shifted):
                return [shifted * (i / count) / (1.0 + (shifted - 1.0) * (i / count)) for i in range(count, -1, -1)]

            def format_sigmas(values):
                return ", ".join(f"{value:.6g}" for value in values)

        count = int(steps)
        values = h3_sigmas(count, float(shift))
        text = f"{count} steps (shift {shift}): {format_sigmas(values)}"
        print(f"[slimdit/sigmas] {text}", flush=True)
        return {"ui": {"text": [text]}, "result": (torch.tensor(values, dtype=torch.float32),)}


NODE_CLASS_MAPPINGS = {
    "SlimDiTConvert": SlimDiTConvert,
    "SlimDiTInspect": SlimDiTInspect,
    "SlimDiTSolAttnStats": SlimDiTSolAttnStats,
    "SlimDiTSigmas": SlimDiTSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SlimDiTConvert": "SlimDiT Convert (INT8 + curves)",
    "SlimDiTInspect": "SlimDiT Inspect Checkpoint",
    "SlimDiTSolAttnStats": "SlimDiT Attention Override Stats",
    "SlimDiTSigmas": "SlimDiT Sigmas (3 / 4 / 6 steps)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
