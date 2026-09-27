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


def _apply_sigma_shift(model, shift_video, shift_audio):
    """Apply ComfyUI's MiniMax-H3 sampling shift. Raises rather than skipping: silently dropping it
    would change the schedule while still rendering, which looks like a quality regression."""
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    out = MiniMaxH3SigmaShift.execute(model, shift_video, shift_audio)
    return out.result[0] if hasattr(out, "result") else out[0]


class SlimDiTLoader:
    """Everything the model side needs in one node: weights, LoRA, VAE, shifts, step count.

    Wiring this by hand takes six nodes (UNETLoader + LoraLoaderModelOnly + VAELoader +
    MiniMaxH3SigmaShift + KSamplerSelect + a sigma list); here it is one node with the same
    dropdowns you expect from a checkpoint loader, plus the step-count choice. It calls ComfyUI's
    own loader implementations, so behaviour is identical to wiring them yourself.
    """

    CATEGORY = "slimdit"
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL", "VAE", "SIGMAS", "SAMPLER")
    RETURN_NAMES = ("model", "vae", "sigmas", "sampler")
    DESCRIPTION = (
        "Pick the SlimDiT checkpoint (or any other MiniMax-H3 one), the DMD LoRA and the VAE, "
        "choose 3 / 4 / 6 sampling steps, and it outputs the patched model, sigmas and sampler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            checkpoints = folder_paths.get_filename_list("diffusion_models")
            loras = folder_paths.get_filename_list("loras")
            vaes = folder_paths.get_filename_list("vae")
        except Exception:  # noqa: BLE001 - outside ComfyUI
            checkpoints, loras, vaes = [], [], []

        def first(items, needle):
            return next((i for i in items if needle in i.lower()), None)

        return {
            "required": {
                "checkpoint": (
                    checkpoints,
                    {"default": first(checkpoints, "slimdit") or (checkpoints[0] if checkpoints else "")},
                ),
                "lora": (["(none)"] + loras, {"default": first(loras, "viggle") or "(none)"}),
                "vae": (vaes, {"default": first(vaes, "video_vae") or first(vaes, "minimax") or (vaes[0] if vaes else "")}),
                "steps": (["3", "4", "6"], {"default": "3"}),
                "shift_video": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    def run(self, checkpoint, lora, vae, steps, shift_video, shift_audio):
        import torch

        import comfy.samplers
        import nodes as comfy_nodes  # ComfyUI's own loader implementations

        model, = comfy_nodes.UNETLoader().load_unet(checkpoint, "default")
        if lora and lora != "(none)":
            model, = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(model, lora, 1.0)
        vae_out, = comfy_nodes.VAELoader().load_vae(vae)
        sampler = comfy.samplers.sampler_object("euler")
        model = _apply_sigma_shift(model, float(shift_video), float(shift_audio))

        from slimdit.sigmas import format_sigmas, h3_sigmas

        values = h3_sigmas(int(steps), float(shift_video))
        sigmas = torch.tensor(values, dtype=torch.float32)
        text = f"{checkpoint} · lora={lora} · vae={vae}\n{int(steps)} steps: {format_sigmas(values)}"
        print(f"[slimdit/loader] {text}", flush=True)
        return {"ui": {"text": [text]}, "result": (model, vae_out, sigmas, sampler)}


def _node_call(cls, **inputs):
    """Call a ComfyUI node class whether it is V1 (FUNCTION stem) or V3 (classmethod execute).

    Inputs are passed by *name*: the executor does the same, and the positional order of a V1
    method need not match its schema (``VAEDecode.decode`` is ``(vae, samples)``, not the order the
    node shows), so names are the only safe contract.
    """
    if hasattr(cls, "execute") and not hasattr(cls, "FUNCTION"):
        out = cls.execute(**inputs)
    else:
        out = getattr(cls(), cls.FUNCTION)(**inputs)
    return out.result if hasattr(out, "result") else out


class ViggleAnimateSlimDiT:
    """The whole Viggle-Animate render behind three inputs and a Run.

    Driving video, reference still and the driving clip's audio all go *into* this node; it runs the
    vendor chain (clip scaled to 0.4 MP, frozen text conditioning, vendor conditioning build, SlimDiT
    weights + shift, euler sampler on the 3/4/6-step schedule, VAE decode), muxes the audio with the
    frames exactly as the H.264 save node does, and shows the result on itself -- so the graph is
    three nodes and every wire ends here.
    """

    CATEGORY = "slimdit"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Viggle-Animate in one node: driving video + driving audio + reference still -> finished mp4. "
        "Defaults are the evaluated configuration (124 frames, 3 steps, shift 3/3)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            checkpoints = folder_paths.get_filename_list("diffusion_models")
            loras = folder_paths.get_filename_list("loras")
            vaes = folder_paths.get_filename_list("vae")
            text_conds = folder_paths.get_filename_list("text_cond")
        except Exception:  # noqa: BLE001 - outside ComfyUI
            checkpoints, loras, vaes, text_conds = [], [], [], []

        def first(items, needle):
            return next((i for i in items if needle in i.lower()), None)

        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Driving video frames at 24 fps (Load Video). Supplies motion, camera, background."}),
                "reference_image": ("IMAGE", {"tooltip": "Single still of the person to place in the video. A repainted frame of the same shot works best."}),
                "steps": (["3", "4", "6"], {"default": "3", "tooltip": "3 is what the finetune and the DMD LoRA were distilled for."}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                   "tooltip": "Frames at 24 fps, snapped to the 17k+5 grid (124 = ~5.2 s). Clamped to the driving clip's own length."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "checkpoint": (checkpoints, {"default": first(checkpoints, "slimdit") or (checkpoints[0] if checkpoints else "")}),
                "lora": (["(none)"] + loras, {"default": first(loras, "viggle") or "(none)"}),
                "vae": (vaes, {"default": first(vaes, "video_vae") or first(vaes, "minimax") or (vaes[0] if vaes else "")}),
                "text_cond": (text_conds, {"default": first(text_conds, "fixed_embed") or (text_conds[0] if text_conds else "")}),
                "shift_video": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "The driving clip's audio, muxed into the mp4 (Load Video's audio output)."}),
            },
        }

    def run(self, video, reference_image, steps, length, seed, checkpoint, lora, vae, text_cond, shift_video, shift_audio, audio=None):
        import torch

        import comfy.samplers
        import nodes as comfy_nodes
        from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced

        registry = comfy_nodes.NODE_CLASS_MAPPINGS  # every loaded node, core and custom
        missing = [n for n in ("ImageScaleToTotalPixels", "ViggleAnimateConditioning", "ViggleTextCondLoader") if n not in registry]
        if missing:
            raise RuntimeError(f"missing nodes for Viggle-Animate: {', '.join(missing)}")

        frames = int(min(int(length), video.shape[0]))
        if frames != int(length):
            print(f"[slimdit/viggle] length {int(length)} clamped to the clip's {frames} frames", flush=True)

        model, = comfy_nodes.UNETLoader().load_unet(checkpoint, "default")
        if lora and lora != "(none)":
            model, = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(model, lora, 1.0)
        vae_model, = comfy_nodes.VAELoader().load_vae(vae)
        model = _apply_sigma_shift(model, float(shift_video), float(shift_audio))

        cond_video, = _node_call(registry["ImageScaleToTotalPixels"], image=video, upscale_method="area", megapixels=0.4, resolution_steps=32)
        text, = _node_call(registry["ViggleTextCondLoader"], text_cond=text_cond)
        positive, latent = _node_call(
            registry["ViggleAnimateConditioning"],
            cond_video=cond_video,
            ref_image=reference_image,
            text_cond=text,
            vae=vae_model,
            width=0,
            height=0,
            length=frames,
        )

        from slimdit.sigmas import format_sigmas, h3_sigmas

        values = h3_sigmas(int(steps), float(shift_video))
        sigmas = torch.tensor(values, dtype=torch.float32)

        guider = Guider_Basic(model)
        guider.set_conds(positive)
        sampled = _node_call(
            SamplerCustomAdvanced,
            noise=Noise_RandomNoise(int(seed)),
            guider=guider,
            sampler=comfy.samplers.sampler_object("euler"),
            sigmas=sigmas,
            latent_image=latent,
        )
        images, = _node_call(comfy_nodes.VAEDecode, samples=sampled[0], vae=vae_model)

        # Mux and save exactly as the H.264 save node does (same widget values as the vendor
        # workflow, including the driving clip's audio), so the result plays on this node and the
        # file lands in the output folder. A mux failure must not throw away a finished render.
        ui: dict = {}
        if "VHS_VideoCombine" in registry:
            try:
                combined = _node_call(
                    registry["VHS_VideoCombine"],
                    images=images,
                    audio=audio,
                    frame_rate=24,
                    loop_count=0,
                    filename_prefix="viggle/Viggle-Animate",
                    format="video/h264-mp4",
                    pix_fmt="yuv420p",
                    crf=18,
                    save_metadata=True,
                    pingpong=False,
                    trim_to_audio=False,
                    save_output=True,
                )
                ui = combined.get("ui", {}) if isinstance(combined, dict) else {}
            except Exception as exc:  # noqa: BLE001 - keep the frames even if the mux fails
                print(f"[slimdit/viggle] mux/save failed ({type(exc).__name__}: {exc}); frames returned unwrapped", flush=True)
        else:
            print("[slimdit/viggle] VHS_VideoCombine not installed; frames returned unwrapped", flush=True)

        text_out = f"{checkpoint} · lora={lora} · {int(steps)} steps {format_sigmas(values).strip()} · {frames} frames · seed {int(seed)}"
        print(f"[slimdit/viggle] {text_out}", flush=True)
        return {"ui": {**ui, "text": [text_out]}, "result": (images,)}


NODE_CLASS_MAPPINGS = {
    "SlimDiTLoader": SlimDiTLoader,
    "ViggleAnimateSlimDiT": ViggleAnimateSlimDiT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SlimDiTLoader": "SlimDiT Loader (weights + LoRA + VAE + steps)",
    "ViggleAnimateSlimDiT": "Viggle Animate (one node: video + still -> frames)",
}

#: Conversion/inspection helpers. Developer tools, so they stay out of the node library unless
#: explicitly asked for: SLIMDIT_DEV_NODES=1.
if os.environ.get("SLIMDIT_DEV_NODES", "0") not in ("", "0", "false", "False"):
    NODE_CLASS_MAPPINGS.update(
        {
            "SlimDiTConvert": SlimDiTConvert,
            "SlimDiTInspect": SlimDiTInspect,
            "SlimDiTSolAttnStats": SlimDiTSolAttnStats,
            "SlimDiTSigmas": SlimDiTSigmas,
        }
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "SlimDiTConvert": "SlimDiT Convert (INT8 + curves)",
            "SlimDiTInspect": "SlimDiT Inspect Checkpoint",
            "SlimDiTSolAttnStats": "SlimDiT Attention Override Stats",
            "SlimDiTSigmas": "SlimDiT Sigmas (3 / 4 / 6 steps)",
        }
    )


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
