"""Official MiniMax-H3 (Viggle-Animate) -> curve-format checkpoint key plan.

The curve format is the layout ComfyUI's ``comfy/ldm/minimax/model.py`` loads (and that the
``ComfyUI-Viggle-Animate-H3`` node documents): ``blocks.*`` instead of ``transformer_blocks.*``,
fused ``qkv_proj``/``out_proj``, the patch/final projections renamed, and an optional
``adaln_t_table`` curve basis that replaces the per-block ``adaln_proj`` linears.

Every entry below was checked against ``drbaph/Viggle-Animate-ComfyUI``'s
``minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors`` (936 tensors, 50 blocks), which is
what the runtime actually consumes.

Ops:
  ``quant``       concatenate sources, rotate, INT8 per-row quantize -> weight + weight_scale
                  + a comfy_quant descriptor.
  ``copy``        pass through unchanged (extra dtypes are cast to the requested storage dtype).
  ``curve``       replace a full-width adaln linear with ``W @ P`` (rank x out_features).
  ``curve_table`` the shared ``adaln_t_table`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["quant", "copy", "concat", "curve", "curve_table", "derived"]

QUANT_BLOCK_LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # output suffix -> official suffixes (concatenated along dim 0 in this order)
    ("attn.qkv_proj.weight", ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight")),
    ("attn.out_proj.weight", ("attn.to_out.0.weight",)),
    ("mlp.fc1.weight", ("ff.net.0.proj.weight",)),
    ("mlp.fc2.weight", ("ff.net.2.weight",)),
)

COPY_BLOCK_LAYERS: tuple[tuple[str, str], ...] = (
    ("norm1.weight", "norm1.weight"),
    ("norm2.weight", "norm2.weight"),
    ("attn.q_norm.weight", "attn.norm_q.weight"),
    ("attn.k_norm.weight", "attn.norm_k.weight"),
)

REFINER_COPY_LAYERS: tuple[tuple[str, str], ...] = (
    ("norm1.weight", "norm1.weight"),
    ("norm2.weight", "norm2.weight"),
    ("attn.q_norm.weight", "attn.norm_q.weight"),
    ("attn.k_norm.weight", "attn.norm_k.weight"),
    ("attn.out_proj.weight", "attn.to_out.0.weight"),
    ("mlp.fc1.weight", "ff.net.0.proj.weight"),
    ("mlp.fc2.weight", "ff.net.2.weight"),
)

#: The reference checkpoint keeps the refiner in BF16, but the official weights ship q/k/v
#: separately, so its fused ``qkv_proj`` has to be assembled here.
REFINER_CONCAT_LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attn.qkv_proj.weight", ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight")),
)

#: The model instantiates these with ``dtype=torch.float32`` (patch/final projections and the
#: unused time embedder), so the reference checkpoint stores them in fp32 -- match it.
GLOBAL_COPY_LAYERS: tuple[tuple[str, str, str], ...] = (
    ("video_patch_proj.weight", "proj_in.weight", "f32"),
    ("video_patch_proj.bias", "proj_in.bias", "f32"),
    ("audio_patch_proj.weight", "audio_proj_in.weight", "f32"),
    ("audio_patch_proj.bias", "audio_proj_in.bias", "f32"),
    ("condition_proj.weight", "context_embedder.weight", "bf16"),
    ("condition_proj.bias", "context_embedder.bias", "bf16"),
    ("final_layer.video_out.weight", "proj_out.weight", "f32"),
    ("final_layer.video_out.bias", "proj_out.bias", "f32"),
    ("final_layer.audio_out.weight", "audio_proj_out.weight", "f32"),
    ("final_layer.audio_out.bias", "audio_proj_out.bias", "f32"),
    ("final_layer.norm.weight", "norm_out.norm.weight", "bf16"),
    ("time_embedder.proj_in.weight", "time_embedder.linear_1.weight", "f32"),
    ("time_embedder.proj_in.bias", "time_embedder.linear_1.bias", "f32"),
    ("time_embedder.proj_out.weight", "time_embedder.linear_2.weight", "f32"),
    ("time_embedder.proj_out.bias", "time_embedder.linear_2.bias", "f32"),
)

#: Sources for the shared curve basis; kept in the checkpoint for non-curve loads.
CURVE_SOURCES = (
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
)

FINAL_ADALN = ("final_layer.adaln_proj.linear", "norm_out.linear")
BLOCK_ADALN = ("blocks.{i}.adaln_proj.linear", "transformer_blocks.{i}.adaln_proj.linear")


@dataclass
class Op:
    """One unit of conversion work: read ``src`` tensors, write ``out`` tensors."""

    out: tuple[str, ...]
    kind: Kind
    src: tuple[str, ...] = ()
    dtype: str = "bf16"
    note: str = ""
    extra: dict = field(default_factory=dict)


def block_count(keys) -> int:
    """Number of distinct ``transformer_blocks.<i>`` indices in ``keys``."""
    return len({k.split(".")[1] for k in keys if k.startswith("transformer_blocks.") and k.split(".")[1].isdigit()})


def build_plan(keys) -> list[Op]:
    """Ordered conversion plan for the official key set (``keys`` may be any container)."""
    keys = set(keys)
    n_blocks = block_count(keys)
    if n_blocks == 0:
        raise ValueError("no transformer_blocks.* keys found -- wrong checkpoint?")

    ops: list[Op] = []
    for i in range(n_blocks):
        src = f"transformer_blocks.{i}"
        dst = f"blocks.{i}"
        for out_suffix, src_suffixes in QUANT_BLOCK_LAYERS:
            ops.append(
                Op(
                    out=(f"{dst}.{out_suffix}",),
                    kind="quant",
                    src=tuple(f"{src}.{s}" for s in src_suffixes),
                )
            )
        for out_suffix, src_suffix in COPY_BLOCK_LAYERS:
            ops.append(Op(out=(f"{dst}.{out_suffix}",), kind="copy", src=(f"{src}.{src_suffix}",)))
        ops.append(
            Op(
                out=(f"{dst}.adaln_proj.linear.weight",),
                kind="curve",
                src=(f"{src}.adaln_proj.linear.weight",),
                dtype="f16",
            )
        )
        ops.append(
            Op(
                out=(f"{dst}.adaln_proj.linear.bias",),
                kind="copy",
                src=(f"{src}.adaln_proj.linear.bias",),
                dtype="f16",
            )
        )

    refiner = sorted({k.split(".")[2] for k in keys if k.startswith("token_refiner.refiner_blocks.")})
    for i in refiner:
        for out_suffix, src_suffixes in REFINER_CONCAT_LAYERS:
            ops.append(
                Op(
                    out=(f"token_refiner.blocks.{i}.{out_suffix}",),
                    kind="concat",
                    src=tuple(f"token_refiner.refiner_blocks.{i}.{s}" for s in src_suffixes),
                )
            )
        for out_suffix, src_suffix in REFINER_COPY_LAYERS:
            ops.append(
                Op(
                    out=(f"token_refiner.blocks.{i}.{out_suffix}",),
                    kind="copy",
                    src=(f"token_refiner.refiner_blocks.{i}.{src_suffix}",),
                )
            )
    ops.append(
        Op(
            out=("token_refiner.final_norm.weight",),
            kind="copy",
            src=("token_refiner.final_norm.weight",),
        )
    )

    for out_name, src_name, copy_dtype in GLOBAL_COPY_LAYERS:
        ops.append(Op(out=(out_name,), kind="copy", src=(src_name,), dtype=copy_dtype))

    ops.append(Op(out=(FINAL_ADALN[0] + ".weight",), kind="curve", src=(FINAL_ADALN[1] + ".weight",), dtype="f16"))
    ops.append(Op(out=(FINAL_ADALN[0] + ".bias",), kind="copy", src=(FINAL_ADALN[1] + ".bias",), dtype="f16"))
    # Not in the official checkpoint: the reference implementation derives it, ComfyUI loads it.
    ops.append(Op(out=("rope.inv_freq",), kind="derived", dtype="f32", note="rope_inv_freq"))
    return ops


def quant_tensor_names(weight_name: str) -> tuple[str, str, str]:
    """The three tensors ComfyUI reads for one INT8 ConvRot linear."""
    base = weight_name[: -len(".weight")]
    return f"{base}.weight", f"{base}.weight_scale", f"{base}.comfy_quant"


def missing_sources(plan: list[Op], keys) -> list[str]:
    keys = set(keys)
    return sorted({s for op in plan for s in op.src if s not in keys})
