"""Convert the official Viggle-Animate checkpoint into a curve + INT8 ConvRot checkpoint.

    python -m slimdit.convert --src /root/work/data/viggle-animate/transformer \\
        --out out/minimax_h3_ref2va_slimdit_int8_convrot.safetensors

The output loads in stock ComfyUI (``Load Diffusion Model``): key layout, quantized tensor
format and the ``adaln_t_table`` curve are all detected from the tensor shapes, exactly as the
reference ``pruned_int8_convrot`` checkpoint does. Everything here is analytic -- the curve is a
rank-8 projection of the real time-embedding trajectory and the INT8 weights are rotated with
comfy-kitchen's Hadamard -- so no training step is involved.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from . import curve as curve_mod
from . import mapping, nvfp4
from .int8 import comfy_quant_blob, dequantize_convrot_weight, quantize_convrot_weight
from .safetensors_io import ShardSet, Writer, numel, to_bytes


@dataclass
class Report:
    curve_errors: list[float]
    quant_errors: list[float]

    def summary(self) -> str:
        lines = []
        if self.curve_errors:
            worst = max(self.curve_errors)
            lines.append(f"curve fit: {len(self.curve_errors)} blocks checked, worst modulation error {worst * 100:.5f}%")
        if self.quant_errors:
            arr = np.array(self.quant_errors)
            lines.append(
                f"int8 quant: {len(arr)} layers, per-row error mean {arr.mean() * 100:.4f}% / p95 {np.percentile(arr, 95) * 100:.4f}%"
            )
        return "\n".join(lines) if lines else "(no verification requested)"


def output_manifest(
    ops: list[mapping.Op], src: ShardSet, rank: int, grid: int, block_limit: int | None, quant: str = "int8_convrot"
) -> list[tuple[str, str, tuple[int, ...]]]:
    """Ordered (name, dtype, shape) list for every tensor the writer will emit."""
    manifest: list[tuple[str, str, tuple[int, ...]]] = [("adaln_t_table", "F32", (grid, rank))]
    for op in ops:
        if block_limit is not None and op.out[0].startswith("blocks.") and int(op.out[0].split(".")[1]) >= block_limit:
            continue
        if op.kind == "quant":
            rows = sum(src.manifest[s][1][0] for s in op.src)
            _, cols = src.manifest[op.src[0]][1]
            base = op.out[0][: -len(".weight")]
            if quant == "int8_convrot":
                manifest.append((f"{base}.weight", "I8", (rows, cols)))
                manifest.append((f"{base}.weight_scale", "F32", (rows, 1)))
                manifest.append((f"{base}.comfy_quant", "U8", (72,)))
            elif quant == "nvfp4":
                weight_shape, scale_shape = nvfp4.quantized_shapes(rows, cols)
                manifest.append((f"{base}.weight", "U8", weight_shape))
                manifest.append((f"{base}.weight_scale", "U8", scale_shape))
                manifest.append((f"{base}.weight_scale_2", "F32", ()))
                manifest.append((f"{base}.comfy_quant", "U8", (len(nvfp4.NVFP4_DESCRIPTOR),)))
            else:
                raise ValueError(f"unknown quantization format {quant!r}")
        elif op.kind == "copy":
            manifest.append((op.out[0], op.dtype.upper(), src.manifest[op.src[0]][1]))
        elif op.kind == "concat":
            rows = sum(src.manifest[s][1][0] for s in op.src)
            _, cols = src.manifest[op.src[0]][1]
            manifest.append((op.out[0], op.dtype.upper(), (rows, cols)))
        elif op.kind == "curve":
            out_features = src.manifest[op.src[0]][1][0]
            manifest.append((op.out[0], op.dtype.upper(), (out_features, rank)))
        elif op.kind == "derived":
            from .rope import INV_FREQ_LEN

            manifest.append((op.out[0], op.dtype.upper(), (INV_FREQ_LEN,)))
        else:
            raise ValueError(f"unhandled op kind {op.kind!r}")
    return manifest


def _prepare(tensor: torch.Tensor, op: mapping.Op) -> torch.Tensor:
    """Apply the op's layout fixups to a source tensor before it is stored."""
    if op.swap_halves:
        half = tensor.shape[0] // 2
        if half * 2 != tensor.shape[0]:
            raise ValueError(f"{op.out[0]}: cannot swap halves of {tuple(tensor.shape)}")
        tensor = torch.cat([tensor[half:], tensor[:half]], dim=0)
    return tensor


def convert(
    src_dir: str | Path,
    out_path: str | Path,
    rank: int = 8,
    grid: int = 1025,
    block_limit: int | None = None,
    verify_blocks: int = 0,
    curve_dtype: str = "f16",
    metadata: dict | None = None,
    quant: str = "int8_convrot",
) -> Report:
    src = ShardSet(src_dir)
    ops = mapping.build_plan(src.manifest.keys())
    missing = mapping.missing_sources(ops, src.manifest)
    if missing:
        raise SystemExit(f"missing source tensors: {missing[:5]} (+{max(0, len(missing) - 5)} more)")

    manifest = output_manifest(ops, src, rank, grid, block_limit, quant)
    report = Report(curve_errors=[], quant_errors=[])

    t0 = time.time()
    writer = Writer(out_path, manifest, metadata or {"format": "pt"})
    completed = False
    try:
        # shared curve basis, built from the official time embedder
        table, basis, curve = curve_mod.build_curve_basis(
            src.get(mapping.CURVE_SOURCES[0]).float().numpy(),
            src.get(mapping.CURVE_SOURCES[1]).float().numpy(),
            src.get(mapping.CURVE_SOURCES[2]).float().numpy(),
            src.get(mapping.CURVE_SOURCES[3]).float().numpy(),
            rank=rank,
            grid=grid,
        )
        writer.append("adaln_t_table", to_bytes(torch.from_numpy(table), "F32"))
        del curve  # only needed for the verification pass below

        progress = tqdm(ops, unit="op", desc="convert")
        for op in progress:
            name = op.out[0]
            if block_limit is not None and name.startswith("blocks.") and int(name.split(".")[1]) >= block_limit:
                continue
            progress.set_postfix_str(name, refresh=False)
            if op.kind == "quant":
                parts = [src.get(s).float() for s in op.src]
                weight = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
                weight = _prepare(weight, op)
                weight_np = weight.numpy()
                base = name[: -len(".weight")]
                if quant == "int8_convrot":
                    q, scale = quantize_convrot_weight(weight_np)
                    writer.append(f"{base}.weight", q.tobytes())
                    writer.append(f"{base}.weight_scale", scale.tobytes())
                    writer.append(f"{base}.comfy_quant", comfy_quant_blob().tobytes())
                    deq = dequantize_convrot_weight(q, scale)
                    report.quant_errors.append(float(np.linalg.norm(deq - weight_np) / np.linalg.norm(weight_np)))
                    del q, scale, deq
                else:
                    q, block_scale, tensor_scale, error = nvfp4.quantize_nvfp4_with_error(weight_np)
                    writer.append(f"{base}.weight", q.tobytes())
                    writer.append(f"{base}.weight_scale", block_scale.tobytes())
                    writer.append(f"{base}.weight_scale_2", np.float32(tensor_scale).tobytes())
                    writer.append(f"{base}.comfy_quant", nvfp4.comfy_quant_blob().tobytes())
                    report.quant_errors.append(error)
                    del q, block_scale, tensor_scale
                del parts, weight, weight_np
            elif op.kind == "copy":
                writer.append(name, to_bytes(_prepare(src.get(op.src[0]), op), op.dtype.upper()))
            elif op.kind == "concat":
                parts = [src.get(s) for s in op.src]
                merged = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
                writer.append(name, to_bytes(_prepare(merged, op), op.dtype.upper()))
                del parts, merged
            elif op.kind == "curve":
                weight = src.get(op.src[0]).float().numpy()
                projected = curve_mod.project_weight(weight, basis)
                writer.append(name, to_bytes(torch.from_numpy(projected), curve_dtype.upper()))
            elif op.kind == "derived":
                from .rope import rope_inv_freq

                writer.append(name, rope_inv_freq().tobytes())
            else:
                raise ValueError(f"unhandled op kind {op.kind!r}")
        completed = True
    finally:
        if completed:
            writer.close()
        else:
            writer.abort()

    if verify_blocks:
        report = _verify(src, table, basis, rank, grid, verify_blocks)
    print(f"wrote {out_path} in {time.time() - t0:.1f}s ({len(manifest)} tensors)")
    return report


def _verify(src: ShardSet, table: np.ndarray, basis: np.ndarray, rank: int, grid: int, n_blocks: int) -> Report:
    """Recompute the exact modulation family for the first ``n_blocks`` blocks and measure the fit."""
    _, _, curve = curve_mod.build_curve_basis(
        src.get(mapping.CURVE_SOURCES[0]).float().numpy(),
        src.get(mapping.CURVE_SOURCES[1]).float().numpy(),
        src.get(mapping.CURVE_SOURCES[2]).float().numpy(),
        src.get(mapping.CURVE_SOURCES[3]).float().numpy(),
        rank=rank,
        grid=grid,
    )
    errors: list[float] = []
    for i in tqdm(range(n_blocks), desc="verify", unit="block"):
        weight = src.get(f"transformer_blocks.{i}.adaln_proj.linear.weight").float().numpy()
        projected = curve_mod.project_weight(weight, basis)
        errors.append(curve_mod.curve_residual(weight, curve, table, projected))
        del weight, projected
    return Report(curve_errors=errors, quant_errors=[])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory holding the official transformer shards")
    ap.add_argument("--out", required=True, help="output .safetensors path")
    ap.add_argument("--rank", type=int, default=8, help="curve rank (time_embed_dim), default 8")
    ap.add_argument("--grid", type=int, default=1025, help="adaln_t_table rows, default 1025")
    ap.add_argument("--block-limit", type=int, default=None, help="only convert the first N blocks (trials)")
    ap.add_argument("--verify-blocks", type=int, default=0, help="measure curve fit on the first N blocks")
    ap.add_argument("--curve-dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument(
        "--quant",
        choices=["int8_convrot", "nvfp4"],
        default="int8_convrot",
        help="weight format for the block linears (nvfp4 needs comfy_kitchen, i.e. a ComfyUI env)",
    )
    args = ap.parse_args(argv)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = convert(
        args.src,
        args.out,
        rank=args.rank,
        grid=args.grid,
        block_limit=args.block_limit,
        verify_blocks=args.verify_blocks,
        curve_dtype=args.curve_dtype,
        quant=args.quant,
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
