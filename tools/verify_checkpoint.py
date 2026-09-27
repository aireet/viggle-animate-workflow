"""Verify a converted checkpoint's contents -- layout parity *and* numeric agreement.

Two modes:

``--reference`` compares every tensor against another slim checkpoint (the reference
``pruned_int8_convrot`` release). INT8 tensors are dequantized on both sides first, so a swapped
row layout, a wrong rotation or a bad scale all show up as a large error. This is the check that
gates a release: identical headers are not enough, a single reordered gated projection is enough
to turn renders into noise.

``--official`` measures the conversion error against the released BF16 weights: INT8 linears
(expected ~1 %) and the adaln curve (expected <0.05 % on the modulation family).

    python tools/verify_checkpoint.py --checkpoint ours.safetensors --reference reference.safetensors
    python tools/verify_checkpoint.py --checkpoint ours.safetensors --official /models/viggle-official
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slimdit import curve as curve_mod  # noqa: E402
from slimdit.int8 import dequantize_convrot_weight, descriptor_fields  # noqa: E402
from slimdit.safetensors_io import ShardSet  # noqa: E402

OFFICIAL_FOR = {
    "attn.qkv_proj.weight": ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"),
    "attn.out_proj.weight": ("attn.to_out.0.weight",),
    "mlp.fc1.weight": ("ff.net.0.proj.weight",),
    "mlp.fc2.weight": ("ff.net.2.weight",),
}


#: The curve path is expected to differ from other releases (ours is derived analytically),
#: so it is excluded from reference comparison and covered by the official-weight mode instead.
CURVE_KEYS = re.compile(r"adaln_t_table|adaln_proj")


def load_value(src: ShardSet, key: str, rotated: bool = False) -> np.ndarray:
    """Tensor as float32, dequantizing INT8 ConvRot or NVFP4 weights.

    ``rotated=True`` returns INT8 ``q * scale`` in the rotated basis, which is equivalent for
    distance comparisons (the un-rotation is a shared orthogonal transform) and much cheaper.
    """
    tensor = src.get(key)
    if key.endswith(".weight"):
        base = key[: -len(".weight")]
        desc_key = f"{base}.comfy_quant"
        fields = descriptor_fields(src.get(desc_key).numpy()) if desc_key in src.manifest else {}
        if fields.get("format") == "nvfp4":
            from slimdit.nvfp4 import _layout

            layout = _layout()
            q = torch.from_numpy(tensor.numpy())
            block_scale = torch.from_numpy(src.get(f"{base}.weight_scale").numpy()).view(torch.float8_e4m3fn)
            rows, cols = q.shape[0], q.shape[1] * 2
            params = layout.Params(
                scale=src.get(f"{base}.weight_scale_2").float(),
                orig_dtype=torch.float32,
                orig_shape=(rows, cols),
                block_scale=block_scale,
            )
            return layout.dequantize(q, params).float().numpy()
        scale_key = f"{base}.weight_scale"
        if scale_key in src.manifest:
            scale = src.get(scale_key).float().numpy()
            if rotated:
                return tensor.numpy().astype(np.float32) * scale.reshape(-1, 1)
            return dequantize_convrot_weight(tensor.numpy(), scale)
    return tensor.float().numpy()


def compare_reference(ours: ShardSet, ref: ShardSet, tolerance: float, limit: int) -> int:
    shared = sorted(k for k in set(ours.manifest) & set(ref.manifest) if not CURVE_KEYS.search(k))
    mismatched_layout = sorted(set(ours.manifest) ^ set(ref.manifest))
    worst: list[tuple[float, str]] = []
    failures = 0
    skipped_format = 0
    for key in shared:
        if key.endswith(".comfy_quant"):
            if bytes(ours.get(key).numpy().tobytes()) != bytes(ref.get(key).numpy().tobytes()):
                print(f"  DESCRIPTOR MISMATCH {key}")
                failures += 1
            continue
        ours_desc = descriptor_fields(ours.get(key[: -len(".weight")] + ".comfy_quant").numpy()) if key.endswith(".weight") else None
        ref_desc = descriptor_fields(ref.get(key[: -len(".weight")] + ".comfy_quant").numpy()) if key.endswith(".weight") else None
        if (ours_desc or {}).get("format") != (ref_desc or {}).get("format"):
            skipped_format += 1
            continue
        a, b = load_value(ours, key, rotated=True), load_value(ref, key, rotated=True)
        if a.shape != b.shape:
            print(f"  SHAPE MISMATCH {key}: {a.shape} vs {b.shape}")
            failures += 1
            continue
        norm = float(np.linalg.norm(b))
        rel = float(np.linalg.norm(a - b) / norm) if norm else 0.0
        worst.append((rel, key))
    worst.sort(reverse=True)
    print(f"compared {len(shared)} tensors (curve keys excluded); {len(mismatched_layout)} keys unique to one side")
    if skipped_format:
        print(f"  {skipped_format} tensors skipped: different quantization format than the reference")
    if mismatched_layout:
        print(f"  one-sided keys (first 5): {mismatched_layout[:5]}")
    print(f"worst {limit} deviations from the reference:")
    for rel, key in worst[:limit]:
        flag = "FAIL" if rel > tolerance else "ok  "
        print(f"  [{flag}] {rel * 100:9.4f}%  {key}")
    failures += sum(1 for rel, _ in worst if rel > tolerance)
    print(f">>> {failures} findings (tolerance {tolerance * 100:.2f}%)")
    return failures


def compare_official(slim: ShardSet, official: ShardSet, layers: list[str], rank: int, grid: int) -> int:
    failures = 0
    for key in layers:
        block = key.split(".")[1]
        suffix = key.split(".", 2)[2]
        parts = [official.get(f"transformer_blocks.{block}.{n}").float() for n in OFFICIAL_FOR[suffix]]
        ref = (parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)).numpy()
        if suffix == "mlp.fc1.weight":
            # The runtime's swiglu takes [up; gate]; the released checkpoint stores [gate; up].
            half = ref.shape[0] // 2
            ref = np.concatenate([ref[half:], ref[:half]], axis=0)
        restored = load_value(slim, key)
        rel = float(np.linalg.norm(restored - ref) / np.linalg.norm(ref))
        print(f"  {key}: quant error {rel * 100:.4f}%")
        failures += rel > 0.15

    emb = (
        official.get("time_embedder.linear_1.weight").float().numpy(),
        official.get("time_embedder.linear_1.bias").float().numpy(),
        official.get("time_embedder.linear_2.weight").float().numpy(),
        official.get("time_embedder.linear_2.bias").float().numpy(),
    )
    table = slim.get("adaln_t_table").float().numpy()
    curve = curve_mod.silu(curve_mod.time_embeddings(np.linspace(0, 1, grid, dtype=np.float32), *emb))
    weight = official.get("transformer_blocks.0.adaln_proj.linear.weight").float().numpy()
    projected = slim.get("blocks.0.adaln_proj.linear.weight").float().numpy()
    residual = curve_mod.curve_residual(weight, curve, table, projected)
    print(f"  blocks.0 curve: modulation error {residual * 100:.5f}% (rank {rank}, grid {grid})")
    failures += residual > 5e-4
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference", default=None, help="another slim checkpoint to compare against")
    ap.add_argument("--official", default=None, help="directory of official shards")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--layers", default="blocks.0.attn.qkv_proj.weight,blocks.0.mlp.fc1.weight,blocks.17.mlp.fc2.weight")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--grid", type=int, default=1025)
    args = ap.parse_args(argv)

    slim = ShardSet(Path(args.checkpoint))
    print(f"checkpoint: {args.checkpoint} ({len(slim.manifest)} tensors)")

    failures = 0
    if args.reference:
        failures += compare_reference(slim, ShardSet(Path(args.reference)), args.tolerance, 10)
    if args.official:
        failures += compare_official(slim, ShardSet(args.official), [k.strip() for k in args.layers.split(",") if k.strip()], args.rank, args.grid)
    print("CONTENTS OK" if not failures else f"CONTENTS SUSPECT ({failures} findings)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
