"""Streaming writer/reader round trip -- the layout contract of a converted checkpoint."""

import numpy as np
import torch

from slimdit.safetensors_io import ShardSet, Writer, to_bytes


def test_round_trip_preserves_names_dtypes_and_values(tmp_path):
    manifest = [
        ("adaln_t_table", "F32", (5, 2)),
        ("blocks.0.attn.qkv_proj.weight", "I8", (4, 8)),
        ("blocks.0.attn.qkv_proj.weight_scale", "F32", (4, 1)),
        ("blocks.0.attn.qkv_proj.comfy_quant", "U8", (72,)),
        ("blocks.0.norm1.weight", "BF16", (6,)),
        ("blocks.0.adaln_proj.linear.weight", "F16", (4, 2)),
    ]
    path = tmp_path / "sample.safetensors"
    payloads = {
        "adaln_t_table": np.arange(10, dtype=np.float32).reshape(5, 2),
        "blocks.0.attn.qkv_proj.weight": np.arange(32, dtype=np.int8).reshape(4, 8),
        "blocks.0.attn.qkv_proj.weight_scale": np.full((4, 1), 0.25, dtype=np.float32),
        "blocks.0.attn.qkv_proj.comfy_quant": np.zeros(72, dtype=np.uint8),
        "blocks.0.norm1.weight": np.linspace(-1, 1, 6, dtype=np.float32),
        "blocks.0.adaln_proj.linear.weight": np.linspace(-0.5, 0.5, 8, dtype=np.float32).reshape(4, 2),
    }

    writer = Writer(path, manifest, metadata={"format": "pt"})
    for name, dtype, _ in manifest:
        writer.append(name, to_bytes(torch.from_numpy(payloads[name]), dtype))
    writer.close()

    reader = ShardSet(tmp_path)
    assert dict(reader.manifest) == {name: (dtype, shape) for name, dtype, shape in manifest}
    # Values survive in their own dtype (INT8 exactly, BF16 to bf16 precision).
    assert np.array_equal(reader.get("blocks.0.attn.qkv_proj.weight").numpy(), payloads["blocks.0.attn.qkv_proj.weight"])
    assert np.array_equal(reader.get("adaln_t_table").numpy(), payloads["adaln_t_table"])
    bf16 = reader.get("blocks.0.norm1.weight")
    assert bf16.dtype == torch.bfloat16
    assert np.allclose(bf16.float().numpy(), payloads["blocks.0.norm1.weight"], atol=1e-2)
    assert np.allclose(
        reader.get("blocks.0.adaln_proj.linear.weight").float().numpy(),
        payloads["blocks.0.adaln_proj.linear.weight"],
        atol=1e-3,
    )


def test_writer_rejects_out_of_order_writes(tmp_path):
    manifest = [("a", "F32", (1,)), ("b", "F32", (1,))]
    writer = Writer(tmp_path / "x.safetensors", manifest)
    try:
        writer.append("b", b"\x00\x00\x00\x00")
        raise AssertionError("out-of-order write was accepted")
    except RuntimeError:
        pass
