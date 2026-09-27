"""The key plan must line up with what ``comfy/ldm/minimax/model.py`` and the loaders expect."""

from slimdit import mapping


def _official_keys(n_blocks=2):
    keys = []
    for i in range(n_blocks):
        p = f"transformer_blocks.{i}"
        keys += [
            f"{p}.adaln_proj.linear.bias",
            f"{p}.adaln_proj.linear.weight",
            f"{p}.attn.norm_k.weight",
            f"{p}.attn.norm_q.weight",
            f"{p}.attn.to_k.weight",
            f"{p}.attn.to_out.0.weight",
            f"{p}.attn.to_q.weight",
            f"{p}.attn.to_v.weight",
            f"{p}.ff.net.0.proj.weight",
            f"{p}.ff.net.2.weight",
            f"{p}.norm1.weight",
            f"{p}.norm2.weight",
        ]
    for i in range(2):
        p = f"token_refiner.refiner_blocks.{i}"
        keys += [
            f"{p}.attn.norm_k.weight",
            f"{p}.attn.norm_q.weight",
            f"{p}.attn.to_k.weight",
            f"{p}.attn.to_out.0.weight",
            f"{p}.attn.to_q.weight",
            f"{p}.attn.to_v.weight",
            f"{p}.ff.net.0.proj.weight",
            f"{p}.ff.net.2.weight",
            f"{p}.norm1.weight",
            f"{p}.norm2.weight",
        ]
    keys += [
        "audio_proj_in.bias",
        "audio_proj_in.weight",
        "audio_proj_out.bias",
        "audio_proj_out.weight",
        "context_embedder.bias",
        "context_embedder.weight",
        "norm_out.linear.bias",
        "norm_out.linear.weight",
        "norm_out.norm.weight",
        "proj_in.bias",
        "proj_in.weight",
        "proj_out.bias",
        "proj_out.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_2.bias",
        "time_embedder.linear_2.weight",
        "rope.inv_freq",
        "token_refiner.final_norm.weight",
    ]
    return keys


def test_every_source_resolves_and_blocks_are_counted():
    keys = _official_keys(n_blocks=3)
    plan = mapping.build_plan(keys)
    assert mapping.block_count(keys) == 3
    assert mapping.missing_sources(plan, keys) == []


def test_output_names_are_the_ones_the_loader_detects():
    keys = _official_keys(n_blocks=1)
    plan = mapping.build_plan(keys)
    outs = {name for op in plan for name in op.out}
    for expected in [
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
        "blocks.0.attn.q_norm.weight",
        "blocks.0.attn.k_norm.weight",
        "blocks.0.norm1.weight",
        "blocks.0.norm2.weight",
        "blocks.0.adaln_proj.linear.weight",
        "video_patch_proj.weight",
        "final_layer.video_out.weight",
        "final_layer.norm.weight",
        "final_layer.adaln_proj.linear.weight",
        "condition_proj.weight",
        "token_refiner.blocks.0.attn.qkv_proj.weight",
    ]:
        assert expected in outs, expected


def test_alignment_projection_keeps_the_pipeline_unquantized():
    """Drbaph's checkpoint quantizes exactly the four per-block linears; everything else is copied."""
    plan = mapping.build_plan(_official_keys(n_blocks=1))
    quantized = {op.out[0] for op in plan if op.kind == "quant"}
    assert quantized == {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
    }
    assert {op.out[0] for op in plan if op.kind == "curve"} == {
        "blocks.0.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.weight",
    }
