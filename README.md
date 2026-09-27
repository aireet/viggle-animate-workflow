# slimdit

Analytic checkpoint slimming for ComfyUI diffusion models. First target: **MiniMax-H3
(Viggle-Animate)** — turns the 66 GB official BF16 transformer into a ~21 GB INT8 checkpoint
that stock ComfyUI loads with `Load Diffusion Model`, no custom nodes required.

Everything is computed from the released weights. No distillation, no fine-tuning, no training
data — the two ideas are:

1. **Low-rank adaln curves.** Every block's full-width `adaln_proj` linear (96768 x 2688) consumes
   the *same* time-embedding curve. That curve turns out to be effectively 4-5 dimensional, so a
   rank-8 basis plus a per-block projection reproduces the modulation to **0.019 %** relative
   error while removing ~1.3 B parameters per block of activation-path weights.
2. **INT8 ConvRot.** The remaining block linears (qkv/out/fc1/fc2) are Hadamard-rotated per
   256-group and row-wise INT8 quantized, which is what ComfyUI's `comfy_kitchen` INT8 kernels
   expect — the rotation spreads outlier channels so the per-row grid fits them properly.

## Results on MiniMax-H3 (Viggle-Animate)

| | |
|---|---|
| Curve fit vs exact modulation family | rank-4 → 99.99 %, **rank-8 → 0.019 %** relative error |
| INT8 weight error | ~0.9-1.2 % per row (absmax grid, the layout's ceiling) |
| Checkpoint | 66 GB BF16 → 21 GB INT8 (+ curve tables) |
| ComfyUI compatibility | stock `Load Diffusion Model`; `adaln_curve_grid`/`time_embed_dim` are detected from the tensor shapes |

## Usage

```bash
# 1) fetch the official shards (66 GB)
python3 tools/fetch_official.py --out /models/viggle-official --workers 6

# 2) convert (CPU only, ~20 min)
python -m slimdit.convert \
    --src /models/viggle-official \
    --out /models/viggle-slimdit/minimax_h3_ref2va_slimdit_int8_convrot.safetensors \
    --verify-blocks 4

# 3) sanity check the layout
python3 tools/compare_manifest.py out.safetensors reference.safetensors
```

Drop the result into `ComfyUI/models/diffusion_models/` and load it like any other checkpoint.

## Layout contract

For every quantized linear ComfyUI reads three tensors — this project writes exactly that:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `<name>.weight` | int8 | `[N, K]` | `W @ H^T` rotated, row-wise quantized |
| `<name>.weight_scale` | float32 | `[N, 1]` | per-row `absmax / 127` |
| `<name>.comfy_quant` | uint8 | `[72]` | `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` |

The Hadamard matrix is the *regular* (4x4-seeded, Kronecker) construction from
`comfy_kitchen.tensor.int8_utils`, normalized by `sqrt(256)` — it must match bit-for-bit, because
the runtime un-rotates with its own copy.

Curve checkpoints carry two extra flavours of tensor: a shared `adaln_t_table` `[grid, rank]`
(one row per `t = i / (grid - 1)`, interpolated at runtime) and, per block,
`adaln_proj.linear.weight` of shape `[96768, rank]`.

### Layout gotchas

Two reorderings are mandatory and are *not* visible in tensor shapes — a wrong one turns renders
into noise while every header check still passes:

* `attn.qkv_proj.weight` = `concat([to_q, to_k, to_v])` (the runtime does
  `split(heads * head_dim, dim=-1)`).
* `mlp.fc1.weight` = the official `ff.net.0.proj.weight` **with its two halves swapped**, because
  the runtime evaluates it as `linear_input_act(fc2, fc1(x), "swiglu")` and that activation's half
  order is the opposite of how the released checkpoint stores the gated projection. This applies
  to the token refiner too.

Always finish a conversion with a content comparison against a known-good checkpoint:

```bash
python tools/verify_checkpoint.py --checkpoint ours.safetensors --reference reference.safetensors
```

It dequantizes INT8 tensors on both sides and reports the worst per-tensor deviations, which is
how the fc1 swap above was found.

## Status

- [x] INT8 ConvRot conversion of MiniMax-H3 with analytic adaln curves
- [ ] NVFP4 variant
- [ ] sm89 / sm120 kernel pack (Triton INT8 + curve ops) with arch-aware routing
- [ ] ComfyUI node: convert + inspect checkpoints from inside the graph

## Notes

The analytic construction here is *more* accurate than the reference `pruned_int8_convrot`
checkpoint's curve path: measuring its stored `A @ table^T` against the exact modulation family on
the same timestep grid gives a 30.8 % relative error (its `adaln_t_table` does match the
input-side basis used here to 0.03 degrees, but its per-block matrix does not reproduce the
official modulation). This project instead derives both factors from the released weights.

Algorithms in `slimdit/hadamard.py` and `slimdit/int8.py` mirror `comfy_kitchen`
(Apache-2.0) so that the emitted checkpoints are exactly what the ComfyUI runtime expects.

Apache-2.0.
