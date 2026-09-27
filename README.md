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

The NVFP4 alternative (`--quant nvfp4`, half the weight bytes at ~9.4 % weight error instead of
~0.9 %) stores four tensors per linear:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `<name>.weight` | uint8 | `[N, K/2]` | E2M1 pairs, high nibble first |
| `<name>.weight_scale` | uint8 (fp8 bits) | `[N, K/16]` | E4M3 block scales, cuBLAS tiled layout |
| `<name>.weight_scale_2` | float32 | scalar | `amax / (448 * 6)` |
| `<name>.comfy_quant` | uint8 | `[19]` | `{"format": "nvfp4"}` |

NVFP4 quantization is delegated to `comfy_kitchen` (bit-exact, and its kernels need a GPU during
conversion) rather than reimplemented — a divergent fp4 rounding rule would silently corrupt
weights, so `slimdit` refuses to guess when the package is missing.

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

### ComfyUI caches models by name and size

Re-converting under the same filename can silently keep serving the *old* model: the rewritten
file has the same size (same tensors, same shapes), so the cached entry still matches and renders
keep using the previous weights. When validating a re-conversion, write to a new name (or restart
the server). This cost us a full debug cycle — the rendered noise was from the cached file, not
from the new weights.

## Verification

Three checks, in increasing strength — run all three before publishing a checkpoint:

1. **Layout parity** — `python3 tools/compare_manifest.py ours.safetensors reference.safetensors`.
   Every tensor name, dtype and shape must match a known-good release (936 tensors / 19.59 GiB
   for MiniMax-H3).
2. **Content parity** — `python tools/verify_checkpoint.py --checkpoint ours.safetensors
   --reference reference.safetensors`. Dequantizes INT8 on both sides and prints the worst
   per-tensor deviations; the two files should agree to ~1 % (INT8 noise). This is the check that
   catches reordering mistakes, which headers cannot see.
3. **Conversion error** — `python tools/verify_checkpoint.py --checkpoint ours.safetensors
   --official /models/viggle-official`. Measures INT8 error per layer (0.9 %) and the adaln curve
   fit (0.019 % of the modulation family, 0.064 % at the four timesteps the 4-step sampler uses).

Measured on the released weights:

| check | result |
|---|---|
| file vs reference checkpoint | 936 tensors, same dtypes/shapes, values agree to 0.78 % |
| INT8 weight error | 0.88-0.94 % per layer |
| curve modulation error | 0.019 % (grid) / 0.064 % (sampled timesteps) |
| render, INT8 | 124-frame 480x864 clip, 4-step DMD sampler: **38.4 dB PSNR against the reference checkpoint's render** (visually identical) |
| render, NVFP4 | same structure, visibly softer: 11.73 GiB (−40 %) at 9.4 % weight error, 22.2 dB PSNR against the INT8 build |

### Render timing

`tools/time_checkpoints.py` runs the same 124-frame workflow (480x864, 4-step DMD, seed varied per
run to defeat ComfyUI's per-node result cache) against each checkpoint on one RTX 5090 capped at
400 W:

| checkpoint | first run (loads the model) | steady state | weights staged in VRAM |
|---|---|---|---|
| reference `pruned_int8_convrot` | 61.1 s | 40.3 / 40.5 s | 19,995 MB |
| this project, INT8 | 49.5 s | 40.6 / 40.7 s | 19,995 MB |
| this project, NVFP4 | 47.2 s | 37.0 / 37.1 s | 11,944 MB |

Inside a steady-state run: ~26-28 s for the three sampler steps and ~12-14 s for VAE decode plus
muxing. The sampler is compute-bound at this sequence length (~50 k tokens), so INT8 and NVFP4
land within noise of each other; NVFP4's win is size and 8 GB of VRAM, not speed.

## Status

- [x] INT8 ConvRot conversion of MiniMax-H3 with analytic adaln curves
- [x] NVFP4 variant (`--quant nvfp4`, 11.7 GiB instead of 19.6 GiB, softer renders)
- [x] ComfyUI node: convert + inspect checkpoints from inside the graph
- [x] Attention router shipped in the node pack (sol <=243 frames, in-place Triton kernel beyond, dense fallback), opt-in
- [ ] Mixed INT8/NVFP4: keep sensitive blocks in INT8, NVFP4 for the rest

## Notes on kernels

**The quantized linears are not the lever.** `tools/bench_int8.py` measures comfy-kitchen's INT8
ConvRot linear on the shapes this project emits:

| device | M=1 | M=512 | M=2048 |
|---|---|---|---|
| RTX 5090 (sm_120) | 0.058-0.104 ms, 616-1690 GiB/s | 420-506 TFLOP/s | 520-583 TFLOP/s |
| RTX 4090 (sm_89) | 0.079-0.174 ms, 455-911 GiB/s | 356-477 TFLOP/s | 412-498 TFLOP/s |

Both architectures run at ~75 % of their INT8 peak and at memory-bandwidth limit for a single
token, and a kernel-level profile of the same stack puts INT8 GEMMs at only **21.8 %** of a
six-step sampling loop. NVFP4 was measured in the same place: 460-802 TOPS, i.e. ~1.3x the INT8
throughput on the GEMM alone, which is why the NVFP4 checkpoint buys size and VRAM (~9 % end to
end in our rig) rather than speed.

**Attention is the lever.** The same profile: **bf16 attention is 67.3 % of sampling** (36.0 s of
55.8 s, 300 calls at ~120 ms on a 5090 — already ~216 TFLOPS, the card's dense bf16 peak), while
decode and H.264 encode add ~13 s. Doing *less* attention work is the only remaining big win, and
the packed MiniMax-H3 sequence has the structure for it: text and conditioning rows sit in one
contiguous span at the front, so they can be pinned as exact keys (`sink_blocks`) while only the
target rows are attended sparsely. ComfyUI already publishes that layout
(`transformer_options["minimax_h3_layout"]`), so the node pack ships the override
(`comfyui/ComfyUI-SlimDiT/sol_attn.py`, opt-in via `SLIMDIT_SOL_ATTN=1`, falls back to dense when
the layout is missing or the sequence is short).

Measured on the service path with identical weights and 10 seeds: sampling 51.85 s -> 36.26 s
(1.42x), end to end 63.7 s -> 46.1 s, and the output sits 32.13 dB from the dense baseline, i.e.
above the 30.85 dB run-to-run numeric floor.

Measured again on this project's ComfyUI rig (124 frames, 4-step DMD, seed varied per run):

| | override off | override on | |
|---|---|---|---|
| first run (loads the model) | 61.3 s | 56.3 s | |
| **steady state** | **41.2 s** | **33.1 s** | **-20 %** |
| sampler's three steps | 26 s | 22 s | |
| model-init warmup forward | 14.3 s | 7.7 s | |
| output vs dense, same seed | | 39.0 dB / 35.2 dB | above the 30.85 dB numeric floor |

**No single kernel wins at every length**, so the pack ships a router
(`comfyui/ComfyUI-SlimDiT/attention_router.py`) instead of one override. Measured on the service
path at 6 steps (same weights, 32 GiB card):

| clip | sol_attn | in-place Triton kernel |
|---|---|---|
| 124 frames | 33.2 s, 26.6 GiB | 38.3 s, 25.4 GiB |
| 243 frames | 103.0 s, 28.2 GiB | 116.4 s, 25.9 GiB |
| 379 frames | **OOM (>31.36 GiB)** | **313.7 s, 29.39 GiB** |

sol_attn is faster but must keep three contiguous bf16 copies of q/k/v alive; the self-written
kernel quantizes in place and therefore runs where sol cannot. The router chooses per call:

* `sol` when the clip is short (<= 243 frames) **and** measured free memory covers the copies
  (with a 1.5 GiB reserve and a 1.15x safety factor),
* `triton` otherwise, and as the automatic fallback if sol still raises out-of-memory,
* `dense` for anything the router does not recognise (a mask, a non-packed call, short sequences).

Enable with `SLIMDIT_ATTN=auto` before starting ComfyUI (`sol` / `triton` / `off` force one branch);
the legacy `VIGGLE_ATTN`, `VIGGLE_SOL_ATTN` and `VIGGLE_TRITON_ATTN` names still work. The router
prints its install line at startup, and `SlimDiT Attention Override Stats` reports how many calls
each branch took plus any sol->kernel OOM fallbacks. It only ever patches
`comfy.ldm.minimax.model.optimized_attention`, so other models in the same process are unaffected.

**Why the copies matter (and why the router is the answer).** sol_attn hands the kernel three
contiguous `(1, S, H, D)` tensors, materialised from the model's packed buffer. At the production
shape (S = 30026, H = 56, D = 128, bf16), `tools/measure_attn_copies.py` measures:

| | |
|---|---|
| three copies per attention call | 1.71 ms, 1232 MiB moved |
| over 300 calls per 124-frame render | **0.51 s** (≈1.5 % of the render) |
| peak VRAM added | **+1.20 GiB** |

So the copies are a *headroom* cost, not a time cost — and at 124 frames the render leaves only
~1.3-1.6 GiB free on a 32 GiB card, which is exactly why sol_attn runs out of memory at longer
clips. `tools/probe_sol_attn_strided.py` confirms sol_attn itself rejects the strided views, so
making *it* zero-copy would need a rewrite of that kernel; the router instead switches to the
in-place Triton kernel (no copies at all) once the length or the free memory says sol cannot fit,
which is the same result without touching comfy-kitchen.

## Comparison with the community checkpoint

`drbaph/Viggle-Animate-ComfyUI`'s `pruned_int8_convrot` is the same idea, and the two files come
out near-equivalent: 936 tensors each, identical dtypes/shapes, 19.59 GiB both. Measured with
`tools/check_curves.py` (adaln) and `tools/verify_checkpoint.py` (weights):

| | slimdit (this project) | community `pruned_int8_convrot` |
|---|---|---|
| adaln curve error, full grid | 0.0256 % | 0.0256 % |
| adaln curve error, at the four sampled timesteps | 0.0643 % | 0.0643 % |
| per-block `A` vs the other file | identical up to a global sign (`|A|` matches to 4 digits) | |
| INT8 error vs official bf16, 5 layers | 0.88-1.20 % | 0.91-1.22 % |
| quantized-domain difference between the two files | 0.70 % | |
| warm render, stock ComfyUI (124 frames) | 40.6 s | 40.3 s |

The curve path is therefore **not** a differentiator: both files independently land on the same
analytic rank-8 construction (a shared input-side basis plus `A = W @ P`, up to a sign that
cancels inside `A @ table^T`). What this project adds is the reproducible pipeline (converter plus
the content checks that caught the fc1 swiglu order), an NVFP4 variant, the ComfyUI node pack, and
the attention override that takes a render from ~41 s to ~33 s.

> An earlier revision of this file claimed the community curve does not reproduce the official
> modulation. That was wrong — it came from a stale local tensor dump, not from the checkpoint.
> `tools/check_curves.py` re-derives the numbers from the actual files; trust it over prose.

Algorithms in `slimdit/hadamard.py` and `slimdit/int8.py` mirror `comfy_kitchen`
(Apache-2.0) so that the emitted checkpoints are exactly what the ComfyUI runtime expects.

Apache-2.0.
