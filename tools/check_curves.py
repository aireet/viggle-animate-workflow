"""Head-to-head: does the community checkpoint's adaln curve reproduce the official modulation?

Reads both checkpoints' (adaln_t_table, per-block A) plus the official bf16 weights, rebuilds the
exact modulation family ``W @ silu(time_embedder(t))`` on the 1025-row grid, and reports the
relative error of ``A @ table^T`` for each checkpoint -- on the full grid and at the four
timesteps the 4-step sampler actually visits.

    python3 check_curves.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/slimdit")

from slimdit import curve as curve_mod  # noqa: E402
from slimdit.safetensors_io import ShardSet  # noqa: E402

OFFICIAL = "/official"
OURS = "/models/viggle/diffusion_models/minimax_h3_ref2va_slimdit_int8_convrot.safetensors"
THEIRS = "/models/viggle/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors"

SIGMAS = [1.0, 0.8571428571428571, 0.6, 0.0]


def main() -> int:
    official = ShardSet(OFFICIAL)
    emb = (
        official.get("time_embedder.linear_1.weight").float().numpy(),
        official.get("time_embedder.linear_1.bias").float().numpy(),
        official.get("time_embedder.linear_2.weight").float().numpy(),
        official.get("time_embedder.linear_2.bias").float().numpy(),
    )
    t = np.linspace(0.0, 1.0, 1025, dtype=np.float32)
    curve = curve_mod.silu(curve_mod.time_embeddings(t, *emb))
    weight = official.get("transformer_blocks.0.adaln_proj.linear.weight").float().numpy()
    exact = weight @ curve.T
    print(f"official modulation: shape {exact.shape}, per-timestep norms {np.round(np.linalg.norm(exact, axis=0)[[0, 146, 410, 1024]], 1)}")

    idx = [int(round((1.0 - s) * 1024)) for s in SIGMAS]
    print(f"sampled timesteps t = {[round(1.0 - s, 4) for s in SIGMAS]} -> table rows {idx}\n")

    for label, path in (("slimdit (ours)", OURS), ("community (pruned_int8_convrot)", THEIRS)):
        slim = ShardSet(path)
        table = slim.get("adaln_t_table").float().numpy()
        projected = slim.get("blocks.0.adaln_proj.linear.weight").float().numpy()
        approx = projected @ table.T
        grid = float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))
        sampled = float(
            np.linalg.norm(approx[:, idx] - exact[:, idx]) / np.linalg.norm(exact[:, idx])
        )
        print(f"{label}:")
        print(f"  table {table.shape} | A {projected.shape} | |A| {np.linalg.norm(projected):.2f} (ratio to ours below)")
        print(f"  full-grid modulation error   {grid * 100:.4f}%")
        print(f"  sampled-timestep error       {sampled * 100:.4f}%")
        print(f"  per-timestep norms at rows {idx}: {np.round(np.linalg.norm(approx[:, idx], axis=0), 1)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
