"""Analytic low-rank replacement for MiniMax-H3's adaln modulation.

The full-width ``adaln_proj`` linears (96768 x 2688, one per block) dominate the modulation
path's parameter count while all of them consume the *same* time-embedding curve. On a dense
grid of timesteps that curve is effectively low dimensional -- measured on the released
Viggle-Animate weights, the top 8 principal directions of ``silu(time_embedder(t))`` capture all
but ~2e-4 of the modulation energy -- so the whole path collapses to

    table[t]  (grid x rank)     : curve coordinates, shared by every block
    A_block   (96768 x rank)    : W_block @ P

with ``P`` the top-``rank`` right singular vectors of ``silu(time_embedder(t))``. ComfyUI already
supports this checkpoint shape (``adaln_curve_grid`` / ``time_embed_dim`` are detected from the
``adaln_t_table`` tensor's shape) and consumes the table *linearly* -- curve checkpoints set
``apply_silu=False`` -- which is why the fit target is ``W @ silu(embedder(t))^T``.

Fitting each block's own output basis would be marginally more accurate for that block but
cannot share a single ``adaln_t_table``, so the input-side basis is what the format allows.
"""

from __future__ import annotations

import math

import numpy as np

FREQ_DIM = 256
TIME_EMBED_DIM = 2688


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def time_embeddings(
    t: np.ndarray,
    proj_in_w: np.ndarray,
    proj_in_b: np.ndarray,
    proj_out_w: np.ndarray,
    proj_out_b: np.ndarray,
) -> np.ndarray:
    """``comfy.ldm.minimax.model.TimeEmbedder`` forward for fp32 timesteps in [0, 1]."""
    half = FREQ_DIM // 2
    freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = t.astype(np.float32)[:, None] * freqs[None, :]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1).astype(np.float32)
    hidden = silu(emb @ proj_in_w.T + proj_in_b)
    return hidden @ proj_out_w.T + proj_out_b


def build_curve_basis(
    proj_in_w: np.ndarray,
    proj_in_b: np.ndarray,
    proj_out_w: np.ndarray,
    proj_out_b: np.ndarray,
    rank: int = 8,
    grid: int = 1025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(table [grid, rank], basis [TIME_EMBED_DIM, rank], curve)``.

    ``curve`` is ``silu(time_embedder(t))`` for ``t = linspace(0, 1, grid)`` -- the exact input
    the adaln linears see, kept so callers can measure the fit without recomputing it.
    """
    t = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    curve = silu(time_embeddings(t, proj_in_w, proj_in_b, proj_out_w, proj_out_b))
    _, _, vt = np.linalg.svd(curve, full_matrices=False)
    basis = np.ascontiguousarray(vt[:rank].T.astype(np.float32))
    table = (curve @ basis).astype(np.float32)
    return table, basis, curve


def project_weight(weight: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """``A = W @ P`` -- the per-block linear consumed alongside the shared table."""
    return weight.astype(np.float32, copy=False) @ basis


def modulation_family(weight: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Exact modulation for every timestep: ``W @ curve(t)^T`` -> ``[M, grid]``."""
    return weight.astype(np.float32, copy=False) @ curve.T


def curve_residual(
    weight: np.ndarray, curve: np.ndarray, table: np.ndarray, projected: np.ndarray
) -> float:
    """Relative Frobenius error of ``A @ table^T`` against the exact modulation family."""
    exact = modulation_family(weight, curve)
    approx = projected @ table.T
    return float(np.linalg.norm(exact - approx) / np.linalg.norm(exact))
