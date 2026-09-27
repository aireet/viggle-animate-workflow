"""The analytic adaln curve: shape contract plus the fit quality on real weights (when present)."""

from pathlib import Path

import numpy as np
import pytest

from slimdit import curve as curve_mod
from slimdit.safetensors_io import ShardSet

OFFICIAL = Path("/root/work/data/viggle-animate/transformer")


def _official_embedder():
    if not OFFICIAL.exists() or not list(OFFICIAL.glob("*.safetensors")):
        pytest.skip("official shards not downloaded")
    src = ShardSet(OFFICIAL)
    needed = ["time_embedder.linear_1.weight", "transformer_blocks.0.adaln_proj.linear.weight"]
    if any(key not in src.manifest for key in needed):
        pytest.skip("official shards still downloading")
    w1 = src.get("time_embedder.linear_1.weight").float().numpy()
    b1 = src.get("time_embedder.linear_1.bias").float().numpy()
    w2 = src.get("time_embedder.linear_2.weight").float().numpy()
    b2 = src.get("time_embedder.linear_2.bias").float().numpy()
    return src, (w1, b1, w2, b2)


def test_time_embedding_shape_and_spread():
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((16, 256)).astype(np.float32)
    b1 = np.zeros(16, np.float32)
    w2 = rng.standard_normal((8, 16)).astype(np.float32)
    b2 = np.zeros(8, np.float32)
    t = np.linspace(0.0, 1.0, 33, dtype=np.float32)
    out = curve_mod.time_embeddings(t, w1, b1, w2, b2)
    assert out.shape == (33, 8)
    assert np.isfinite(out).all()
    assert not np.allclose(out[0], out[-1])


def test_shared_curve_basis_inverts_the_real_modulation_family():
    """Rank-8 input-side basis: the modulation every block sees must survive the projection."""
    src, embedder = _official_embedder()
    table, basis, curve = curve_mod.build_curve_basis(*embedder, rank=8, grid=1025)
    assert table.shape == (1025, 8)
    assert basis.shape == (2688, 8)

    weight = src.get("transformer_blocks.0.adaln_proj.linear.weight").float().numpy()
    projected = curve_mod.project_weight(weight, basis)
    error = curve_mod.curve_residual(weight, curve, table, projected)
    assert error < 5e-4, f"rank-8 curve lost {error * 100:.4f}% of the modulation family"


def test_curve_row_index_is_the_normalized_timestep():
    """ComfyUI indexes the table as ``t.clamp(0,1) * (rows - 1)``; row i must be t = i / (rows - 1)."""
    src, embedder = _official_embedder()
    table, basis, _ = curve_mod.build_curve_basis(*embedder, rank=4, grid=1025)
    t = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    direct = curve_mod.silu(curve_mod.time_embeddings(t, *embedder)) @ basis
    assert np.allclose(table[[0, 256, 1024]], direct, atol=1e-4)
