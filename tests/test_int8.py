"""INT8 ConvRot storage contract: the blob ComfyUI parses and the rotation it undoes."""

import numpy as np

from slimdit.hadamard import build_hadamard, rotate
from slimdit.int8 import BLOB_LEN, comfy_quant_blob, dequantize_convrot_weight, descriptor_fields, quantize_convrot_weight

# Byte-for-byte the descriptor inside drbaph/Viggle-Animate-ComfyUI's int8 checkpoints.
REFERENCE_BLOB = (
    b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
)


def test_descriptor_matches_the_runtime_blob():
    blob = comfy_quant_blob()
    assert blob.dtype == np.uint8
    assert blob.shape == (BLOB_LEN,)
    assert bytes(blob[: len(REFERENCE_BLOB)]) == REFERENCE_BLOB
    assert bytes(blob[len(REFERENCE_BLOB):]) == b"\x00" * (BLOB_LEN - len(REFERENCE_BLOB))
    assert descriptor_fields(blob) == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def test_hadamard_is_an_orthonormal_symmetric_basis():
    h = build_hadamard(256)
    assert np.allclose(h, h.T)
    assert np.allclose(h @ h, np.eye(256), atol=1e-5)


def test_rotation_is_its_own_inverse_and_group_wise():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 512)).astype(np.float32)
    h = build_hadamard(256)
    assert np.allclose(rotate(rotate(w, h, 256), h, 256), w, atol=1e-4)
    # Groups must not mix: touching the second half of the features may not move the first half.
    w2 = w.copy()
    w2[:, 256:] += 5.0
    assert np.allclose(rotate(w, h, 256)[:, :256], rotate(w2, h, 256)[:, :256])


def test_quantization_error_beats_the_unrotated_grid_on_outlier_rows():
    """The point of ConvRot: an outlier-heavy row quantizes far better after rotation."""
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 256)).astype(np.float32) * 0.05
    w[:, 0] = 40.0  # one huge channel per row

    q, scale = quantize_convrot_weight(w, 256)
    rotated_err = np.linalg.norm(dequantize_convrot_weight(q, scale, 256) - w) / np.linalg.norm(w)

    abs_max = np.abs(w).max(axis=1, keepdims=True)
    plain_scale = np.clip(abs_max / 127.0, 1e-30, None)
    plain_back = np.rint(w / plain_scale).clip(-128, 127) * plain_scale
    plain_err = np.linalg.norm(plain_back - w) / np.linalg.norm(w)

    assert rotated_err < plain_err / 5
    assert rotated_err < 0.01
