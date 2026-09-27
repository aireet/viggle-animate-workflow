"""``rope.inv_freq`` must equal the table the reference checkpoint ships."""

import numpy as np

from slimdit.rope import rope_inv_freq

# Read out of drbaph/Viggle-Animate-ComfyUI's pruned_int8_convrot checkpoint (F32[16]).
REFERENCE = np.array(
    [
        1.0,
        0.5623413324356079,
        0.3162277638912201,
        0.17782793939113617,
        0.10000000149011612,
        0.05623412877321243,
        0.03162277862429619,
        0.017782794311642647,
        0.009999999776482582,
        0.005623413249850273,
        0.003162277862429619,
        0.0017782794311642647,
        0.0010000000474974513,
        0.000562341301701963,
        0.0003162277862429619,
        0.00017782794020604342,
    ],
    dtype=np.float32,
)


def test_matches_reference_table():
    values = rope_inv_freq()
    assert values.shape == (16,)
    assert values.dtype == np.float32
    assert np.array_equal(values, REFERENCE)


def test_is_a_geometric_sequence():
    values = rope_inv_freq()
    assert values[0] == 1.0
    # Each step multiplies by 10000 ** (-1/16) == 10 ** -0.25.
    assert np.allclose(values[1:] / values[:-1], 10 ** -0.25, rtol=1e-6)
