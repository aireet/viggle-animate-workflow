"""The step-count choice must reproduce the upstream schedule exactly."""

import numpy as np

from slimdit.sigmas import DEFAULT_SHIFT, UPSTREAM_SIGMAS, h3_sigmas


def test_three_steps_reproduces_the_upstream_workflow_schedule():
    """The vendor workflow hard-codes this list; our 3-step option must equal it."""
    assert np.allclose(h3_sigmas(3, DEFAULT_SHIFT), UPSTREAM_SIGMAS, rtol=0, atol=1e-15)


def test_shift_formula_matches_the_two_middle_points():
    """1.0/2/3/1/3/0 through shift=3 -> 1.0/0.857142857/0.6/0.0."""
    sigmas = h3_sigmas(3, 3.0)
    assert sigmas[0] == 1.0
    assert sigmas[-1] == 0.0
    assert abs(sigmas[1] - 0.8571428571428571) < 1e-12
    assert abs(sigmas[2] - 0.6) < 1e-12


def test_more_steps_are_a_finer_grid_under_the_same_shift():
    three = h3_sigmas(3)
    six = h3_sigmas(6)
    assert len(six) == 7 and len(three) == 4
    # every 3-step point must also appear in the 6-step schedule (same grid, finer sampling)
    for value in three:
        assert any(abs(value - other) < 1e-12 for other in six)
    # monotone decreasing, terminating at zero
    assert all(a > b for a, b in zip(six, six[1:]))
    assert six[-1] == 0.0


def test_rejects_nonsense():
    for bad in (0, -1):
        try:
            h3_sigmas(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"steps={bad} should raise")
    try:
        h3_sigmas(3, shift=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("shift=0 should raise")
