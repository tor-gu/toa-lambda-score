from math import exp, log

import numpy as np
import pandas as pd
import pytest
from pytest import fixture

from score.fit import make_of, set_up_params


@fixture
def results_1():
    "3 player round robin with circular wins"
    return pd.DataFrame(
        data=[
            {"winner_id": 0, "loser_id": 1},
            {"winner_id": 1, "loser_id": 2},
            {"winner_id": 2, "loser_id": 0},
        ]
    )


@fixture
def results_2():
    "3 player round robin with one dominant player"
    return pd.DataFrame(
        data=[
            {"winner_id": 0, "loser_id": 1},
            {"winner_id": 0, "loser_id": 2},
            {"winner_id": 1, "loser_id": 2},
        ]
    )


def test_set_up_params_1(results_1):
    wins, games = set_up_params(3, results_1)
    assert {0: 1, 1: 1, 2: 1} == wins
    assert {(0, 1): 1, (0, 2): 1, (1, 2): 1} == games


def test_set_up_params_2(results_2):
    wins, games = set_up_params(3, results_2)
    assert {0: 2, 1: 1} == wins
    assert {(0, 1): 1, (0, 2): 1, (1, 2): 1} == games


@pytest.mark.parametrize("sd", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("offset", [-1, 1, 0])
def test_make_of_1(results_1, sd, offset):
    """Test the round-robin tournament with various constant values and sds"""
    of = make_of(3, results_1, sd=sd)
    actual = of([offset] * 3)
    # For this tournament, if x = [offset] * 3, the of reduces to...
    expected = -3 * (offset**2 / (2 * sd**2) + log(2))
    np.testing.assert_approx_equal(expected, actual)


@pytest.mark.parametrize("sd", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("offset", [-1, 1, 0])
def test_make_of_2(results_2, sd, offset):
    """Test the total-order tournament with various constant values and sds"""
    of = make_of(3, results_2, sd=sd)
    actual = of([offset] * 3)
    # For this tournament, if x = [offset] * 3, the of reduces to...
    expected = -3 * (offset**2 / (2 * sd**2) + log(2))
    np.testing.assert_approx_equal(expected, actual)


@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("sd", [0.5, 1.0, 2.0])
def test_make_of_3(results_1, sd, scale):
    """Test round-robin tournament with values (3,2,1) and various sds and scales."""
    of = make_of(3, results_1, sd=sd, scale=scale)
    actual = of([3, 2, 1])
    # For this tournament, with value [3,2,1], the of reduces to...
    w0 = 1  # 1 wins
    w1 = 1  # 1 win
    w2 = 1  # 1 win
    expected = (
        -7 / sd**2
        + scale * (3 * w0 + 2 * w1 + 1 * w2)
        - log(exp(3 * scale) + exp(2 * scale))
        - log(exp(3 * scale) + exp(1 * scale))
        - log(exp(2 * scale) + exp(1 * scale))
    )
    np.testing.assert_approx_equal(expected, actual)


@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("sd", [0.5, 1.0, 2.0])
def test_make_of_4(results_2, sd, scale):
    """Test total-order tournament with values (3,2,1) and various sds and scales."""
    of = make_of(3, results_2, sd=sd, scale=scale)
    actual = of([3, 2, 1])
    # For this tournament, with value [3,2,1], the of reduces to...
    w0 = 2  # 2 wins
    w1 = 1  # 1 win
    w2 = 0  # 0 win
    expected = (
        -7 / sd**2
        + scale * (3 * w0 + 2 * w1 + 1 * w2)
        - log(exp(3 * scale) + exp(2 * scale))
        - log(exp(3 * scale) + exp(1 * scale))
        - log(exp(2 * scale) + exp(1 * scale))
    )
    np.testing.assert_approx_equal(expected, actual)
