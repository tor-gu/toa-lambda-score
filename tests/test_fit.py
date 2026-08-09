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


# ── tallying edge cases ──────────────────────────────────────────────────────


@fixture
def results_3():
    """Player 0 wins twice and loses once to 1; player 3 never plays at all."""
    return pd.DataFrame(
        data=[
            {"winner_id": 0, "loser_id": 1},
            {"winner_id": 0, "loser_id": 1},
            {"winner_id": 1, "loser_id": 0},
            {"winner_id": 2, "loser_id": 1},
        ]
    )


def test_repeated_matchups_are_summed(results_3):
    _, games = set_up_params(5, results_3)
    # 0 vs 1 played three times regardless of which way round each went.
    assert games == {(0, 1): 3, (1, 2): 1}


def test_players_with_no_wins_or_no_games_are_omitted(results_3):
    wins, games = set_up_params(5, results_3)
    assert wins == {0: 2, 1: 1, 2: 1}
    # 3 and 4 never played, so they appear in neither dict.
    assert 3 not in wins and 4 not in wins
    assert all(3 not in pair and 4 not in pair for pair in games)


def test_set_up_params_returns_plain_ints(results_3):
    wins, games = set_up_params(5, results_3)
    assert all(type(k) is int and type(v) is int for k, v in wins.items())
    assert all(
        type(i) is int and type(j) is int and type(g) is int
        for (i, j), g in games.items()
    )


# ── objective function numerics ──────────────────────────────────────────────


def test_of_accepts_a_list_and_an_array(results_1):
    of = make_of(3, results_1, sd=1.0, scale=1.0)
    # scipy hands of() an ndarray; the tests above hand it a list.
    np.testing.assert_approx_equal(of([1.0, 0.0, -1.0]), of(np.array([1.0, 0.0, -1.0])))


@pytest.mark.parametrize("magnitude", [1e2, 800.0, 1e5])
def test_of_does_not_overflow_on_large_strengths(results_1, magnitude):
    """log(exp(a) + exp(b)) overflows above ~709; logaddexp does not.

    An OverflowError here escapes the Lambda as a 500.
    """
    of = make_of(3, results_1, sd=1.0, scale=1.0)
    assert np.isfinite(of([magnitude, 0.0, -magnitude]))
