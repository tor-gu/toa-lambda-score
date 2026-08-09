from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import LinearConstraint, OptimizeResult, minimize

from .consts import FIT_COLUMN


def _winner_loser_arrays(
    results: pd.DataFrame, column_names: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the winner and loser id columns as integer numpy arrays."""
    winner, loser = column_names
    return (
        results[winner].to_numpy(dtype=np.intp),
        results[loser].to_numpy(dtype=np.intp),
    )


def _wins_by_player(player_count: int, winners: np.ndarray) -> np.ndarray:
    """Players are assumed to have ids from 0 to player_count-1.
    Calculate the wins for each player as a numpy array."""
    return np.bincount(winners, minlength=player_count)


def _games_by_matchup(
    player_count: int, winners: np.ndarray, losers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """We're going to calculate games per matchup and return the results
    as three arrays, which look like this (in tabular format):

        | lo_id | hi_id | games |
        | ----- | ----- | ----- |
        | 0     | 1     | 2     |
        | 0     | 2     | 1     |
        | 0     | 4     | 1     |
        | ...   | ...   | ...   |

    We will always have lo_id < hi_id (assuming that there are no 'self-matches')
    and games > 0 (because we omit matchups with no games).

    The calculation is a little slick, to take advantage of using np.unique
    to return a histogram:  We'll encode the id pairs (a,b) as a*N + b,
    where N is the player_count. (And then we'll decode them with
    arithmetic modulo N).

    The encoding is safe as long as player_count < sqrt(MAX_INT64).
    """

    # Convert from winners and losers to lo_id and hi_id
    lo, hi = np.minimum(winners, losers), np.maximum(winners, losers)

    # Encode (a,b) as a*N + b
    encoded_matchups = lo * player_count + hi

    # Calculate the histogram
    keys, pair_games = np.unique(encoded_matchups, return_counts=True)

    # Decode and return
    return keys // player_count, keys % player_count, pair_games


def set_up_params(
    player_count: int,
    results: pd.DataFrame,
    column_names: tuple[str, str] = (FIT_COLUMN.WINNER_ID, FIT_COLUMN.LOSER_ID),
) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    """
    Given a a dataframe of tournement results, calculate the number of wins
    for each player and the number of games played between each pair of players.

    The player IDs are assumed to be integers from 0 to player_count - 1.

    The results dataframe is assumed to have columns for winner and loser ids, specified
    by column_names.

    Args:
        player_count: The number of players in the tournament.
        results: A dataframe with columns for winner and loser ids.
        column_names: A tuple of the column names for winner and loser ids.

    Returns:
        A tuple of two dictionaries:
        - wins_by_player: A dictionary mapping player id to total wins.
        - games_by_matchup: A dictionary mapping (i, j) to total games played between
            player i and player j (with i < j).
        Players with zero wins and matchups with zero games are excluded
        from the respective dictionaries.

    """
    # Extract the numpy arrays
    winners, losers = _winner_loser_arrays(results, column_names)

    # Get wins per player
    wins = _wins_by_player(player_count, winners)

    # Get games per matchup
    low_ids, hi_ids, pair_games = _games_by_matchup(player_count, winners, losers)

    # Now convert back to dicts. We use int() so callers see plain Python 
    # ints rather than numpy scalars.
    wins_by_player = {int(i): int(v) for i, v in enumerate(wins) if v != 0}
    games_by_matchup = {
        (int(i), int(j)): int(g) for i, j, g in zip(low_ids, hi_ids, pair_games)
    }

    # Done
    return wins_by_player, games_by_matchup


def make_of(
    player_count: int,
    results: pd.DataFrame,
    column_names: tuple[str, str] = (FIT_COLUMN.WINNER_ID, FIT_COLUMN.LOSER_ID),
    sd: float = 1.0,
    scale: float = 1.0,
) -> Callable[[npt.ArrayLike], float]:
    """
    The objective function for a modified version of the Bradley-Terry model, which we
    will maximize to find the best fit strengths.

    The function is defined as:
    of(x) = -sum(x_i^2) / (2 * sd^2)
            + sum(wi * x_i)
            - sum(gij * log(exp(x_i) + exp(x_j)))

    where:
    - x_i is the strength of player i
    - wi is the total wins for player i
    - gij is the total games played between player i and player j
    """

    winners, losers = _winner_loser_arrays(results, column_names)
    wins = _wins_by_player(player_count, winners).astype(float)
    pair_i, pair_j, pair_games = _games_by_matchup(player_count, winners, losers)
    pair_games = pair_games.astype(float)
    two_variance = 2 * sd**2

    def of(x: npt.ArrayLike) -> float:
        xa = np.asarray(x, dtype=float)
        sx = scale * xa
        # logaddexp(a, b) is log(exp(a) + exp(b)) without the overflow.
        return (
            -np.dot(xa, xa) / two_variance
            + scale * np.dot(wins, xa)
            - np.dot(pair_games, np.logaddexp(sx[pair_i], sx[pair_j]))
        )

    return of


def fit_of(
    of: Callable[[npt.ArrayLike], float],
    initial_strengths: np.ndarray,
    maxiter: int = 1000,
) -> OptimizeResult:
    constraint = LinearConstraint(np.ones((1, len(initial_strengths))), 0, 0)
    res = minimize(
        lambda x: -of(x),
        x0=initial_strengths,
        method="SLSQP",
        options={"maxiter": maxiter},
        constraints=[constraint],
    )
    return res
