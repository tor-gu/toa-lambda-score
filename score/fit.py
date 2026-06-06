from math import exp, log

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .consts import FIT_COLUMN


def set_up_params(
    player_count, results, column_names=(FIT_COLUMN.WINNER_ID, FIT_COLUMN.LOSER_ID)
):
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
        - wins_by_team: A dictionary mapping player id to total wins.
        - games_by_matchup: A dictionary mapping (i, j) to total games played between
            player i and player j (with i < j).
        Players with zero wins or zero games are excluded from the respective
        dictionaries.

    """
    N = player_count
    winner, loser = column_names
    wins_by_matchup = {(i, j): 0 for i in range(N) for j in range(N) if i != j}
    for _, row in results.iterrows():
        a, b = row[winner], row[loser]
        wins_by_matchup[(a, b)] += 1
    wins_by_player = {
        i: sum(wins_by_matchup[(i, j)] for j in range(N) if i != j) for i in range(N)
    }
    wins_by_player = {key: val for key, val in wins_by_player.items() if val != 0}
    games_by_matchup = {
        (i, j): wins_by_matchup[(i, j)] + wins_by_matchup[(j, i)]
        for i in range(N)
        for j in range(i + 1, N)
    }
    games_by_matchup = {key: val for key, val in games_by_matchup.items() if val != 0}
    return wins_by_player, games_by_matchup


def make_of(
    player_count,
    results,
    column_names=(FIT_COLUMN.WINNER_ID, FIT_COLUMN.LOSER_ID),
    sd=1.0,
    scale=1.0,
):
    """
    The objective function for a modified version of the Bradley-Terry model, which we
    will maximize to find the best fit strengths.

    The function is defined as:
    of(x) = -sum(x_i^2) / (2 * sd) + sum(wi * x_i) - sum(gij * log(exp(x_i) + exp(x_j)))

    where:
    - x_i is the strength of player i
    - wi is the total wins for player i
    - gij is the total games played between player i and player j
    """

    wins, games = set_up_params(player_count, results, column_names)
    sd2 = 2 * sd

    def of(x: list[float]) -> float:
        return (
            -sum(np.array(x) ** 2) / sd2
            + scale * sum(wi * x[i] for i, wi in wins.items())
            - sum(
                gij * log(exp(scale * x[i]) + exp(scale * x[j]))
                for (i, j), gij in games.items()
            )
        )

    return of


def fit_of(of, initial_strengths, maxiter=1000):
    constraint = LinearConstraint(np.ones((1, len(initial_strengths))), 0, 0)
    res = minimize(
        lambda x: -of(x),
        x0=initial_strengths,
        method="SLSQP",
        options={"maxiter": maxiter},
        constraints=[constraint],
    )
    return res
