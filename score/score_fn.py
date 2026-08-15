from math import log

import numpy as np
import pandas as pd

from .consts import COLUMN, FIT_COLUMN, RESULT_COLUMN
from .fit import fit_of, make_of
from .robustness import calculate_robustness


def score(
    results: list[dict],
    sd: float,
    unit_win_prob: float,
    initial_strengths: dict[str, float] | None = None,
) -> list[dict]:
    """
    results: list of {"winner": str, "loser": str, "match_id": str}
    initial_strengths: a dictionary (id --> strength) of initial values.
    returns: list of {"id": str, "score": float, "robustness": float}
    """
    # Store the results in a dataframe
    results_df = pd.DataFrame(results)

    # Sort the players by ID and generate a map of of ids to indices.
    # (integers from 0 to n-1). We'll pass these indices (rather than
    # the external player ids) to the scoring function.
    players = sorted(
        list(
            set(results_df[RESULT_COLUMN.WINNER]) | set(results_df[RESULT_COLUMN.LOSER])
        )
    )
    player_ids = {player: idx for idx, player in enumerate(players)}

    # Set up a dataframe to fit the scores to the model. This is a copy
    # of the results_df with winner and loser player indices
    fit_df = results_df.copy()
    fit_df[FIT_COLUMN.WINNER_ID] = fit_df[RESULT_COLUMN.WINNER].map(player_ids)
    fit_df[FIT_COLUMN.LOSER_ID] = fit_df[RESULT_COLUMN.LOSER].map(player_ids)

    # Calculate the scaling factor for the model
    scale = log(unit_win_prob / (1 - unit_win_prob))

    # Generate the objective function
    n = len(player_ids)
    of = make_of(n, fit_df, sd=sd, scale=scale)

    if initial_strengths:
        x0 = np.array([initial_strengths.get(player, 0.0) for player in players])
        x0 -= x0.mean() # Recenter to 0
    else:
        x0 = np.zeros(n)

    # Calculate the fit -- this is where the action happens.
    res = fit_of(of, x0)

    # Generate the scores dataframe from the results
    id_by_idx = {v: k for k, v in player_ids.items()}
    scores_df = pd.DataFrame(
        {
            COLUMN.ID: [id_by_idx[i] for i in range(n)],
            FIT_COLUMN.SCORE: res.x.round(3),
        }
    )

    # Add robustness scores to the dataframe
    scores_df = calculate_robustness(fit_df, scores_df, lmda=scale)
    scores_df[FIT_COLUMN.ROBUSTNESS] = scores_df[FIT_COLUMN.ROBUSTNESS].round(3)

    # Return the result as list of dicts
    return scores_df[[COLUMN.ID, FIT_COLUMN.SCORE, FIT_COLUMN.ROBUSTNESS]].to_dict(
        orient="records"
    )
