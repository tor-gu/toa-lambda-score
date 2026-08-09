from math import log

import numpy as np
import pandas as pd

from .consts import COLUMN, FIT_COLUMN, RESULT_COLUMN
from .fit import fit_of, make_of
from .robustness import calculate_robustness


def score(results: list[dict], sd: float, unit_win_prob: float) -> list[dict]:
    """
    results: list of {"winner": str, "loser": str, "match_id": str}
    returns: list of {"id": str, "score": float, "robustness": float}
    """
    results_df = pd.DataFrame(results)
    players = sorted(
        list(
            set(results_df[RESULT_COLUMN.WINNER]) | set(results_df[RESULT_COLUMN.LOSER])
        )
    )
    player_ids = {player: idx for idx, player in enumerate(players)}

    fit_df = results_df.copy()
    fit_df[FIT_COLUMN.WINNER_ID] = fit_df[RESULT_COLUMN.WINNER].map(player_ids)
    fit_df[FIT_COLUMN.LOSER_ID] = fit_df[RESULT_COLUMN.LOSER].map(player_ids)

    scale = log(unit_win_prob / (1 - unit_win_prob))
    n = len(player_ids)
    of = make_of(n, fit_df, sd=sd, scale=scale)
    res = fit_of(of, np.zeros(n))

    id_by_idx = {v: k for k, v in player_ids.items()}
    scores_df = pd.DataFrame(
        {
            COLUMN.ID: [id_by_idx[i] for i in range(n)],
            FIT_COLUMN.SCORE: res.x.round(3),
        }
    )
    scores_df = calculate_robustness(fit_df, scores_df, lmda=scale)
    scores_df[FIT_COLUMN.ROBUSTNESS] = scores_df[FIT_COLUMN.ROBUSTNESS].round(3)

    return scores_df[[COLUMN.ID, FIT_COLUMN.SCORE, FIT_COLUMN.ROBUSTNESS]].to_dict(
        orient="records"
    )
