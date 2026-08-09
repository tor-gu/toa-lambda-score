import numpy as np
import pandas as pd

from .consts import COLUMN, FIT_COLUMN, RESULT_COLUMN

# Column names for our internal calculations
ROBUSTNESS_CONTRIBUTION = "robustness_contribution"
WINNER_ROBUSTNESS = "winner_robustness"
LOSER_ROBUSTNESS = "loser_robustness"
TOTAL_ROBUSTNESS = "total_robustness"
SCORE_WINNER = f"{FIT_COLUMN.SCORE}_{RESULT_COLUMN.WINNER}"
SCORE_LOSER = f"{FIT_COLUMN.SCORE}_{RESULT_COLUMN.LOSER}"
SCORE_DIFF = "score_diff"
P = "p"


def calculate_robustness(
    matches_df: pd.DataFrame, scores_df: pd.DataFrame, lmda: float = 1.0
) -> pd.DataFrame:
    """Calculate a robustness score for each player. The robustness
    well be added as new column to `scores_df`, which will be returned"""
    # Add the score estimates to the matches DF
    merged_df = merge_scores(matches_df.copy(), scores_df)

    # Calculate the contributions from each match
    contributions = match_contributions(merged_df, lmda)
    merged_df[ROBUSTNESS_CONTRIBUTION] = contributions

    # Aggregate the robustness for the winners and losers separately
    winner_agg = aggregate_robustness(
        merged_df, RESULT_COLUMN.WINNER, WINNER_ROBUSTNESS
    )
    loser_agg = aggregate_robustness(merged_df, RESULT_COLUMN.LOSER, LOSER_ROBUSTNESS)

    # Combine the winner and loser robustness
    total_robustness = calculate_total_robustness(scores_df, winner_agg, loser_agg)

    # And the column and return. The total is positional (the merge in
    # calculate_total_robustness preserves scores_df row order but not its
    # index), so assign the values rather than let pandas align on the index.
    scores_df[FIT_COLUMN.ROBUSTNESS] = total_robustness.to_numpy()
    return scores_df


def aggregate_robustness(
    robustness_df: pd.DataFrame, id_col: str, robustness_col: str
) -> pd.DataFrame:
    """Aggregate (sum) the robustness contributions for each match, for either
    the winners or the losers (specified by `id_col`).

    The result will be returned as a dataframe with an `id_col` column and a
    summed robustness column named `robustness_col`"""
    agg_df = robustness_df.groupby(id_col)[ROBUSTNESS_CONTRIBUTION].sum().reset_index()
    agg_df.rename(columns={ROBUSTNESS_CONTRIBUTION: robustness_col}, inplace=True)
    return agg_df


def calculate_total_robustness(
    scores_df: pd.DataFrame, winner_agg: pd.DataFrame, loser_agg: pd.DataFrame
) -> pd.Series:
    """Combine the rubustness scores for winners and losers into a total
    robustness score for each player."""
    robustness_df = (
        scores_df.copy()
        .merge(winner_agg, left_on=COLUMN.ID, right_on=RESULT_COLUMN.WINNER, how="left")
        .merge(loser_agg, left_on=COLUMN.ID, right_on=RESULT_COLUMN.LOSER, how="left")
        .fillna(0)
    )
    robustness_df[TOTAL_ROBUSTNESS] = (
        robustness_df[WINNER_ROBUSTNESS] + robustness_df[LOSER_ROBUSTNESS]
    )
    return robustness_df[TOTAL_ROBUSTNESS]


def merge_scores(matches_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = matches_df.merge(
        scores_df, left_on=RESULT_COLUMN.WINNER, right_on=COLUMN.ID, how="left"
    )

    # These suffixes will make score_winner and score_loser
    merged_df = merged_df.merge(
        scores_df,
        left_on=RESULT_COLUMN.LOSER,
        right_on=COLUMN.ID,
        how="left",
        suffixes=(f"_{RESULT_COLUMN.WINNER}", f"_{RESULT_COLUMN.LOSER}"),
    )

    return merged_df


def match_contributions(merged_df: pd.DataFrame, lmda: float) -> pd.Series:
    """Calculate the contributions of each match to the robustness score
    of its participants. We need to know the scaling factor lambda in
    order to calculate the probability of victory in each match"""

    # Get the score delta for each match
    contributions_df = merged_df[
        [RESULT_COLUMN.MATCH_ID, SCORE_WINNER, SCORE_LOSER]
    ].copy()
    contributions_df[SCORE_DIFF] = (
        contributions_df[SCORE_WINNER] - contributions_df[SCORE_LOSER]
    )

    # Calculate the probability P of victory for the first player in each
    # match, exp(lambda * delta) / (1 + exp(lambda * delta))
    contributions_df[P] = np.exp(lmda * contributions_df[SCORE_DIFF]) / (
        1 + np.exp(lmda * contributions_df[SCORE_DIFF])
    )
    # The robustness is 4 * P * (1-P)
    contributions_df[ROBUSTNESS_CONTRIBUTION] = (
        4 * contributions_df[P] * (1 - contributions_df[P])
    )

    # Return the robustness scores
    return contributions_df[ROBUSTNESS_CONTRIBUTION]
