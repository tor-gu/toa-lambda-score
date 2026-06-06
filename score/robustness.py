import numpy as np

from util.consts import COLUMN, FIT_COLUMN, RESULT_COLUMN

# Column names for our internal calculations
ROBUSTNESS_CONTRIBUTION = "robustness_contribution"
WINNER_ROBUSTNESS = "winner_robustness"
LOSER_ROBUSTNESS = "loser_robustness"
TOTAL_ROBUSTNESS = "total_robustness"
SCORE_WINNER = f"{FIT_COLUMN.SCORE}_{RESULT_COLUMN.WINNER}"
SCORE_LOSER = f"{FIT_COLUMN.SCORE}_{RESULT_COLUMN.LOSER}"
SCORE_DIFF = "score_diff"
P = "p"


def calculate_robustness(matches_df, scores_df, lmda=1.0):
    merged_df = merge_scores(matches_df.copy(), scores_df)

    contributions = match_contributions(merged_df, lmda)
    merged_df[ROBUSTNESS_CONTRIBUTION] = contributions

    winner_agg = aggregate_robustness(
        merged_df, RESULT_COLUMN.WINNER, WINNER_ROBUSTNESS
    )
    loser_agg = aggregate_robustness(merged_df, RESULT_COLUMN.LOSER, LOSER_ROBUSTNESS)
    total_robustness = calculate_total_robustness(scores_df, winner_agg, loser_agg)
    scores_df[FIT_COLUMN.ROBUSTNESS] = total_robustness
    return scores_df


def aggregate_robustness(robustness_df, id_col, robustness_col):
    agg_df = robustness_df.groupby(id_col)[ROBUSTNESS_CONTRIBUTION].sum().reset_index()
    agg_df.rename(columns={ROBUSTNESS_CONTRIBUTION: robustness_col}, inplace=True)
    return agg_df


def calculate_total_robustness(scores_df, winner_agg, loser_agg):
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


def merge_scores(matches_df, scores_df):
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


def match_contributions(merged_df, lmda):
    contributions_df = merged_df[
        [RESULT_COLUMN.MATCH_ID, SCORE_WINNER, SCORE_LOSER]
    ].copy()
    contributions_df[SCORE_DIFF] = (
        contributions_df[SCORE_WINNER] - contributions_df[SCORE_LOSER]
    )
    contributions_df[P] = np.exp(lmda * contributions_df[SCORE_DIFF]) / (
        1 + np.exp(lmda * contributions_df[SCORE_DIFF])
    )
    contributions_df[ROBUSTNESS_CONTRIBUTION] = (
        4 * contributions_df[P] * (1 - contributions_df[P])
    )
    return contributions_df[ROBUSTNESS_CONTRIBUTION]
