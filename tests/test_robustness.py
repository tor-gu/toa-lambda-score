import numpy as np
import pandas as pd
import pytest
from pytest import fixture

from score.robustness import (
    aggregate_robustness,
    calculate_robustness,
    calculate_total_robustness,
    match_contributions,
    merge_scores,
)

# 4·p·(1−p) at lambda=1 for score gaps of 1 and 2, i.e. the two matches in the fixture.
GAP_1_CONTRIBUTION = 0.7864477329659274
GAP_2_CONTRIBUTION = 0.41997434161402614

EXPECTED_ROBUSTNESS = {
    "a": GAP_1_CONTRIBUTION + GAP_2_CONTRIBUTION,  # wins both matches
    "b": GAP_1_CONTRIBUTION,
    "c": GAP_2_CONTRIBUTION,
}


@fixture
def scores():
    "Three players a > b > c, one point apart."
    return pd.DataFrame({"id": ["a", "b", "c"], "score": [1.0, 0.0, -1.0]})


@fixture
def matches():
    "`a` beats `b` (gap 1.0) and `a` beats `c` (gap 2.0)."
    return pd.DataFrame(
        data=[
            {"match_id": "1", "winner": "a", "loser": "b"},
            {"match_id": "2", "winner": "a", "loser": "c"},
        ]
    )


@fixture
def merged(matches, scores):
    "`matches` with both players' scores attached."
    return merge_scores(matches.copy(), scores)


@fixture
def contributions_df(merged):
    "`merged` with the per-match robustness contribution attached."
    merged["robustness_contribution"] = match_contributions(merged, 1.0)
    return merged


@fixture
def winner_agg(contributions_df):
    return aggregate_robustness(contributions_df, "winner", "winner_robustness")


@fixture
def loser_agg(contributions_df):
    return aggregate_robustness(contributions_df, "loser", "loser_robustness")


def merged_with_gaps(*gaps):
    "A minimal merged-style frame with one match per score gap."
    return pd.DataFrame(
        {
            "match_id": [str(i) for i in range(len(gaps))],
            "score_winner": list(gaps),
            "score_loser": [0.0] * len(gaps),
        }
    )


# ── match_contributions ──────────────────────────────────────────────────────


def test_equal_scores_give_the_maximum_contribution():
    """Gap 0 means p=0.5, and 4·p·(1−p) peaks at 1 there."""
    contributions = match_contributions(merged_with_gaps(0.0), 1.0)
    np.testing.assert_approx_equal(contributions.iloc[0], 1.0)


@pytest.mark.parametrize("gap", [0.5, 1.0, 2.0, 5.0])
def test_contribution_is_symmetric_in_the_score_gap(gap):
    """An upset and an expected win of the same size count the same."""
    contributions = match_contributions(merged_with_gaps(gap, -gap), 1.0)
    np.testing.assert_approx_equal(contributions.iloc[0], contributions.iloc[1])


def test_a_larger_gap_gives_a_smaller_contribution():
    contributions = list(
        match_contributions(merged_with_gaps(0.0, 0.5, 1.0, 2.0, 5.0), 1.0)
    )
    assert all(a > b for a, b in zip(contributions, contributions[1:]))


@pytest.mark.parametrize("lmda", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("gap", [-10.0, -1.0, 0.0, 1.0, 10.0])
def test_contributions_are_between_zero_and_one(gap, lmda):
    contributions = match_contributions(merged_with_gaps(gap), lmda)
    assert 0.0 <= contributions.iloc[0] <= 1.0


def test_a_larger_lambda_sharpens_the_contribution():
    """lambda scales the gap, so a fixed gap counts for less as lambda grows."""
    contributions = [
        match_contributions(merged_with_gaps(1.0), lmda).iloc[0]
        for lmda in (0.5, 1.0, 3.0)
    ]
    assert all(a > b for a, b in zip(contributions, contributions[1:]))


@pytest.mark.parametrize(
    "gap,expected", [(1.0, GAP_1_CONTRIBUTION), (2.0, GAP_2_CONTRIBUTION)]
)
def test_contribution_matches_the_closed_form(gap, expected):
    contributions = match_contributions(merged_with_gaps(gap), 1.0)
    np.testing.assert_approx_equal(contributions.iloc[0], expected)


# ── merge_scores ─────────────────────────────────────────────────────────────


def test_merge_scores_attaches_both_players_scores(merged):
    assert list(merged["score_winner"]) == [1.0, 1.0]
    assert list(merged["score_loser"]) == [0.0, -1.0]


def test_merge_scores_keeps_one_row_per_match(matches, merged):
    assert len(merged) == len(matches)
    assert list(merged["match_id"]) == ["1", "2"]


# ── aggregate_robustness ─────────────────────────────────────────────────────


def test_aggregate_sums_the_contributions_for_each_player(contributions_df):
    agg = aggregate_robustness(contributions_df, "winner", "winner_robustness")
    by_id = dict(zip(agg["winner"], agg["winner_robustness"]))
    np.testing.assert_approx_equal(by_id["a"], EXPECTED_ROBUSTNESS["a"])


def test_aggregate_returns_the_id_column_alongside_the_total(contributions_df):
    agg = aggregate_robustness(contributions_df, "loser", "loser_robustness")
    assert list(agg.columns) == ["loser", "loser_robustness"]


def test_aggregate_omits_players_absent_from_that_side(contributions_df):
    """`b` and `c` never win, so they get no row in the winner aggregate."""
    agg = aggregate_robustness(contributions_df, "winner", "winner_robustness")
    assert list(agg["winner"]) == ["a"]


# ── calculate_total_robustness ───────────────────────────────────────────────


def test_total_is_the_sum_of_the_two_sides(scores, winner_agg, loser_agg):
    total = calculate_total_robustness(scores, winner_agg, loser_agg)
    np.testing.assert_approx_equal(total.iloc[0], EXPECTED_ROBUSTNESS["a"])


def test_a_player_absent_from_one_side_keeps_its_other_side(
    scores, winner_agg, loser_agg
):
    """`b` never wins, so its winner robustness fills with 0 rather than NaN."""
    total = calculate_total_robustness(scores, winner_agg, loser_agg)
    np.testing.assert_approx_equal(total.iloc[1], EXPECTED_ROBUSTNESS["b"])


def test_a_player_with_no_matches_gets_zero(scores, winner_agg, loser_agg):
    bystander = pd.DataFrame({"id": ["d"], "score": [0.5]})
    total = calculate_total_robustness(
        pd.concat([scores, bystander], ignore_index=True), winner_agg, loser_agg
    )
    assert total.iloc[3] == 0.0


def test_total_follows_scores_df_row_order(scores, winner_agg, loser_agg):
    """calculate_robustness assigns this Series positionally, so order matters."""
    reversed_scores = scores.iloc[::-1].reset_index(drop=True)
    total = calculate_total_robustness(reversed_scores, winner_agg, loser_agg)
    # `a` is last now, so its total moves with it.
    np.testing.assert_approx_equal(total.iloc[2], EXPECTED_ROBUSTNESS["a"])


# ── calculate_robustness ─────────────────────────────────────────────────────


def assert_robustness_is_expected(result):
    for id_, expected in EXPECTED_ROBUSTNESS.items():
        actual = result.loc[result["id"] == id_, "robustness"].iloc[0]
        np.testing.assert_approx_equal(actual, expected)


def test_adds_a_robustness_column(matches, scores):
    result = calculate_robustness(matches, scores, lmda=1.0)
    assert "robustness" in result.columns


def test_each_player_gets_the_sum_of_its_own_matches(matches, scores):
    assert_robustness_is_expected(calculate_robustness(matches, scores, lmda=1.0))


def test_does_not_mutate_matches_df(matches, scores):
    before = matches.copy()
    calculate_robustness(matches, scores, lmda=1.0)
    pd.testing.assert_frame_equal(matches, before)


def test_mutates_and_returns_scores_df(matches, scores):
    """The robustness column lands on the caller's frame, not on a copy."""
    result = calculate_robustness(matches, scores, lmda=1.0)
    assert result is scores


def test_ignores_the_scores_df_index(matches, scores):
    """A non-default index must not misalign the assignment into all-NaN."""
    scores.index = [10, 11, 12]
    result = calculate_robustness(matches, scores, lmda=1.0)
    assert result["robustness"].notna().all()
    assert_robustness_is_expected(result)


def test_lambda_defaults_to_one(matches, scores):
    default = calculate_robustness(matches, scores.copy())
    explicit = calculate_robustness(matches, scores.copy(), lmda=1.0)
    pd.testing.assert_series_equal(default["robustness"], explicit["robustness"])


def test_match_row_order_does_not_change_the_result(matches, scores):
    forward = calculate_robustness(matches, scores.copy(), lmda=1.0)
    backward = calculate_robustness(matches.iloc[::-1], scores.copy(), lmda=1.0)
    pd.testing.assert_series_equal(forward["robustness"], backward["robustness"])
