import pytest

from score.score_fn import score

RESULTS = [
    {"winner": "a", "loser": "b", "match_id": "1"},
    {"winner": "a", "loser": "b", "match_id": "2"},
    {"winner": "a", "loser": "c", "match_id": "3"},
    {"winner": "b", "loser": "c", "match_id": "4"},
]
SD = 0.622
UWP = 0.9
GOLDEN = {"a": 0.484, "b": -0.089, "c": -0.395}


def test_returns_one_record_per_album():
    result = score(RESULTS, SD, UWP)
    assert len(result) == 3
    assert {r["id"] for r in result} == {"a", "b", "c"}


def test_scores_sum_to_zero():
    result = score(RESULTS, SD, UWP)
    assert abs(sum(r["score"] for r in result)) < 0.01


def test_dominant_player_scores_highest():
    # "a" wins all matches, so a > b > c
    result = score(RESULTS, SD, UWP)
    by_id = {r["id"]: r["score"] for r in result}
    assert by_id["a"] > by_id["b"] > by_id["c"]


def test_robustness_is_non_negative():
    result = score(RESULTS, SD, UWP)
    assert all(r["robustness"] >= 0 for r in result)


def test_robustness_is_rounded_to_three_decimals():
    result = score(RESULTS, SD, UWP)
    assert all(r["robustness"] == round(r["robustness"], 3) for r in result)


def test_robustness_does_not_depend_on_row_order():
    """Rounding hides the last-bit noise from summing contributions in row order."""
    shuffled = list(reversed(RESULTS))
    by_id = {r["id"]: r["robustness"] for r in score(RESULTS, SD, UWP)}
    shuffled_by_id = {r["id"]: r["robustness"] for r in score(shuffled, SD, UWP)}
    assert by_id == shuffled_by_id


def test_scores_are_stable():
    """Golden values, so a change to the fit has something to fail against."""
    result = score(RESULTS, SD, UWP)
    actual = {r["id"]: round(r["score"], 3) for r in result}
    assert actual == GOLDEN


def test_robustness_is_stable():
    """Golden values, so a change to the robustness math has something to fail on."""
    result = score(RESULTS, SD, UWP)
    actual = {r["id"]: r["robustness"] for r in result}
    assert actual == {"a": 1.82, "b": 2.273, "c": 1.337}


# ── initial strengths ─────────────────────────────────────────────────────────


def scores_by_id(initial_strengths):
    return {r["id"]: r["score"] for r in score(RESULTS, SD, UWP, initial_strengths)}


def assert_golden(initial_strengths):
    """The seed is a warm start: it must not move where the fit lands.

    Compared with a tolerance rather than exactly, because a different start
    can shift the last rounded digit.
    """
    assert scores_by_id(initial_strengths) == pytest.approx(GOLDEN, abs=1e-3)


@pytest.mark.parametrize("seed", [None, {}], ids=["none", "empty"])
def test_no_seed_matches_the_golden_scores(seed):
    assert_golden(seed)


def test_seeding_with_the_answer_reproduces_it():
    assert_golden(dict(GOLDEN))


def test_a_bad_seed_still_converges():
    """Reversed order, and far from the optimum."""
    assert_golden({"a": -5.0, "b": 0.0, "c": 5.0})


def test_a_partial_seed_converges():
    """Only "a" is listed, so "b" and "c" start at 0."""
    assert_golden({"a": 0.484})


def test_extraneous_ids_are_ignored():
    """Albums that played no matches have no index in the fit."""
    assert_golden({**GOLDEN, "zzz": 5.0, "yyy": -5.0})


def test_seed_does_not_add_records():
    result = score(RESULTS, SD, UWP, {"zzz": 5.0})
    assert {r["id"] for r in result} == {"a", "b", "c"}
