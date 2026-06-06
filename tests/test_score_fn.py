from score.score_fn import score

RESULTS = [
    {"winner": "a", "loser": "b", "match_id": "1"},
    {"winner": "a", "loser": "b", "match_id": "2"},
    {"winner": "a", "loser": "c", "match_id": "3"},
    {"winner": "b", "loser": "c", "match_id": "4"},
]
SD = 0.622
UWP = 0.9


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
