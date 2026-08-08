import json

import pytest

from payload import InvalidPayload, get_params, get_results, parse_body

MATCH = {"winner": "a", "loser": "b", "match_id": "1"}
VALID = {"results": [MATCH], "sd": 1.5, "unit_win_prob": 0.9}
NO_ENV: dict[str, str] = {}
FULL_ENV = {"SD": "1.5", "UNIT_WIN_PROB": "0.9"}


def rejection(fn, *args):
    """Assert `fn(*args)` is rejected, and hand back the message."""
    with pytest.raises(InvalidPayload) as excinfo:
        fn(*args)
    return str(excinfo.value)


# ── parse_body ────────────────────────────────────────────────────────────────


def test_accepts_bare_event():
    assert parse_body(VALID) == VALID


def test_accepts_dict_under_body_key():
    assert parse_body({"body": VALID}) == VALID


def test_accepts_json_string_under_body_key():
    assert parse_body({"body": json.dumps(VALID)}) == VALID


def test_malformed_json_is_rejected():
    assert "not valid JSON" in rejection(parse_body, {"body": "{not json"})


@pytest.mark.parametrize(
    "body", ["[1, 2, 3]", '"a string"', "7"], ids=["array", "string", "number"]
)
def test_non_object_body_is_rejected(body):
    assert "must be a JSON object" in rejection(parse_body, {"body": body})


def test_non_dict_event_is_rejected():
    assert "must be a JSON object" in rejection(parse_body, [1, 2, 3])


# ── get_results ───────────────────────────────────────────────────────────────


def test_valid_results_pass_through():
    assert get_results(VALID) == [MATCH]


def test_missing_results_is_rejected():
    assert "must include 'results'" in rejection(get_results, {"sd": 1.5})


def test_results_not_a_list_is_rejected():
    assert "must be an array" in rejection(get_results, {"results": {"winner": "a"}})


def test_empty_results_is_rejected():
    assert "must not be empty" in rejection(get_results, {"results": []})


def test_result_element_not_an_object_is_rejected():
    assert "must be an object" in rejection(get_results, {"results": ["a beats b"]})


@pytest.mark.parametrize("field", ["winner", "loser", "match_id"])
def test_result_missing_required_field_is_rejected(field):
    row = {k: v for k, v in MATCH.items() if k != field}
    assert f"missing '{field}'" in rejection(get_results, {"results": [row]})


@pytest.mark.parametrize("field", ["winner", "loser", "match_id"])
def test_result_null_required_field_is_rejected(field):
    row = {**MATCH, field: None}
    assert f"missing '{field}'" in rejection(get_results, {"results": [row]})


def test_self_match_is_rejected():
    # score() would raise KeyError deep in the fit: the matchup dict has no
    # (i, i) entry.
    row = {**MATCH, "loser": "a"}
    assert "same album as winner and loser" in rejection(
        get_results, {"results": [row]}
    )


def test_message_names_the_offending_index():
    results = [MATCH, MATCH, {"winner": "c", "match_id": "3"}]
    assert "results[2]" in rejection(get_results, {"results": results})


# ── get_params ────────────────────────────────────────────────────────────────


def test_valid_params_pass_through():
    assert get_params(VALID, NO_ENV) == (1.5, 0.9)


def test_numeric_strings_are_accepted():
    assert get_params({"sd": "1.5", "unit_win_prob": "0.9"}, NO_ENV) == (1.5, 0.9)


@pytest.mark.parametrize("field", ["sd", "unit_win_prob"])
def test_missing_param_is_rejected(field):
    body = {k: v for k, v in VALID.items() if k != field}
    assert f"missing required parameter '{field}'" in rejection(
        get_params, body, NO_ENV
    )


@pytest.mark.parametrize("field", ["sd", "unit_win_prob"])
def test_non_numeric_param_is_rejected(field):
    assert "must be a number" in rejection(
        get_params, {**VALID, field: "quite large"}, NO_ENV
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_param_is_rejected(value):
    # sd=nan used to sail through and return HTTP 200 with nan scores.
    assert "must be finite" in rejection(get_params, {**VALID, "sd": value}, NO_ENV)


@pytest.mark.parametrize("value", [0, -1.5])
def test_non_positive_sd_is_rejected(value):
    assert "must be greater than 0" in rejection(
        get_params, {**VALID, "sd": value}, NO_ENV
    )


@pytest.mark.parametrize("value", [0, 0.4, 0.5, 1.0, 1.5])
def test_unit_win_prob_outside_range_is_rejected(value):
    # At 0.5 the scale is 0 and the win term vanishes; below it the model
    # inverts; at or above 1 the log is undefined.
    assert "strictly between 0.5 and 1" in rejection(
        get_params, {**VALID, "unit_win_prob": value}, NO_ENV
    )


# ── environment fallback ──────────────────────────────────────────────────────


def test_params_fall_back_to_env():
    assert get_params({}, FULL_ENV) == (1.5, 0.9)


def test_payload_wins_over_env():
    assert get_params({"sd": 2.0, "unit_win_prob": 0.8}, FULL_ENV) == (2.0, 0.8)


def test_each_param_falls_back_independently():
    assert get_params({"sd": 2.0}, FULL_ENV) == (2.0, 0.9)
    assert get_params({"unit_win_prob": 0.8}, FULL_ENV) == (1.5, 0.8)


def test_invalid_env_param_is_rejected():
    env = {**FULL_ENV, "SD": "not a number"}
    assert "SD env var must be a number" in rejection(get_params, {}, env)


def test_env_value_is_range_checked_too():
    env = {**FULL_ENV, "UNIT_WIN_PROB": "1.5"}
    assert "strictly between 0.5 and 1" in rejection(get_params, {}, env)


def test_explicit_zero_in_payload_does_not_fall_back_to_env():
    """`or` would have skipped the 0 and silently used the env value instead."""
    assert "'sd' must be greater than 0" in rejection(
        get_params, {**VALID, "sd": 0}, FULL_ENV
    )


def test_ignores_the_process_environment(monkeypatch):
    """`env` is the only environment this module knows about.

    Nothing here reads os.environ, so a process variable must not leak into a
    call that was given an env of its own.
    """
    monkeypatch.setenv("SD", "99.0")
    monkeypatch.setenv("UNIT_WIN_PROB", "0.99")
    assert get_params({}, FULL_ENV) == (1.5, 0.9)
    assert "missing required parameter 'sd'" in rejection(get_params, {}, NO_ENV)
