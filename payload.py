import json
import math
from collections.abc import Mapping

from score.consts import RESULT_COLUMN

RESULTS_FIELD = "results"
SD_FIELD = "sd"
UNIT_WIN_PROB_FIELD = "unit_win_prob"

SD_ENV_VAR = "SD"
UNIT_WIN_PROB_ENV_VAR = "UNIT_WIN_PROB"

# score() needs all three: winner/loser to build the fit, match_id to attribute
# robustness contributions.
REQUIRED_RESULT_FIELDS = (
    RESULT_COLUMN.WINNER,
    RESULT_COLUMN.LOSER,
    RESULT_COLUMN.MATCH_ID,
)


class InvalidPayload(ValueError):
    """Input the caller could fix by sending a different payload."""


def parse_body(event) -> dict:
    """Unwrap the request body from a Lambda event.  There are three cases we
    accept:
    - The event *is* the body
    - The event is a dict with a "body" value...
      - ... which is a dict
      - ... which is a json string.
    """
    body = event.get("body", event) if isinstance(event, dict) else event
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError as e:
            raise InvalidPayload(f"body is not valid JSON: {e}") from e
    if not isinstance(body, dict):
        raise InvalidPayload("payload must be a JSON object")
    return body


def get_results(body: dict) -> list:
    """Extract the 'results' part of the body. It should look like this:
    ```
    {
        // other fields
        "results": [
            {"winner": "aaa", "loser": "bbb", "match_id": "37", // ...other fields },
            // ...
        ]
    }
    ```
    """
    if RESULTS_FIELD not in body:
        raise InvalidPayload(f"payload must include '{RESULTS_FIELD}' array")
    results = body[RESULTS_FIELD]
    if not isinstance(results, list):
        raise InvalidPayload(f"'{RESULTS_FIELD}' must be an array")
    if not results:
        raise InvalidPayload(f"'{RESULTS_FIELD}' must not be empty")

    for i, row in enumerate(results):
        if not isinstance(row, dict):
            raise InvalidPayload(f"{RESULTS_FIELD}[{i}] must be an object")
        for field in REQUIRED_RESULT_FIELDS:
            if row.get(field) is None:
                raise InvalidPayload(f"{RESULTS_FIELD}[{i}] is missing '{field}'")
        if row[RESULT_COLUMN.WINNER] == row[RESULT_COLUMN.LOSER]:
            raise InvalidPayload(
                f"{RESULTS_FIELD}[{i}] has the same album as winner and loser"
            )
    return results


def _get_number(body: dict, field: str, env_var: str, env: Mapping[str, str]) -> float:
    """Get a number from the body, falling back to `env` if not present."""
    if field in body:
        raw, source = body[field], f"payload field '{field}'"
    else:
        raw, source = env.get(env_var), f"{env_var} env var"
        if raw is None:
            raise InvalidPayload(
                f"missing required parameter '{field}' "
                f"(payload field or {env_var} env var)"
            )
    try:
        value = float(raw)
    except (TypeError, ValueError) as e:
        raise InvalidPayload(f"{source} must be a number, got {raw!r}") from e
    if not math.isfinite(value):
        raise InvalidPayload(f"{source} must be finite, got {value}")
    return value


def get_params(body: dict, env: Mapping[str, str]) -> tuple[float, float]:
    """Return (sd, unit_win_prob), from the payload or falling back to `env`.

    The handler will pass `os.environ` for `env`.
    """
    sd = _get_number(body, SD_FIELD, SD_ENV_VAR, env)
    if sd <= 0:
        raise InvalidPayload(f"'{SD_FIELD}' must be greater than 0, got {sd}")

    unit_win_prob = _get_number(body, UNIT_WIN_PROB_FIELD, UNIT_WIN_PROB_ENV_VAR, env)
    if not 0.5 < unit_win_prob < 1:
        raise InvalidPayload(
            f"'{UNIT_WIN_PROB_FIELD}' must be strictly between 0.5 and 1, "
            f"got {unit_win_prob}"
        )
    return sd, unit_win_prob
