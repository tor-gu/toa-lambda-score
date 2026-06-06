import json
import os

from score.score_fn import score


def _get_params(body: dict) -> tuple[float, float]:
    raw_sd = body.get("sd") or os.environ.get("SD")
    raw_uwp = body.get("unit_win_prob") or os.environ.get("UNIT_WIN_PROB")
    if raw_sd is None:
        raise ValueError(
            "missing required parameter 'sd' (payload field or SD env var)"
        )
    if raw_uwp is None:
        raise ValueError(
            "missing required parameter 'unit_win_prob' "
            "(payload field or UNIT_WIN_PROB env var)"
        )
    return float(raw_sd), float(raw_uwp)


def handler(event, context):
    if "body" in event:
        body = (
            json.loads(event["body"])
            if isinstance(event["body"], str)
            else event["body"]
        )
    else:
        body = event

    results = body.get("results")
    if results is None:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "payload must include 'results' array"}),
        }

    try:
        sd, unit_win_prob = _get_params(body)
    except ValueError as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}

    return {
        "statusCode": 200,
        "body": json.dumps({"scores": score(results, sd, unit_win_prob)}),
    }
