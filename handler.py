import json
import os
import time

from toa.logging import Domain, get_logger

from score.score_fn import score

logger = get_logger(name="score", domain=Domain.SCORING_PIPELINE)


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
        logger.warning("missing results field in payload")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "payload must include 'results' array"}),
        }

    try:
        sd, unit_win_prob = _get_params(body)
    except ValueError as e:
        logger.warning("missing scoring parameters", extra={"error": str(e)})
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}

    t0 = time.monotonic()
    logger.info("handler started", extra={"num_results": len(results)})
    try:
        scores = score(results, sd, unit_win_prob)
        duration_ms = round((time.monotonic() - t0) * 1000)
        logger.info(
            "scoring complete",
            extra={"num_scores": len(scores), "duration_ms": duration_ms},
        )
        return {
            "statusCode": 200,
            "body": json.dumps({"scores": scores}),
        }
    except Exception:
        logger.exception(
            "handler error",
            extra={"duration_ms": round((time.monotonic() - t0) * 1000)},
        )
        raise
