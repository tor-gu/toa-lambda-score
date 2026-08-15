import os
import time

from toa.http import json_envelope
from toa.logging import Domain, get_logger

from payload import (
    InvalidPayload,
    get_initial_strengths,
    get_params,
    get_results,
    parse_body,
)
from score.score_fn import score

logger = get_logger(name="score", domain=Domain.SCORING_PIPELINE)


def handler(event, context):
    try:
        body = parse_body(event)
        results = get_results(body)
        initial_strengths = get_initial_strengths(body)
        sd, unit_win_prob = get_params(body, os.environ)
    except InvalidPayload as e:
        logger.warning("rejected invalid payload", extra={"error": str(e)})
        return json_envelope(400, {"error": str(e)})

    t0 = time.monotonic()
    logger.info(
        "handler started",
        extra={
            "num_results": len(results),
            "num_initial_strengths": len(initial_strengths),
        },
    )
    try:
        scores = score(results, sd, unit_win_prob, initial_strengths)
        duration_ms = round((time.monotonic() - t0) * 1000)
        logger.info(
            "scoring complete",
            extra={"num_scores": len(scores), "duration_ms": duration_ms},
        )
        return json_envelope(200, {"scores": scores})
    except Exception:
        logger.exception(
            "handler error",
            extra={"duration_ms": round((time.monotonic() - t0) * 1000)},
        )
        raise
