"""Run the scoring logic on a payload file, without a Lambda runtime.

Validates exactly what the deployed handler validates — same `payload` module —
so a payload that works here works there. The handler itself is not imported,
because it needs the common layer (`toa.*`) and dev/requirements-dev.txt does
not install it.

    ./dev/run.sh dev/sample_payload.json
"""

import json
import os
import sys
from pathlib import Path

from payload import InvalidPayload, get_params, get_results, parse_body
from score.score_fn import score


def die(message: str):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def main(path: Path):
    try:
        raw = path.read_text()
    except OSError as e:
        die(f"cannot read {path}: {e}")

    try:
        body = parse_body(raw)
        results = get_results(body)
        sd, unit_win_prob = get_params(body, os.environ)
    except InvalidPayload as e:
        # The deployed handler answers these with a 400.
        die(str(e))

    print(json.dumps({"scores": score(results, sd, unit_win_prob)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <payload.json>", file=sys.stderr)
        sys.exit(1)
    main(Path(sys.argv[1]))
