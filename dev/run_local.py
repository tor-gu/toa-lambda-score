import json
import os
import sys
from pathlib import Path

from score_fn import score


def _get_params(payload: dict) -> tuple[float, float]:
    raw_sd = payload.get("sd") or os.environ.get("SD")
    raw_uwp = payload.get("unit_win_prob") or os.environ.get("UNIT_WIN_PROB")
    if raw_sd is None:
        print("error: missing 'sd' (payload field or SD env var)", file=sys.stderr)
        sys.exit(1)
    if raw_uwp is None:
        print("error: missing 'unit_win_prob' (payload field or UNIT_WIN_PROB env var)", file=sys.stderr)
        sys.exit(1)
    return float(raw_sd), float(raw_uwp)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <payload.json>", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(Path(sys.argv[1]).read_text())
    results = payload.get("results")
    if results is None:
        print("error: payload must include 'results' array", file=sys.stderr)
        sys.exit(1)

    sd, unit_win_prob = _get_params(payload)
    output = score(results, sd, unit_win_prob)
    print(json.dumps({"scores": output}, indent=2))
