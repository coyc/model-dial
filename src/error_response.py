"""Error response builders — message extraction."""

import json


def extract_error_message(body_text: str) -> str:
    """Try to extract a human-readable error message from provider JSON response."""
    try:
        obj = json.loads(body_text)
        err = obj.get("error")
        if isinstance(err, dict):
            msg = err.get("message", "")
        elif isinstance(err, str):
            msg = err
        else:
            msg = ""
        if msg:
            return msg[:200]
    except (json.JSONDecodeError, TypeError):
        pass
    return body_text[:200]
