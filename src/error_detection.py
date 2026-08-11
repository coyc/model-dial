"""Error detection utilities: retryable and quota error classification."""

import re

# HTTP status codes that indicate the model is unavailable
_RETRYABLE_STATUSES = {408, 429, 402, 403, 503}

# Error substrings that indicate the model is unavailable
_RETRYABLE_ERRORS = {
    "rate_limit",
    "rate limit",
    "insufficient_quota",
    "quota",
    "limit exceeded",
    "limit reached",
    "resourceexhausted",
    "too many requests",
    "capacity",
    "overloaded",
    "unavailable",
    "degraded",
    "extra_forbidden",
    "extra inputs are not permitted",
    "maximum context length",
    "context_length_exceeded",
    "maximum context",
}

# Regex patterns for quota/rate-limit errors (subset of retryable: requires credential rotation)
_QUOTA_PATTERNS = [
    re.compile(r'quota\s*(?:.{,30})?(?:exhausted|exceeded|depleted|insufficient|reached)', re.IGNORECASE),
    re.compile(r'free\s+quota\s*(?:.{,30})?exhausted', re.IGNORECASE),
    re.compile(r'rate\s*limit\s*(?:.{,30})?(?:reached|exceeded|surpassed)', re.IGNORECASE),
    re.compile(r'rate[- ]?limited', re.IGNORECASE),
    re.compile(r'too\s*many\s*requests', re.IGNORECASE),
    re.compile(r'insufficient_quota', re.IGNORECASE),
    re.compile(r'credits?\s*(?:.{,30})?(?:exhausted|depleted|insufficient)', re.IGNORECASE),
    re.compile(r'tokens?\s*per\s*minute', re.IGNORECASE),
    re.compile(r'exceeded\s*(?:your\s+)?current\s+quota', re.IGNORECASE),
    re.compile(r'depleted\s+your\s+(?:monthly\s+)?included', re.IGNORECASE),
    re.compile(r'resource\s*exhausted', re.IGNORECASE),
    re.compile(r'accessdenied\.?\s*unpurchased', re.IGNORECASE),
    re.compile(r'access\s+to\s+model\s+denied', re.IGNORECASE),
    re.compile(r'not\s+found\s+for\s+account', re.IGNORECASE),
]

# ---------------------------------------------------------------------------


def is_retryable_error(status: int, body: str) -> bool:
    """Check if error indicates model is unavailable."""
    if status in _RETRYABLE_STATUSES:
        return True
    lower = body.lower()
    return any(err in lower for err in _RETRYABLE_ERRORS)


def is_quota_error(body: str) -> bool:
    """Check if error body indicates quota/rate-limit exhaustion.

    This is a subset of retryable errors: the provider's quota or rate limit
    has been exhausted. Requires credential rotation (not model switch).
    """
    if not body:
        return False
    return any(p.search(body) for p in _QUOTA_PATTERNS)
