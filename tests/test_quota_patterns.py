#!/usr/bin/env python3
"""Unit tests for is_quota_error() in src/error_detection.py.

Covers all _QUOTA_PATTERNS: quota exhaustion, free quota, rate limiting,
too many requests, insufficient quota, credits, tokens-per-minute,
current quota exceeded, monthly included depleted, resource exhaustion,
and rate-limited upstream errors.
"""

import pytest

from src.error_detection import is_quota_error


# ---------------------------------------------------------------------------
# Quota exhaustion (generic)
# ---------------------------------------------------------------------------
class TestQuotaExhausted:
    @pytest.mark.parametrize("msg", [
        "quota exhausted",
        "Quota exceeded",
        "quota depleted",
        "quota insufficient",
        "quota reached",
        "API quota exceeded",
        "Your quota has been exhausted",
        "Daily quota limit reached",
        "quota limit exceeded",
        "Monthly quota depleted",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Free quota exhausted
# ---------------------------------------------------------------------------
class TestFreeQuotaExhausted:
    @pytest.mark.parametrize("msg", [
        "free quota exhausted",
        "free quota has been exhausted",
        "The free quota has been exhausted. To continue accessing the model on a paid basis",
        "FREE QUOTA EXHAUSTED",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Rate limit reached / exceeded
# ---------------------------------------------------------------------------
class TestRateLimitReached:
    @pytest.mark.parametrize("msg", [
        "rate limit reached",
        "Rate limit exceeded",
        "rate limit surpassed",
        "API rate limit reached",
        "rate limit 429 too many requests",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Rate-limited (hyphenated) — from OpenRouter upstream
# ---------------------------------------------------------------------------
class TestRateLimited:
    @pytest.mark.parametrize("msg", [
        "temporarily rate-limited upstream",
        "rate-limited",
        "Rate Limited",
        "rate limited upstream",
        "google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly",
        "poolside/laguna-s-2.1:free is temporarily rate-limited upstream. Please retry shortly",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Too many requests
# ---------------------------------------------------------------------------
class TestTooManyRequests:
    @pytest.mark.parametrize("msg", [
        "too many requests",
        "Too Many Requests",
        "TOO MANY REQUESTS",
        "HTTP 429: too many requests",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Insufficient quota (underscore)
# ---------------------------------------------------------------------------
class TestInsufficientQuota:
    @pytest.mark.parametrize("msg", [
        "insufficient_quota",
        "Error: insufficient_quota",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Credits exhausted / depleted
# ---------------------------------------------------------------------------
class TestCreditsExhausted:
    @pytest.mark.parametrize("msg", [
        "credits exhausted",
        "credit depleted",
        "credits insufficient",
        "Your credits have been exhausted",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Tokens per minute
# ---------------------------------------------------------------------------
class TestTokensPerMinute:
    @pytest.mark.parametrize("msg", [
        "tokens per minute",
        "100000 tokens per minute limit",
        "Token per minute limit exceeded",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Exceeded your current quota — Google AI specific
# ---------------------------------------------------------------------------
class TestExceededCurrentQuota:
    @pytest.mark.parametrize("msg", [
        "exceeded your current quota",
        "You exceeded your current quota, please check your plan and billing details",
        "exceeded current quota",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Depleted your monthly included — OpenAI specific
# ---------------------------------------------------------------------------
class TestDepletedMonthlyIncluded:
    @pytest.mark.parametrize("msg", [
        "depleted your monthly included",
        "You have depleted your monthly included quota",
        "depleted your included",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Resource exhausted — Google Cloud / NVIDIA
# ---------------------------------------------------------------------------
class TestResourceExhausted:
    @pytest.mark.parametrize("msg", [
        "ResourceExhausted",
        "resource exhausted",
        "Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (48/48)",
        "ResourceExhausted: Worker local total request limit reached (48/48)",
        "RESOURCE EXHAUSTED",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Real-world messages from result-test.json
# ---------------------------------------------------------------------------
class TestRealWorldMessages:
    """All unique quota error messages observed in result-test.json."""

    @pytest.mark.parametrize("msg", [
        # Google AI — HTTP 429
        'HTTP 429: {\n  "error": {\n    "code": 429,\n'
        '"message": "You exceeded your current quota, please check your plan and '
        'billing details."',

        # NVIDIA — SSE stream error (code 502)
        'data: {"error":{"code":502,"message":"Upstream error from Nvidia: '
        'ResourceExhausted: Worker local total request limit reached (48/48)"}}',

        # OpenRouter — rate-limited upstream (HTTP 429)
        'HTTP 429: {"error":{"message":"Provider returned error","code":429,'
        '"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily '
        'rate-limited upstream. Please retry shortly"}}}',

        # OpenRouter — free quota exhausted (HTTP 403)
        'HTTP 403: {"error":{"message":"The free quota has been exhausted. '
        'To continue accessing the model on a paid basis, please complete your '
        'payment information"}}',

        # NVIDIA — SSE stream (ResourceExhausted, no prefix)
        'data: {"error":{"message":"ResourceExhausted: Worker local total '
        'request limit reached (48/48)","type":"internal_server_error","code":500}}',

        # OpenRouter — rate-limited upstream variant
        'HTTP 429: {"error":{"message":"Provider returned error","code":429,'
        '"metadata":{"raw":"poolside/laguna-s-2.1:free is temporarily '
        'rate-limited upstream. Please retry shortly"}}}',

        # Alibaba — access denied, model not purchased on this credential (HTTP 403)
        '{"error":{"message":"Access to model denied. Please make sure you are '
        'eligible for using the model.","id":"26a1ac50-fda2-98c1-8a90-2936d2d9f649",'
        '"type":"AccessDenied.Unpurchased","code":"AccessDenied.Unpurchased"}}',
    ])
    def test_real_world_quota_error(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Access-denied / unpurchased model (requires credential rotation)
# ---------------------------------------------------------------------------
class TestAccessDeniedUnpurchased:
    @pytest.mark.parametrize("msg", [
        "AccessDenied.Unpurchased",
        "accessdenied.unpurchased",
        '"type":"AccessDenied.Unpurchased"',
        "Access to model denied. Please make sure you are eligible for using the model.",
        "access to model denied",
    ])
    def test_matches(self, msg):
        assert is_quota_error(msg) is True


# ---------------------------------------------------------------------------
# Negative: messages that must NOT be flagged as quota errors
# ---------------------------------------------------------------------------
class TestNotQuotaErrors:
    @pytest.mark.parametrize("msg", [
        "",
        "Model loaded successfully",
        "Hello world",
        "context_length_exceeded",
        "maximum context",
        "connection timeout",
        "internal server error",
        "invalid api key",
        "model not found",
        "unsupported model",
        "HTTP 500: internal server error",
        "HTTP 400: bad request",
    ])
    def test_not_quota_error(self, msg):
        assert is_quota_error(msg) is False
