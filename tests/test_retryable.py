#!/usr/bin/env python3
"""Unit tests for is_retryable_error() in src/error_detection.py.

Covers all _RETRYABLE_STATUSES and _RETRYABLE_ERRORS patterns.
"""

import pytest

from src.error_detection import is_retryable_error


# ---------------------------------------------------------------------------
class TestIsRetryableError:
    @pytest.mark.parametrize("status", [408, 429, 402, 403, 503])
    def test_retryable_status_codes(self, status):
        assert is_retryable_error(status, "") is True

    def test_non_retryable_status(self):
        assert is_retryable_error(400, "") is False
        assert is_retryable_error(500, "") is False

    @pytest.mark.parametrize(
        "body",
        [
            "rate_limit_exceeded",
            "You have exceeded the rate limit",
            "insufficient_quota",
            "quota exceeded",
            "Too Many Requests",
            "capacity exceeded",
            "model is overloaded",
            "service unavailable",
            "extra_forbidden",
            "Extra inputs are not permitted",
        ],
    )
    def test_retryable_error_messages(self, body):
        assert is_retryable_error(200, body) is True

    def test_normal_error_not_retryable(self):
        assert is_retryable_error(200, "normal error message") is False
