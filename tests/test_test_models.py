#!/usr/bin/env python3
"""
Unit tests for test-models.py

All network access is mocked via unittest.mock — no real HTTP requests are made.

Run:
    python3 -m pytest tests/test_test_models.py -v
or:
    python3 -m unittest tests.test_test_models -v
"""

import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

# ---------------------------------------------------------------------------
# Load the script under test (hyphenated filename → importlib)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

_SPEC = importlib.util.spec_from_file_location("test_models", SCRIPT_DIR / "src/test-models.py")
if _SPEC is None:
    raise FileNotFoundError("Could not load test-models.py")

m = importlib.util.module_from_spec(_SPEC)
if _SPEC.loader is None:
    raise AttributeError("Loader is None for test-models.py")

_SPEC.loader.exec_module(m)


# ---------------------------------------------------------------------------
# Helpers: fake aiohttp objects
# ---------------------------------------------------------------------------
class FakeSSEContent:
    """Async iterator over SSE lines."""

    def __init__(self, lines: list[str]):
        self._lines = [l.encode("utf-8") for l in lines]
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._idx]
        self._idx += 1
        return line


class FakeResponse:
    """Mimics aiohttp.ClientResponse for testing."""

    def __init__(self, status: int = 200, sse_lines: list[str] | None = None, body: str = ""):
        self.status = status
        self._sse_lines = sse_lines or []
        self._body = body

    @property
    def content(self):
        return FakeSSEContent(self._sse_lines)

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakePostContext:
    """Async context manager that yields a FakeResponse."""

    def __init__(self, response: FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *a):
        return False


class FakeSession:
    """Mimics aiohttp.ClientSession for testing."""

    def __init__(self, response: FakeResponse):
        self._response = response
        self._last_kwargs = {}

    def post(self, url, **kwargs):
        self._last_kwargs = {"url": url, **kwargs}
        return FakePostContext(self._response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# load_api_keys
# ---------------------------------------------------------------------------
class TestLoadApiKeys(unittest.TestCase):
    def test_loads_keys(self):
        config = {
            "groq": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://api.groq.com/openai/v1", "api_key": "gsk_123", "current": True},
                ],
            },
            "nokey": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://nokey.test/v1"},
                ],
            },
        }
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(config))), \
             mock.patch.object(Path, "exists", return_value=True):
            result = m.load_api_keys(Path("/fake/providers.json"))
        self.assertEqual(result, {"groq": "gsk_123"})

    def test_missing_file_returns_empty(self):
        with mock.patch.object(Path, "exists", return_value=False):
            result = m.load_api_keys(Path("/nonexistent/path.json"))
        self.assertEqual(result, {})

    def test_no_providers_returns_empty(self):
        with mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
             mock.patch.object(Path, "exists", return_value=True):
            result = m.load_api_keys(Path("/fake/config.json"))
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# ProviderCredentialManager (imported from shared module)
# ---------------------------------------------------------------------------
class TestProviderCredentialManager(unittest.TestCase):
    def test_imported_from_shared_module(self):
        """ProviderCredentialManager is importable from the shared module."""
        from credential_manager import ProviderCredentialManager as SharedPCM
        self.assertIs(m.ProviderCredentialManager, SharedPCM)

    def test_get_credential_returns_current(self):
        """Returns the credential marked current (smoke test via import path)."""
        providers = {
            "groq": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://a.com/v1", "api_key": "K0"},
                    {"base_url": "https://b.com/v1", "api_key": "K1", "current": True},
                    {"base_url": "https://c.com/v1", "api_key": "K2"},
                ],
            },
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="providers_test_")
        json.dump(providers, tmp, indent=2)
        tmp.close()
        path = Path(tmp.name)
        try:
            mgr = m.ProviderCredentialManager(path)
            cred = mgr.get_credential("groq")
            self.assertEqual(cred["api_key"], "K1")
        finally:
            path.unlink()

    def test_advance_credential_rotates(self):
        """Advance moves to next credential (smoke test via import path)."""
        providers = {
            "groq": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://a.com/v1", "api_key": "K0"},
                    {"base_url": "https://b.com/v1", "api_key": "K1", "current": True},
                    {"base_url": "https://c.com/v1", "api_key": "K2"},
                ],
            },
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="providers_test_")
        json.dump(providers, tmp, indent=2)
        tmp.close()
        path = Path(tmp.name)
        try:
            mgr = m.ProviderCredentialManager(path)
            self.assertEqual(mgr.advance_credential("groq")["api_key"], "K2")
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# parse_openai_sse_line
# ---------------------------------------------------------------------------
class TestParseOpenaiSseLine(unittest.TestCase):
    def test_normal_content(self):
        line = 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
        self.assertEqual(m.parse_openai_sse_line(line), ("Hello", None))

    def test_done_marker(self):
        self.assertEqual(m.parse_openai_sse_line("data: [DONE]"), (None, None))

    def test_non_data_line(self):
        self.assertEqual(m.parse_openai_sse_line(": heartbeat"), (None, None))

    def test_empty_content(self):
        line = 'data: {"choices":[{"delta":{}}]}'
        self.assertEqual(m.parse_openai_sse_line(line), (None, None))

    def test_malformed_json(self):
        self.assertEqual(m.parse_openai_sse_line("data: {bad json"), (None, None))

    def test_no_choices(self):
        line = 'data: {"model":"gpt-4o"}'
        self.assertEqual(m.parse_openai_sse_line(line), (None, None))

    def test_reasoning_content(self):
        line = 'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}'
        self.assertEqual(m.parse_openai_sse_line(line), (None, "thinking"))

    def test_reasoning_field(self):
        """Qwen3.5 uses 'reasoning' instead of 'reasoning_content'"""
        line = 'data: {"choices":[{"delta":{"reasoning":"Process"}}]}'
        self.assertEqual(m.parse_openai_sse_line(line), (None, "Process"))

    def test_content_and_reasoning(self):
        line = 'data: {"choices":[{"delta":{"content":"4","reasoning_content":"think"}}]}'
        self.assertEqual(m.parse_openai_sse_line(line), ("4", "think"))


# ---------------------------------------------------------------------------
# parse_google_sse_line
# ---------------------------------------------------------------------------
class TestParseGoogleSseLine(unittest.TestCase):
    def test_normal_content(self):
        line = 'data: {"candidates":[{"content":{"parts":[{"text":"42"}]}}]}'
        self.assertEqual(m.parse_google_sse_line(line), ("42", None))

    def test_non_data_line(self):
        self.assertEqual(m.parse_google_sse_line("event: message"), (None, None))

    def test_empty_candidates(self):
        line = 'data: {"candidates":[]}'
        self.assertEqual(m.parse_google_sse_line(line), (None, None))

    def test_malformed_json(self):
        self.assertEqual(m.parse_google_sse_line("data: not json"), (None, None))

    def test_no_parts(self):
        line = 'data: {"candidates":[{"content":{}}]}'
        self.assertEqual(m.parse_google_sse_line(line), (None, None))


# ---------------------------------------------------------------------------
# build_openai_request
# ---------------------------------------------------------------------------
class TestBuildOpenaiRequest(unittest.TestCase):
    def test_url_construction(self):
        req = m.build_openai_request("https://api.groq.com/openai/v1", "llama-3", "KEY", "hi")
        self.assertEqual(req["url"], "https://api.groq.com/openai/v1/chat/completions")

    def test_trailing_slash_stripped(self):
        req = m.build_openai_request("https://api.example.com/v1/", "m", "K", "hi")
        self.assertEqual(req["url"], "https://api.example.com/v1/chat/completions")

    def test_headers(self):
        req = m.build_openai_request("https://x/v1", "m", "SECRET", "hi")
        self.assertEqual(req["headers"]["Authorization"], "Bearer SECRET")
        self.assertEqual(req["headers"]["User-Agent"], m.USER_AGENT)

    def test_body(self):
        req = m.build_openai_request("https://x/v1", "gpt-4o", "K", "What is 2+2?")
        self.assertEqual(req["body"]["model"], "gpt-4o")
        self.assertEqual(req["body"]["messages"][0]["content"], "What is 2+2?")
        self.assertTrue(req["body"]["stream"])
        self.assertEqual(req["body"]["max_tokens"], m.DEFAULT_MAX_TOKENS)


# ---------------------------------------------------------------------------
# build_google_request
# ---------------------------------------------------------------------------
class TestBuildGoogleRequest(unittest.TestCase):
    def test_url_construction(self):
        req = m.build_google_request("https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-flash", "AIza_key", "hi")
        self.assertIn("generativelanguage.googleapis.com", req["url"])
        self.assertIn("models/gemini-2.5-flash:streamGenerateContent", req["url"])
        self.assertNotIn("key=", req["url"])  # key goes in header, not URL

    def test_api_key_in_header(self):
        req = m.build_google_request("https://generativelanguage.googleapis.com/v1beta", "m", "AQ.Ab8test", "hi")
        self.assertEqual(req["headers"]["x-goog-api-key"], "AQ.Ab8test")

    def test_body_format(self):
        req = m.build_google_request("https://generativelanguage.googleapis.com/v1beta", "m", "K", "prompt text")
        self.assertEqual(req["body"]["contents"][0]["parts"][0]["text"], "prompt text")
        self.assertEqual(req["body"]["generationConfig"]["maxOutputTokens"], m.DEFAULT_MAX_TOKENS)


# ---------------------------------------------------------------------------
# ProviderSemaphore
# ---------------------------------------------------------------------------
class TestProviderSemaphore(unittest.TestCase):
    def test_creates_semaphore_per_provider(self):
        ps = m.ProviderSemaphore(max_per_provider=2)
        sem_a = ps.get("groq")
        sem_b = ps.get("nvidia")
        self.assertIsNot(sem_a, sem_b)
        self.assertEqual(sem_a._value, 2)

    def test_same_provider_returns_same_semaphore(self):
        ps = m.ProviderSemaphore()
        self.assertIs(ps.get("groq"), ps.get("groq"))


# ---------------------------------------------------------------------------
# test_model (async, mocked aiohttp)
# ---------------------------------------------------------------------------
class TestTestModel(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_successful_openai_model(self):
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"4"}}]}',
            "data: [DONE]",
        ]
        session = FakeSession(FakeResponse(status=200, sse_lines=sse_lines))
        result = self._run(
            m.test_model(session, "groq", "llama-3", "https://x/v1", "KEY", "openai", "hi", 10)
        )
        self.assertEqual(result["test"]["status"], "ok")
        self.assertEqual(result["test"]["answer"], "4")
        self.assertIsNotNone(result["test"]["ttft_ms"])
        self.assertIsNotNone(result["test"]["total_ms"])

    def test_http_error(self):
        session = FakeSession(FakeResponse(status=401, body='{"error":"unauthorized"}'))
        result = self._run(
            m.test_model(session, "groq", "m", "https://x/v1", "KEY", "openai", "hi", 10)
        )
        self.assertEqual(result["test"]["status"], "error")
        self.assertIn("401", result["test"]["error"])

    def test_timeout_error(self):
        class TimeoutPostContext:
            async def __aenter__(self):
                raise asyncio.TimeoutError()
            async def __aexit__(self, *a):
                return False

        session = mock.MagicMock()
        session.post = mock.MagicMock(return_value=TimeoutPostContext())
        session.__aenter__ = mock.AsyncMock(return_value=session)
        session.__aexit__ = mock.AsyncMock(return_value=False)

        result = self._run(
            m.test_model(session, "groq", "m", "https://x/v1", "KEY", "openai", "hi", 1)
        )
        self.assertEqual(result["test"]["status"], "error")
        self.assertIn("Timeout", result["test"]["error"])

    def test_google_model(self):
        sse_lines = [
            'data: {"candidates":[{"content":{"parts":[{"text":"4"}]}}]}',
        ]
        session = FakeSession(FakeResponse(status=200, sse_lines=sse_lines))
        result = self._run(
            m.test_model(session, "google", "gemini-2.5-flash", "https://x", "KEY", "google", "hi", 10)
        )
        self.assertEqual(result["test"]["status"], "ok")
        self.assertEqual(result["test"]["answer"], "4")

    def test_empty_stream(self):
        session = FakeSession(FakeResponse(status=200, sse_lines=["data: [DONE]"]))
        result = self._run(
            m.test_model(session, "groq", "m", "https://x/v1", "KEY", "openai", "hi", 10)
        )
        self.assertEqual(result["test"]["status"], "ok")
        self.assertEqual(result["test"]["answer"], "")
        self.assertIsNone(result["test"]["ttft_ms"])

    def test_ttft_recorded_only_once(self):
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"H"}}]}',
            'data: {"choices":[{"delta":{"content":"i"}}]}',
            "data: [DONE]",
        ]
        session = FakeSession(FakeResponse(status=200, sse_lines=sse_lines))
        result = self._run(
            m.test_model(session, "groq", "m", "https://x/v1", "KEY", "openai", "hi", 10)
        )
        self.assertEqual(result["test"]["answer"], "Hi")
        self.assertIsNotNone(result["test"]["ttft_ms"])


# ---------------------------------------------------------------------------
# run_tests (async, mocked aiohttp)
# ---------------------------------------------------------------------------
def _checked_entry(provider: str, model_id: str) -> dict:
    """Helper to create a pre-computed checked entry."""
    return {
        "provider": provider,
        "model_id": model_id,
        "capabilities": {"tool_call": True},
        "requirements_breakdown": {"simple": True},
        "rejected": False,
    }


class TestRunTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def _mock_cred_manager(self, creds: dict[str, dict | None]):
        """Create a mock ProviderCredentialManager that returns given credentials.

        creds: {provider_id: {"base_url": ..., "api_key": ...} | None}
        """
        mgr = mock.MagicMock(spec=m.ProviderCredentialManager)

        def _get(pid):
            return creds.get(pid)

        def _count(pid):
            return 1 if creds.get(pid) else 0

        mgr.get_credential = mock.MagicMock(side_effect=_get)
        mgr.credential_count = mock.MagicMock(side_effect=_count)
        mgr.advance_credential = mock.MagicMock(return_value=None)
        return mgr

    def test_tests_all_models_in_checked_results(self):
        """All models in checked_results are tested."""
        checked = [
            _checked_entry("groq", "m1"),
            _checked_entry("nvidia", "m2"),
        ]
        cred_mgr = self._mock_cred_manager({
            "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key": "K1"},
            "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "api_key": "K2"},
        })
        tested = []
        fake_providers_cfg = {
            "groq": {"type": "openai"},
            "nvidia": {"type": "openai"},
        }

        async def fake_test_model(*a, **kw):
            tested.append(a[2])  # positional: model_id
            return {"provider": a[1],  # positional: provider_id
                    "model_id": a[2],  # positional: model_id
                    "test": {"status": "ok", "ttft_ms": 100, "total_ms": 200, "answer": "4", "error": None}}

        with mock.patch.object(m, "test_model", side_effect=fake_test_model), \
             mock.patch.object(m, "ProviderCredentialManager", return_value=cred_mgr), \
             mock.patch.object(m, "PROVIDERS_CFG", fake_providers_cfg), \
             mock.patch("aiohttp.ClientSession") as mock_session_cls, \
             mock.patch("aiohttp.TCPConnector") as mock_tcp_cls, \
             mock.patch.object(sys, "stderr"):
            mock_session = mock.AsyncMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session
            mock_tcp_cls.return_value = mock.MagicMock()

            result = self._run(
                m.run_tests(checked, "hi", 10, 5)
            )

        self.assertEqual(result["total"], 2)
        self.assertIn("m1", tested)
        self.assertIn("m2", tested)

    def test_skips_provider_without_api_key(self):
        """Provider without API key is skipped."""
        checked = [_checked_entry("nokey", "m1")]
        cred_mgr = self._mock_cred_manager({"nokey": None})

        async def noop(*a, **kw):
            return {"test": {"status": "ok"}}

        with mock.patch.object(m, "test_model", side_effect=noop), \
             mock.patch.object(m, "ProviderCredentialManager", return_value=cred_mgr), \
             mock.patch("aiohttp.ClientSession") as mock_session_cls, \
             mock.patch("aiohttp.TCPConnector") as mock_tcp_cls, \
             mock.patch.object(sys, "stderr"):
            mock_session = mock.AsyncMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session
            mock_tcp_cls.return_value = mock.MagicMock()

            result = self._run(
                m.run_tests(checked, "hi", 10, 5)
            )

        # nokey has no credential -> skipped
        self.assertEqual(result["total"], 0)

    def test_summary_computation(self):
        """Summary counts correct/incorrect per provider."""
        checked = [
            _checked_entry("groq", "m1"),
            _checked_entry("groq", "m2"),
        ]
        cred_mgr = self._mock_cred_manager({
            "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key": "K"},
        })
        call_count = 0
        fake_providers_cfg = {"groq": {"type": "openai"}}

        async def fake_test_model(*a, **kw):
            nonlocal call_count
            call_count += 1
            model_id = a[2]  # positional: session, provider_id, model_id, ...
            if model_id == "m1":
                return {"provider": "groq", "model_id": "m1", "test": {"status": "ok", "ttft_ms": 100, "total_ms": 200, "answer": "4", "correct": True, "error": None}}
            return {"provider": "groq", "model_id": "m2", "test": {"status": "error", "ttft_ms": None, "total_ms": 50, "answer": None, "correct": None, "error": "fail"}}

        with mock.patch.object(m, "test_model", side_effect=fake_test_model), \
             mock.patch.object(m, "ProviderCredentialManager", return_value=cred_mgr), \
             mock.patch.object(m, "PROVIDERS_CFG", fake_providers_cfg), \
             mock.patch("aiohttp.ClientSession") as mock_session_cls, \
             mock.patch("aiohttp.TCPConnector") as mock_tcp_cls, \
             mock.patch.object(sys, "stderr"):
            mock_session = mock.AsyncMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session
            mock_tcp_cls.return_value = mock.MagicMock()

            result = self._run(
                m.run_tests(checked, "hi", 10, 5)
            )

        # Debug: check mock was called
        self.assertGreater(call_count, 0, "test_model was never called")
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["ok"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["summary"]["groq"]["ok"], 1)
        self.assertEqual(result["summary"]["groq"]["failed"], 1)
        self.assertEqual(result["summary"]["groq"]["correct"], 1)
        self.assertEqual(result["summary"]["groq"]["avg_ttft_ms"], 100)

    def test_google_api_type_detected(self):
        """Google provider uses google API type."""
        checked = [_checked_entry("google", "gemini")]
        cred_mgr = self._mock_cred_manager({
            "google": {"base_url": "https://generativelanguage.googleapis.com/v1beta", "api_key": "GKEY"},
        })
        captured = {}

        async def fake_test_model(*a, **kw):
            captured["api_type"] = a[5]  # positional: session, provider_id, model_id, base_url, api_key, api_type, ...
            return {"provider": "google", "model_id": "gemini", "test": {"status": "ok", "ttft_ms": 50, "total_ms": 100, "answer": "4", "correct": True, "error": None}}

        with mock.patch.object(m, "test_model", side_effect=fake_test_model), \
             mock.patch.object(m, "ProviderCredentialManager", return_value=cred_mgr), \
             mock.patch("aiohttp.ClientSession") as mock_session_cls, \
             mock.patch("aiohttp.TCPConnector") as mock_tcp_cls, \
             mock.patch.object(sys, "stderr"):
            mock_session = mock.AsyncMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session
            mock_tcp_cls.return_value = mock.MagicMock()

            self._run(
                m.run_tests(checked, "hi", 10, 5)
            )

        self.assertEqual(captured["api_type"], "google")


# ---------------------------------------------------------------------------
# Tests for validate_answer
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tests for test-threshold filtering (ttft_ms / total_ms)
# ---------------------------------------------------------------------------
class TestTestThresholdFiltering(unittest.TestCase):
    """Per-category test thresholds filter requirements_breakdown after testing."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _mock_cred_manager(self, creds: dict[str, dict | None]):
        mgr = mock.MagicMock(spec=m.ProviderCredentialManager)
        mgr.get_credential = mock.MagicMock(side_effect=lambda pid: creds.get(pid))
        mgr.credential_count = mock.MagicMock(side_effect=lambda pid: 1 if creds.get(pid) else 0)
        mgr.advance_credential = mock.MagicMock(return_value=None)
        return mgr

    def _run_with_reqs(self, checked, reqs, fake_test_model):
        cred_mgr = self._mock_cred_manager({
            "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key": "K"},
        })
        fake_providers_cfg = {"groq": {"type": "openai"}}
        with mock.patch.object(m, "REQUIREMENTS", reqs), \
             mock.patch.object(m, "test_model", side_effect=fake_test_model), \
             mock.patch.object(m, "ProviderCredentialManager", return_value=cred_mgr), \
             mock.patch.object(m, "PROVIDERS_CFG", fake_providers_cfg), \
             mock.patch("aiohttp.ClientSession") as mock_session_cls, \
             mock.patch("aiohttp.TCPConnector") as mock_tcp_cls, \
             mock.patch.object(sys, "stderr"):
            mock_session = mock.AsyncMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session
            mock_tcp_cls.return_value = mock.MagicMock()
            return self._run(m.run_tests(checked, "hi", 10, 5))

    def test_ttft_above_threshold_filters_category(self):
        """Model with ttft_ms > threshold is filtered from that category."""
        checked = [_checked_entry("groq", "fast-model")]
        checked[0]["requirements_breakdown"] = {"simple": True, "coder": True}
        reqs = {
            "simple": {"test": {"ttft_ms": 500, "total_ms": None}},
            "coder": {"test": {"ttft_ms": None, "total_ms": None}},
        }

        async def fake_test_model(*a, **kw):
            return {"provider": "groq", "model_id": "fast-model",
                    "test": {"status": "ok", "ttft_ms": 800, "total_ms": 200, "answer": "4", "correct": True, "error": None}}

        result = self._run_with_reqs(checked, reqs, fake_test_model)
        r = result["results"][0]
        self.assertFalse(r["requirements_breakdown"]["simple"])
        self.assertTrue(r["requirements_breakdown"]["coder"])

    def test_total_ms_above_threshold_filters_category(self):
        """Model with total_ms > threshold is filtered from that category."""
        checked = [_checked_entry("groq", "slow-model")]
        checked[0]["requirements_breakdown"] = {"simple": True}
        reqs = {"simple": {"test": {"ttft_ms": None, "total_ms": 1000}}}

        async def fake_test_model(*a, **kw):
            return {"provider": "groq", "model_id": "slow-model",
                    "test": {"status": "ok", "ttft_ms": 100, "total_ms": 1500, "answer": "4", "correct": True, "error": None}}

        result = self._run_with_reqs(checked, reqs, fake_test_model)
        self.assertFalse(result["results"][0]["requirements_breakdown"]["simple"])

    def test_below_threshold_passes(self):
        """Model within thresholds keeps its breakdown entries."""
        checked = [_checked_entry("groq", "good-model")]
        checked[0]["requirements_breakdown"] = {"simple": True}
        reqs = {"simple": {"test": {"ttft_ms": 1000, "total_ms": 5000}}}

        async def fake_test_model(*a, **kw):
            return {"provider": "groq", "model_id": "good-model",
                    "test": {"status": "ok", "ttft_ms": 200, "total_ms": 800, "answer": "4", "correct": True, "error": None}}

        result = self._run_with_reqs(checked, reqs, fake_test_model)
        self.assertTrue(result["results"][0]["requirements_breakdown"]["simple"])

    def test_null_thresholds_no_filtering(self):
        """null thresholds mean no filtering."""
        checked = [_checked_entry("groq", "m1")]
        checked[0]["requirements_breakdown"] = {"simple": True}
        reqs = {"simple": {"test": {"ttft_ms": None, "total_ms": None}}}

        async def fake_test_model(*a, **kw):
            return {"provider": "groq", "model_id": "m1",
                    "test": {"status": "ok", "ttft_ms": 99999, "total_ms": 99999, "answer": "4", "correct": True, "error": None}}

        result = self._run_with_reqs(checked, reqs, fake_test_model)
        self.assertTrue(result["results"][0]["requirements_breakdown"]["simple"])

    def test_absent_test_section_no_filtering(self):
        """Missing 'test' section in requirements means no filtering."""
        checked = [_checked_entry("groq", "m1")]
        checked[0]["requirements_breakdown"] = {"simple": True}
        reqs = {"simple": {"models_catalog": {"tool_call": True}}}

        async def fake_test_model(*a, **kw):
            return {"provider": "groq", "model_id": "m1",
                    "test": {"status": "ok", "ttft_ms": 99999, "total_ms": 99999, "answer": "4", "correct": True, "error": None}}

        result = self._run_with_reqs(checked, reqs, fake_test_model)
        self.assertTrue(result["results"][0]["requirements_breakdown"]["simple"])

    def test_error_models_not_affected(self):
        """Models with status != 'ok' are not checked against thresholds."""
        checked = [_checked_entry("groq", "err-model")]
        checked[0]["requirements_breakdown"] = {"simple": True}
        reqs = {"simple": {"test": {"ttft_ms": 100, "total_ms": 100}}}

        async def fake_test_model(*a, **kw):
            return {"provider": "groq", "model_id": "err-model",
                    "test": {"status": "error", "ttft_ms": None, "total_ms": None, "answer": None, "correct": None, "error": "timeout"}}

        result = self._run_with_reqs(checked, reqs, fake_test_model)
        self.assertTrue(result["results"][0]["requirements_breakdown"]["simple"])

    def test_ttft_checked_before_total_ms(self):
        """If ttft_ms exceeds threshold, total_ms is not checked (elif)."""
        checked = [_checked_entry("groq", "m1")]
        checked[0]["requirements_breakdown"] = {"simple": True}
        reqs = {"simple": {"test": {"ttft_ms": 500, "total_ms": 100}}}

        async def fake_test_model(*a, **kw):
            # ttft_ms=600 > 500 → filtered. total_ms=50 < 100 but irrelevant.
            return {"provider": "groq", "model_id": "m1",
                    "test": {"status": "ok", "ttft_ms": 600, "total_ms": 50, "answer": "4", "correct": True, "error": None}}

        result = self._run_with_reqs(checked, reqs, fake_test_model)
        self.assertFalse(result["results"][0]["requirements_breakdown"]["simple"])


# ---------------------------------------------------------------------------
# Tests for validate_answer
# ---------------------------------------------------------------------------
class TestValidateAnswer(unittest.TestCase):
    """Tests for the validate_answer() function (JSON format)."""

    def test_valid_json_int(self):
        self.assertTrue(m.validate_answer('{"answer": 4, "reasoning": "addition"}'))

    def test_valid_json_string(self):
        self.assertTrue(m.validate_answer('{"answer": "4", "reasoning": "math"}'))

    def test_valid_json_with_markdown_fences(self):
        self.assertTrue(m.validate_answer('```json\n{"answer": 4, "reasoning": "sum"}\n```'))

    def test_valid_json_with_plain_fences(self):
        self.assertTrue(m.validate_answer('```\n{"answer": 4, "reasoning": "ok"}\n```'))

    def test_valid_json_extra_fields(self):
        self.assertTrue(m.validate_answer('{"answer": 4, "reasoning": "two plus two", "confidence": 0.99}'))

    def test_wrong_answer(self):
        self.assertFalse(m.validate_answer('{"answer": 5, "reasoning": "wrong"}'))

    def test_missing_answer_field(self):
        self.assertFalse(m.validate_answer('{"reasoning": "no answer field"}'))

    def test_not_json(self):
        self.assertFalse(m.validate_answer("The answer is 4"))

    def test_not_json_number_only(self):
        self.assertFalse(m.validate_answer("4"))

    def test_invalid_json(self):
        self.assertFalse(m.validate_answer("{bad json}"))

    def test_json_array_not_dict(self):
        self.assertFalse(m.validate_answer('[4, "addition"]'))

    def test_empty_string(self):
        self.assertFalse(m.validate_answer(""))

    def test_none(self):
        self.assertFalse(m.validate_answer(None))

    def test_json_with_answer_null(self):
        self.assertFalse(m.validate_answer('{"answer": null, "reasoning": "none"}'))

    def test_json_empty_object(self):
        self.assertFalse(m.validate_answer("{}"))

    def test_custom_expected(self):
        self.assertTrue(m.validate_answer('{"answer": 7, "reasoning": "custom"}', expected="7"))

    def test_default_expected(self):
        self.assertEqual(m.DEFAULT_EXPECTED, "4")

    # --- thinking before JSON (MiniMax style) ---
    def test_thinking_before_json(self):
        answer = """The user wants me to calculate 2+2, which equals 4.

The language appears to be English based on the request, so I'll respond in English.
{"answer": 4, "reasoning": "Addition"}"""
        self.assertTrue(m.validate_answer(answer))

    def test_thinking_before_json_wrong_answer(self):
        answer = """Let me think...
{"answer": 5, "reasoning": "wrong"}"""
        self.assertFalse(m.validate_answer(answer))

    def test_multiple_json_objects_uses_last(self):
        answer = '{"answer": 5, "reasoning": "first"}\nsome text\n{"answer": 4, "reasoning": "last"}'
        self.assertTrue(m.validate_answer(answer))

    def test_code_block_with_thinking(self):
        answer = """I'll calculate that for you.

```json
{"answer": 4, "reasoning": "sum"}
```"""
        self.assertTrue(m.validate_answer(answer))

    # --- JSON at the beginning, extra text after (chatty model) ---
    def test_json_at_start_with_chat_tail(self):
        answer = '{"answer": 4, "reasoning": "arithmetic"}Could you clarify whether you\'d like the full reasoning?'
        self.assertTrue(m.validate_answer(answer))

    def test_json_at_start_with_chat_tail_wrong(self):
        answer = '{"answer": 5, "reasoning": "wrong"}Could you clarify?'
        self.assertFalse(m.validate_answer(answer))

    def test_json_at_start_multiline_chat(self):
        answer = '{"answer": 4, "reasoning": "calc"}\n\nLet me know if you need more details!'
        self.assertTrue(m.validate_answer(answer))

    # --- JSON in the middle of text ---
    def test_json_in_middle(self):
        answer = 'Sure, let me calculate that.\n{"answer": 4, "reasoning": "math"}\nThe answer is 4.'
        self.assertTrue(m.validate_answer(answer))

    def test_json_in_middle_wrong(self):
        answer = 'Thinking...\n{"answer": 3, "reasoning": "wrong"}\nDone.'
        self.assertFalse(m.validate_answer(answer))


if __name__ == "__main__":
    unittest.main(verbosity=2)
