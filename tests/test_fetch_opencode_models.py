#!/usr/bin/env python3
"""
Unit tests for fetch-models.py

All network access is mocked via unittest.mock — no real HTTP requests are made.

Run:
    python3 -m unittest test_fetch_models.py -v
or:
    ./units.sh
"""

import importlib.util
import json
import sys
import unittest
from io import StringIO
from unittest import mock
from pathlib import Path
from urllib.error import HTTPError
from http.client import HTTPMessage

# Make the script importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Load the main script
_SPEC = importlib.util.spec_from_file_location(
    "fetch_models", SCRIPT_DIR / "../src/fetch-models.py"
)
if _SPEC is None:
    raise FileNotFoundError("Could not load fetch-models.py")

m = importlib.util.module_from_spec(_SPEC)
if _SPEC.loader is None:
    raise AttributeError("Loader is None for fetch-models.py")

_SPEC.loader.exec_module(m)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class FakeResponse:
    """Mimics http.client.HTTPResponse for urlopen mocking."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_response(obj) -> FakeResponse:
    return FakeResponse(json.dumps(obj).encode("utf-8"))


# ---------------------------------------------------------------------------
# fetch_openai_models (OpenAI-compatible /v1/models)
# ---------------------------------------------------------------------------
class TestFetchOpenAIModels(unittest.TestCase):
    @mock.patch.object(m.urllib.request, "urlopen")
    def test_parses_data_array(self, mock_urlopen):
        mock_urlopen.return_value = make_response({
            "data": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "llama-3.1-8b", "owned_by": "meta"},
            ]
        })
        models = m.fetch_openai_models("https://api.example.com/v1", "KEY")
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["id"], "gpt-4o")
        self.assertEqual(models[1]["owned_by"], "meta")

    @mock.patch.object(m.urllib.request, "urlopen")
    def test_empty_data(self, mock_urlopen):
        mock_urlopen.return_value = make_response({"data": []})
        self.assertEqual(m.fetch_openai_models("https://x/v1", "K"), [])

    @mock.patch.object(m.urllib.request, "urlopen")
    def test_trailing_slash_in_base_url(self, mock_urlopen):
        captured = {}

        def fake_urlopen(req, timeout=None, context=None):
            captured["url"] = req.full_url
            return make_response({"data": [{"id": "m"}]})

        mock_urlopen.side_effect = fake_urlopen
        m.fetch_openai_models("https://api.example.com/v1/", "KEY")
        self.assertTrue(captured["url"].endswith("/v1/models"))

    @mock.patch.object(m.urllib.request, "urlopen")
    def test_http_error_propagates(self, mock_urlopen):
        http_message = HTTPMessage()
        mock_urlopen.side_effect = HTTPError(
            "https://x/v1/models", 401, "Unauthorized", http_message, None
        )
        with self.assertRaises(HTTPError):
            m.fetch_openai_models("https://x/v1", "BAD")

    @mock.patch.object(m.urllib.request, "urlopen")
    def test_sends_bearer_header(self, mock_urlopen):
        captured = {}

        def fake_urlopen(req, timeout=None, context=None):
            captured["headers"] = dict(req.headers)
            return make_response({"data": []})

        mock_urlopen.side_effect = fake_urlopen
        m.fetch_openai_models("https://x/v1", "SECRET")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer SECRET")


# ---------------------------------------------------------------------------
# fetch_google_models (native Google Generative AI format)
# ---------------------------------------------------------------------------
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class TestFetchGoogleModels(unittest.TestCase):
    @mock.patch.object(m.urllib.request, "urlopen")
    def test_parses_models_array(self, mock_urlopen):
        mock_urlopen.return_value = make_response({
            "models": [
                {"name": "models/gemini-2.5-flash", "version": "001"},
                {"name": "models/gemini-1.5-pro", "version": "001"},
            ]
        })
        models = m.fetch_google_models(GOOGLE_BASE_URL, "APIKEY")
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["id"], "gemini-2.5-flash")
        self.assertEqual(models[0]["owned_by"], "google")
        self.assertEqual(models[1]["id"], "gemini-1.5-pro")

    @mock.patch.object(m.urllib.request, "urlopen")
    def test_empty_models(self, mock_urlopen):
        mock_urlopen.return_value = make_response({"models": []})
        self.assertEqual(m.fetch_google_models(GOOGLE_BASE_URL, "K"), [])

    @mock.patch.object(m.urllib.request, "urlopen")
    def test_api_key_in_url(self, mock_urlopen):
        captured = {}

        def fake_urlopen(req, timeout=None, context=None):
            captured["url"] = req.full_url
            return make_response({"models": []})

        mock_urlopen.side_effect = fake_urlopen
        m.fetch_google_models(GOOGLE_BASE_URL, "MYKEY")
        self.assertIn("key=MYKEY", captured["url"])


# ---------------------------------------------------------------------------
# main() end-to-end (fully mocked: no file read, no network)
# ---------------------------------------------------------------------------
SAMPLE_CONFIG = {
    "openrouter": {
        "type": "openai",
        "credentials": [
            {"base_url": "https://openrouter.ai/api/v1", "api_key": "or-key", "current": True},
        ],
    },
    "alibaba": {
        "type": "openai",
        "credentials": [
            {"base_url": "https://custom.alibaba/v1", "api_key": "ali-key", "current": True},
        ],
    },
    "google": {
        "type": "google",
        "credentials": [
            {"base_url": "https://generativelanguage.googleapis.com/v1beta", "api_key": "gkey", "current": True},
        ],
    },
    "nokey": {
        "type": "openai",
        "credentials": [
            {"base_url": "https://nokey.test/v1"},
        ],
    },
}


class TestMainFlow(unittest.TestCase):
    def _run_main(self, extra_args=None):
        argv = ["fetch-models.py"]
        if extra_args:
            argv.extend(extra_args)
        with mock.patch.object(sys, "argv", argv), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(SAMPLE_CONFIG))), \
             mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(m, "fetch_openai_models", return_value=[
                 {"id": "model-a", "owned_by": "provider"},
             ]) as fetch_openai_mock, \
             mock.patch.object(m, "fetch_google_models", return_value=[
                 {"id": "gemini-flash", "owned_by": "google"},
             ]) as fetch_google_mock, \
             mock.patch.object(sys, "stderr"), \
             mock.patch.object(sys, "stdout", new_callable=StringIO) as out:
            m.main()
            return json.loads(out.getvalue()), fetch_openai_mock, fetch_google_mock

    def test_nokey_provider_skipped(self):
        out, fetch_openai, fetch_google = self._run_main()
        called_providers = [c.args[0] for c in fetch_openai.call_args_list]
        self.assertNotIn("nokey", called_providers)

    def test_google_uses_fetch_google(self):
        out, fetch_openai, fetch_google = self._run_main()
        fetch_google.assert_called_once_with(GOOGLE_BASE_URL, "gkey")

    def test_openai_providers_called(self):
        out, fetch_openai, fetch_google = self._run_main()
        called_urls = [c.args[0] for c in fetch_openai.call_args_list]
        self.assertIn("https://openrouter.ai/api/v1", called_urls)
        self.assertIn("https://custom.alibaba/v1", called_urls)

    def test_output_structure(self):
        out, _, _ = self._run_main()
        self.assertIn("providers", out)
        providers = {p["provider"]: p for p in out["providers"]}
        self.assertIn("openrouter", providers)
        self.assertEqual(providers["openrouter"]["count"], 1)
        self.assertEqual(providers["openrouter"]["models"][0]["id"], "model-a")
        self.assertIn("google", providers)
        self.assertEqual(providers["google"]["models"][0]["id"], "gemini-flash")

    def test_model_filter_applied(self):
        config_with_filter = {
            "openrouter": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://openrouter.ai/api/v1", "api_key": "or-key", "current": True},
                ],
                "model_filter": ":free",
            },
        }

        with mock.patch.object(sys, "argv", ["fetch-models.py"]), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(config_with_filter))), \
             mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(m, "fetch_openai_models", return_value=[
                 {"id": "model-a:free", "owned_by": "provider"},
                 {"id": "model-b:paid", "owned_by": "provider"},
                 {"id": "model-c:free", "owned_by": "provider"},
             ]), \
             mock.patch.object(sys, "stderr"), \
             mock.patch.object(sys, "stdout", new_callable=StringIO) as out:
            m.main()
            result = json.loads(out.getvalue())

        provider = result["providers"][0]
        self.assertEqual(provider["count"], 2)
        ids = [m["id"] for m in provider["models"]]
        self.assertIn("model-a:free", ids)
        self.assertIn("model-c:free", ids)
        self.assertNotIn("model-b:paid", ids)

    def test_http_error_captured(self):
        http_message = HTTPMessage()
        error = HTTPError("https://x/v1/models", 500, "Server Error", http_message, None)

        with mock.patch.object(sys, "argv", ["fetch-models.py"]), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(SAMPLE_CONFIG))), \
             mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(m, "fetch_openai_models", side_effect=error), \
             mock.patch.object(m, "fetch_google_models", side_effect=error), \
             mock.patch.object(sys, "stderr"), \
             mock.patch.object(sys, "stdout", new_callable=StringIO) as out:
            m.main()
            result = json.loads(out.getvalue())
            self.assertIn("errors", result)
            self.assertTrue(len(result["errors"]) > 0)


# ---------------------------------------------------------------------------
# --providers argument
# ---------------------------------------------------------------------------
class TestProvidersFilter(unittest.TestCase):
    def _run_main(self, providers_value=None):
        argv = ["fetch-models.py"]
        if providers_value is not None:
            argv.extend(["--providers", providers_value])
        with mock.patch.object(sys, "argv", argv), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(SAMPLE_CONFIG))), \
             mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(m, "fetch_openai_models", return_value=[
                 {"id": "model-a", "owned_by": "provider"},
             ]) as fetch_openai_mock, \
             mock.patch.object(m, "fetch_google_models", return_value=[
                 {"id": "gemini-flash", "owned_by": "google"},
             ]) as fetch_google_mock, \
             mock.patch.object(sys, "stderr"), \
             mock.patch.object(sys, "stdout", new_callable=StringIO) as out:
            m.main()
            return json.loads(out.getvalue()), fetch_openai_mock, fetch_google_mock

    def test_single_provider(self):
        """--providers openrouter fetches only openrouter"""
        out, fetch_openai, fetch_google = self._run_main("openrouter")
        called_urls = [c.args[0] for c in fetch_openai.call_args_list]
        self.assertEqual(len(called_urls), 1)
        self.assertIn("openrouter.ai", called_urls[0])
        fetch_google.assert_not_called()

    def test_multiple_providers(self):
        """--providers openrouter,alibaba fetches both"""
        out, fetch_openai, fetch_google = self._run_main("openrouter,alibaba")
        called_urls = [c.args[0] for c in fetch_openai.call_args_list]
        self.assertEqual(len(called_urls), 2)
        self.assertIn("openrouter.ai", called_urls[0])
        self.assertIn("alibaba", called_urls[1])
        fetch_google.assert_not_called()

    def test_google_provider_only(self):
        """--providers google fetches only google"""
        out, fetch_openai, fetch_google = self._run_main("google")
        fetch_google.assert_called_once()
        fetch_openai.assert_not_called()

    def test_output_filtered(self):
        """Output only contains requested providers"""
        out, _, _ = self._run_main("openrouter")
        provider_ids = [p["provider"] for p in out["providers"]]
        self.assertEqual(provider_ids, ["openrouter"])

    def test_nonexistent_provider_exits(self):
        """--providers with unknown provider exits with error"""
        with mock.patch.object(sys, "argv", ["fetch-models.py", "--providers", "nonexistent"]), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(SAMPLE_CONFIG))), \
             mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(sys, "stderr"), \
             self.assertRaises(SystemExit) as ctx:
            m.main()
        self.assertEqual(ctx.exception.code, 1)

    def test_empty_string_provider_exits(self):
        """--providers with empty string exits with error (no matching providers)"""
        with mock.patch.object(sys, "argv", ["fetch-models.py", "--providers", ""]), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(SAMPLE_CONFIG))), \
             mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(sys, "stderr"), \
             self.assertRaises(SystemExit) as ctx:
            m.main()
        self.assertEqual(ctx.exception.code, 1)

    def test_partial_match_filters_correctly(self):
        """--providers with partial names filters exactly"""
        out, fetch_openai, fetch_google = self._run_main("openrouter,google")
        called_urls = [c.args[0] for c in fetch_openai.call_args_list]
        self.assertEqual(len(called_urls), 1)
        self.assertIn("openrouter.ai", called_urls[0])
        fetch_google.assert_called_once()

    def test_no_providers_flag_fetches_all(self):
        """Without --providers, all configured providers are fetched"""
        out, fetch_openai, fetch_google = self._run_main()
        called_urls = [c.args[0] for c in fetch_openai.call_args_list]
        self.assertEqual(len(called_urls), 2)  # openrouter + alibaba
        fetch_google.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
