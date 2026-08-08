#!/usr/bin/env python3
"""
Unit tests for src/credential_manager.py (shared credential rotation logic).

Run:
    python3 -m pytest tests/test_credential_manager.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.credential_manager import ProviderCredentialManager


def _make_providers(providers: dict) -> Path:
    """Write a temporary providers.json and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="providers_test_"
    )
    json.dump(providers, tmp, indent=2)
    tmp.close()
    return Path(tmp.name)


class TestGetCredential(unittest.TestCase):
    def test_returns_current(self):
        """Returns the credential marked current."""
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
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            cred = mgr.get_credential("groq")
            self.assertEqual(cred["api_key"], "K1")
            self.assertEqual(cred["base_url"], "https://b.com/v1")
        finally:
            path.unlink()

    def test_returns_first_if_no_current(self):
        """Returns the first credential if none is marked current, and marks it."""
        providers = {
            "nvidia": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://a.com/v1", "api_key": "K0"},
                    {"base_url": "https://b.com/v1", "api_key": "K1"},
                ],
            },
        }
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            cred = mgr.get_credential("nvidia")
            self.assertEqual(cred["api_key"], "K0")
            # Verify the first credential was marked current in the file
            with open(path) as f:
                data = json.load(f)
            self.assertTrue(data["nvidia"]["credentials"][0].get("current"))
            self.assertIsNone(data["nvidia"]["credentials"][1].get("current"))
        finally:
            path.unlink()

    def test_unknown_provider(self):
        """Returns None for unknown provider."""
        path = _make_providers({"groq": {"type": "openai", "credentials": []}})
        try:
            mgr = ProviderCredentialManager(path)
            self.assertIsNone(mgr.get_credential("nonexistent"))
        finally:
            path.unlink()

    def test_empty_credentials(self):
        """Returns None when credentials list is empty."""
        providers = {"groq": {"type": "openai", "credentials": []}}
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            self.assertIsNone(mgr.get_credential("groq"))
        finally:
            path.unlink()

    def test_wraps_on_invalid_index(self):
        """If credential_index exceeds list length, resets to 0."""
        providers = {
            "nvidia": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://a.com/v1", "api_key": "K0"},
                    {"base_url": "https://b.com/v1", "api_key": "K1"},
                ],
            },
        }
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            # Manually corrupt the index
            mgr._credential_index["nvidia"] = 99
            cred = mgr.get_credential("nvidia")
            self.assertEqual(cred["api_key"], "K0")
            self.assertEqual(mgr._credential_index["nvidia"], 0)
        finally:
            path.unlink()


class TestCredentialCount(unittest.TestCase):
    def test_returns_correct_count(self):
        providers = {
            "groq": {"type": "openai", "credentials": [
                {"base_url": "https://a.com/v1", "api_key": "K0"},
                {"base_url": "https://b.com/v1", "api_key": "K1"},
                {"base_url": "https://c.com/v1", "api_key": "K2"},
            ]},
            "nvidia": {"type": "openai", "credentials": []},
        }
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            self.assertEqual(mgr.credential_count("groq"), 3)
            self.assertEqual(mgr.credential_count("nvidia"), 0)
            self.assertEqual(mgr.credential_count("nonexistent"), 0)
        finally:
            path.unlink()


class TestAdvanceCredential(unittest.TestCase):
    def test_rotates(self):
        """Advance moves to next credential and returns it."""
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
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            self.assertEqual(mgr.get_credential("groq")["api_key"], "K1")
            new_cred = mgr.advance_credential("groq")
            self.assertEqual(new_cred["api_key"], "K2")
            new_cred = mgr.advance_credential("groq")
            self.assertEqual(new_cred["api_key"], "K0")
            new_cred = mgr.advance_credential("groq")
            self.assertEqual(new_cred["api_key"], "K1")
        finally:
            path.unlink()

    def test_persists_current_flag(self):
        """Advance updates the current flag in providers.json."""
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
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            mgr.advance_credential("groq")
            with open(path) as f:
                data = json.load(f)
            creds = data["groq"]["credentials"]
            self.assertIsNone(creds[0].get("current"))
            self.assertIsNone(creds[1].get("current"))
            self.assertTrue(creds[2].get("current"))
        finally:
            path.unlink()

    def test_unknown_provider(self):
        """Advance returns None for unknown provider."""
        path = _make_providers({"groq": {"type": "openai", "credentials": []}})
        try:
            mgr = ProviderCredentialManager(path)
            self.assertIsNone(mgr.advance_credential("nonexistent"))
        finally:
            path.unlink()

    def test_wraps_around(self):
        """Wrapping around persists correctly."""
        providers = {
            "nvidia": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://a.com/v1", "api_key": "K0"},
                    {"base_url": "https://b.com/v1", "api_key": "K1"},
                    {"base_url": "https://c.com/v1", "api_key": "K2", "current": True},
                ],
            },
        }
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            new_cred = mgr.advance_credential("nvidia")
            self.assertEqual(new_cred["api_key"], "K0")
            with open(path) as f:
                data = json.load(f)
            creds = data["nvidia"]["credentials"]
            self.assertIsNone(creds[2].get("current"))
            self.assertTrue(creds[0].get("current"))
        finally:
            path.unlink()

    def test_single_cred_stays_on_same(self):
        """With 1 credential, advance always returns the same cred (wrap)."""
        providers = {
            "alibaba": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://a.com/v1", "api_key": "A", "current": True},
                ],
            },
        }
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            new_cred = mgr.advance_credential("alibaba")
            self.assertEqual(new_cred["api_key"], "A")
            self.assertEqual(mgr.credential_index["alibaba"], 0)
        finally:
            path.unlink()

    def test_empty_credentials(self):
        """Advance returns None when credentials list is empty."""
        path = _make_providers({"nvidia": {"type": "openai", "credentials": []}})
        try:
            mgr = ProviderCredentialManager(path)
            self.assertIsNone(mgr.advance_credential("nvidia"))
        finally:
            path.unlink()


class TestFullRotationCycle(unittest.TestCase):
    def test_full_cycle(self):
        """Rotate through all credentials and verify correct order."""
        providers = {
            "alibaba": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://a.com/v1", "api_key": "A"},
                    {"base_url": "https://b.com/v1", "api_key": "B"},
                    {"base_url": "https://c.com/v1", "api_key": "C"},
                ],
            },
        }
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            self.assertEqual(mgr.get_credential("alibaba")["api_key"], "A")
            self.assertEqual(mgr.advance_credential("alibaba")["api_key"], "B")
            self.assertEqual(mgr.advance_credential("alibaba")["api_key"], "C")
            self.assertEqual(mgr.advance_credential("alibaba")["api_key"], "A")
        finally:
            path.unlink()


class TestIndependentProviders(unittest.TestCase):
    def test_independent(self):
        """Credential rotation is independent per provider."""
        providers = {
            "groq": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://a.com/v1", "api_key": "G0"},
                    {"base_url": "https://b.com/v1", "api_key": "G1"},
                ],
            },
            "nvidia": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://c.com/v1", "api_key": "N0"},
                    {"base_url": "https://d.com/v1", "api_key": "N1"},
                ],
            },
        }
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            self.assertEqual(mgr.advance_credential("groq")["api_key"], "G1")
            self.assertEqual(mgr.get_credential("nvidia")["api_key"], "N0")
            self.assertEqual(mgr.advance_credential("nvidia")["api_key"], "N1")
            self.assertEqual(mgr.get_credential("groq")["api_key"], "G1")
        finally:
            path.unlink()


class TestMissingFile(unittest.TestCase):
    def test_init_missing_file(self):
        """Constructor with non-existent file creates empty manager."""
        mgr = ProviderCredentialManager(Path("/nonexistent/path.json"))
        self.assertIsNone(mgr.get_credential("any"))
        self.assertEqual(mgr.credential_count("any"), 0)
        self.assertEqual(mgr.providers_creds, {})
        self.assertEqual(mgr.credential_index, {})


class TestProvidersCredsProperty(unittest.TestCase):
    def test_exposes_full_config(self):
        """providers_creds returns the full provider config dict."""
        providers = {
            "groq": {"type": "openai", "model_filter": ":free", "credentials": [
                {"base_url": "https://a.com/v1", "api_key": "K0"},
            ]},
        }
        path = _make_providers(providers)
        try:
            mgr = ProviderCredentialManager(path)
            self.assertIn("groq", mgr.providers_creds)
            self.assertEqual(mgr.providers_creds["groq"]["type"], "openai")
            self.assertEqual(mgr.providers_creds["groq"]["model_filter"], ":free")
        finally:
            path.unlink()
