#!/usr/bin/env python3
"""Unit tests for src/commands.py (switch category, rotate credentials)."""

import json
import sys
from pathlib import Path

import pytest

from src.commands import cmd_switch, cmd_reset, cmd_rotate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_path(tmp_path, monkeypatch):
    """Set _CURRENT_MODEL_PATH to temp file and return it."""
    p = tmp_path / "current-model.json"
    monkeypatch.setattr("src.commands._CURRENT_MODEL_PATH", p)
    return p


@pytest.fixture
def prov_path(tmp_path, monkeypatch):
    """Set _PROVIDERS_PATH to temp file and return it."""
    p = tmp_path / "providers.json"
    monkeypatch.setattr("src.commands._PROVIDERS_PATH", p)
    return p


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


# ===================================================================
# cmd_switch
# ===================================================================

class TestCmdSwitch:
    """Tests for cmd_switch()."""

    def test_advance_to_next_model(self, model_path):
        """Current flag moves from model 0 → model 1."""
        _write_json(model_path, {
            "simple": {
                "models": [
                    {"provider": "p1", "model_id": "m1", "current": True},
                    {"provider": "p2", "model_id": "m2"},
                    {"provider": "p3", "model_id": "m3"},
                ]
            }
        })
        cmd_switch("simple")
        data = json.loads(model_path.read_text())
        models = data["simple"]["models"]
        assert models[0].get("current") != True
        assert models[1].get("current") == True

    def test_wrap_around(self, model_path):
        """Last model wraps to first."""
        _write_json(model_path, {
            "simple": {
                "models": [
                    {"provider": "p1", "model_id": "m1"},
                    {"provider": "p2", "model_id": "m2"},
                    {"provider": "p3", "model_id": "m3", "current": True},
                ]
            }
        })
        cmd_switch("simple")
        data = json.loads(model_path.read_text())
        models = data["simple"]["models"]
        assert models[0].get("current") == True
        assert models[2].get("current") != True

    def test_no_explicit_current_starts_from_first(self, model_path):
        """When no current flag, first model is considered current → advance to second."""
        _write_json(model_path, {
            "simple": {
                "models": [
                    {"provider": "p1", "model_id": "m1"},
                    {"provider": "p2", "model_id": "m2"},
                ]
            }
        })
        cmd_switch("simple")
        data = json.loads(model_path.read_text())
        models = data["simple"]["models"]
        assert models[0].get("current") != True, "first should lose current"
        assert models[1].get("current") == True, "second should gain current"

    def test_exits_on_missing_file(self, model_path):
        """Exit with code 1 when current-model.json doesn't exist."""
        assert not model_path.exists()
        with pytest.raises(SystemExit) as exc:
            cmd_switch("simple")
        assert exc.value.code == 1

    def test_exits_on_invalid_json(self, model_path):
        """Exit with code 1 on malformed JSON."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("{bad json}")
        with pytest.raises(SystemExit) as exc:
            cmd_switch("simple")
        assert exc.value.code == 1

    def test_exits_on_missing_category(self, model_path):
        """Exit with code 1 when category doesn't exist."""
        _write_json(model_path, {"simple": {"models": []}})
        with pytest.raises(SystemExit) as exc:
            cmd_switch("coder")
        assert exc.value.code == 1

    def test_exits_on_empty_models(self, model_path):
        """Exit with code 1 when models list is empty."""
        _write_json(model_path, {"simple": {"models": []}})
        with pytest.raises(SystemExit) as exc:
            cmd_switch("simple")
        assert exc.value.code == 1


# ===================================================================
# cmd_reset
# ===================================================================

class TestCmdReset:
    """Tests for cmd_reset()."""

    def test_removes_current_model_file(self, model_path):
        """current-model.json is deleted."""
        _write_json(model_path, {
            "simple": {
                "models": [
                    {"provider": "p1", "model_id": "m1", "current": True},
                ]
            }
        })
        assert model_path.exists()
        cmd_reset()
        assert not model_path.exists()

    def test_no_error_when_file_missing(self, model_path):
        """No error if current-model.json doesn't exist."""
        assert not model_path.exists()
        cmd_reset()  # should not raise

    def test_output_message_when_file_exists(self, model_path):
        """Prints confirmation when file is removed."""
        _write_json(model_path, {"simple": {"models": []}})
        cmd_reset()
        assert not model_path.exists()


# ===================================================================
# ===================================================================
# cmd_rotate
# ===================================================================

class TestCmdRotate:
    """Tests for cmd_rotate()."""

    def test_advance_to_next_credential(self, prov_path):
        """Current flag moves from cred 0 → cred 1."""
        _write_json(prov_path, {
            "groq": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://a.com", "api_key": "key1", "current": True},
                    {"base_url": "https://a.com", "api_key": "key2"},
                    {"base_url": "https://a.com", "api_key": "key3"},
                ]
            }
        })
        cmd_rotate("groq")
        data = json.loads(prov_path.read_text())
        creds = data["groq"]["credentials"]
        assert creds[0].get("current") != True
        assert creds[1].get("current") == True

    def test_wrap_around(self, prov_path):
        """Last credential wraps to first."""
        _write_json(prov_path, {
            "nvidia": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://b.com", "api_key": "k1"},
                    {"base_url": "https://b.com", "api_key": "k2"},
                    {"base_url": "https://b.com", "api_key": "k3", "current": True},
                ]
            }
        })
        cmd_rotate("nvidia")
        data = json.loads(prov_path.read_text())
        creds = data["nvidia"]["credentials"]
        assert creds[0].get("current") == True
        assert creds[2].get("current") != True

    def test_no_explicit_current_starts_from_first(self, prov_path):
        """When no current flag, first credential is considered current → advance to second."""
        _write_json(prov_path, {
            "testprov": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://c.com", "api_key": "k1"},
                    {"base_url": "https://c.com", "api_key": "k2"},
                ]
            }
        })
        cmd_rotate("testprov")
        data = json.loads(prov_path.read_text())
        creds = data["testprov"]["credentials"]
        assert creds[0].get("current") != True
        assert creds[1].get("current") == True

    def test_single_credential_stays_on_same(self, prov_path):
        """Only one credential — rotates in place (1 → 1)."""
        _write_json(prov_path, {
            "groq": {
                "type": "openai",
                "credentials": [
                    {"base_url": "https://groq.com", "api_key": "only-key", "current": True},
                ]
            }
        })
        cmd_rotate("groq")
        data = json.loads(prov_path.read_text())
        creds = data["groq"]["credentials"]
        assert len(creds) == 1
        assert creds[0].get("current") == True

    def test_exits_on_missing_file(self, prov_path):
        """Exit with code 1 when providers.json doesn't exist."""
        assert not prov_path.exists()
        with pytest.raises(SystemExit) as exc:
            cmd_rotate("groq")
        assert exc.value.code == 1

    def test_exits_on_invalid_json(self, prov_path):
        """Exit with code 1 on malformed JSON."""
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        prov_path.write_text("{bad}")
        with pytest.raises(SystemExit) as exc:
            cmd_rotate("groq")
        assert exc.value.code == 1

    def test_exits_on_missing_provider(self, prov_path):
        """Exit with code 1 when provider not found."""
        _write_json(prov_path, {"other": {"credentials": []}})
        with pytest.raises(SystemExit) as exc:
            cmd_rotate("nonexistent")
        assert exc.value.code == 1

    def test_exits_on_empty_credentials(self, prov_path):
        """Exit with code 1 when credentials list is empty."""
        _write_json(prov_path, {"emptyprov": {"type": "openai", "credentials": []}})
        with pytest.raises(SystemExit) as exc:
            cmd_rotate("emptyprov")
        assert exc.value.code == 1
