#!/usr/bin/env python3
"""Tests for tools/generate_opencode_config.py."""

import json
import tempfile
from pathlib import Path

import pytest

from tools.generate_opencode_config import (
    GATEWAY_BASE_URL,
    _build_model_entry,
    generate_opencode_config,
    main,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def minimal_config():
    """Minimal config with one requirement."""
    return {
        "requirements": {
            "fast": {
                "models_catalog": {
                    "tool_call": True,
                    "limit": {"context": 60000, "output": 8000},
                },
            },
        },
        "gateway": {"api_key": "test-key", "port": 8765},
    }


@pytest.fixture
def full_config():
    """Full config with all capability types."""
    return {
        "requirements": {
            "fast": {
                "models_catalog": {
                    "tool_call": True,
                    "limit": {"context": 60000, "output": 8000},
                },
            },
            "coder": {
                "models_catalog": {
                    "tool_call": True,
                    "reasoning": True,
                    "limit": {"context": 200000, "output": 32000},
                },
            },
            "visual": {
                "models_catalog": {
                    "tool_call": True,
                    "reasoning": True,
                    "attachment": True,
                    "modalities": {"input": ["text", "image"]},
                    "limit": {"context": 60000, "output": 16000},
                },
            },
            "planner": {
                "models_catalog": {
                    "tool_call": True,
                    "reasoning": True,
                    "limit": {"context": 900000, "output": 120000},
                },
            },
        },
        "gateway": {"api_key": "secret", "port": 8765},
    }


# =========================================================================
# _build_model_entry — capability mapping
# =========================================================================

class TestBuildModelEntry:
    """_build_model_entry maps models_catalog fields to OpenCode model entry."""

    def test_tool_call_only(self):
        entry = _build_model_entry("fast", {"tool_call": True})
        assert entry == {"name": "Fast (auto)", "tool_call": True}

    def test_tool_call_and_reasoning(self):
        entry = _build_model_entry("coder", {
            "tool_call": True,
            "reasoning": True,
            "limit": {"context": 200000, "output": 32000},
        })
        assert entry["tool_call"] is True
        assert entry["reasoning"] is True
        assert entry["limit"] == {"context": 200000, "output": 32000}

    def test_full_visual_capabilities(self):
        entry = _build_model_entry("visual", {
            "tool_call": True,
            "reasoning": True,
            "attachment": True,
            "modalities": {"input": ["text", "image"]},
            "limit": {"context": 60000, "output": 16000},
        })
        assert entry == {
            "name": "Visual (auto)",
            "tool_call": True,
            "reasoning": True,
            "attachment": True,
            "modalities": {"input": ["text", "image"], "output": ["text"]},
            "limit": {"context": 60000, "output": 16000},
        }

    def test_false_flags_are_omitted(self):
        """Boolean fields set to False must not appear in the entry."""
        entry = _build_model_entry("fast", {
            "tool_call": True,
            "reasoning": False,
            "attachment": False,
        })
        assert "reasoning" not in entry
        assert "attachment" not in entry
        assert entry["tool_call"] is True

    def test_no_modalities_field(self):
        """When models_catalog has no modalities, entry should not either."""
        entry = _build_model_entry("fast", {"tool_call": True})
        assert "modalities" not in entry

    def test_name_capitalization(self):
        """Category name is capitalized in the entry."""
        entry = _build_model_entry("planner", {"tool_call": True})
        assert entry["name"] == "Planner (auto)"

    def test_name_with_hyphen(self):
        """Hyphenated category names capitalize only the first letter."""
        entry = _build_model_entry("qa-navigation", {"tool_call": True})
        assert entry["name"] == "Qa-navigation (auto)"


# =========================================================================
# generate_opencode_config — full config generation
# =========================================================================

class TestGenerateOpencodeConfig:
    """generate_opencode_config builds the full OpenCode provider structure."""

    def test_minimal_config_structure(self, minimal_config):
        result = generate_opencode_config(minimal_config)
        assert "provider" in result
        assert "gateway" in result["provider"]
        gateway = result["provider"]["gateway"]
        assert gateway["npm"] == "@ai-sdk/openai-compatible"
        assert gateway["name"] == "Model-Dial Gateway"
        assert gateway["options"]["baseURL"] == GATEWAY_BASE_URL
        assert gateway["options"]["apiKey"] == "test-key"
        assert "fast" in gateway["models"]

    def test_full_config_all_models_present(self, full_config):
        result = generate_opencode_config(full_config)
        models = result["provider"]["gateway"]["models"]
        assert set(models.keys()) == {"fast", "coder", "visual", "planner"}

    def test_models_sorted_alphabetically(self, full_config):
        result = generate_opencode_config(full_config)
        models = result["provider"]["gateway"]["models"]
        assert list(models.keys()) == sorted(models.keys())

    def test_visual_has_modalities_with_output_text(self, full_config):
        result = generate_opencode_config(full_config)
        visual = result["provider"]["gateway"]["models"]["visual"]
        assert visual["modalities"]["input"] == ["text", "image"]
        assert visual["modalities"]["output"] == ["text"]

    def test_empty_api_key(self):
        """Empty string api_key is passed through."""
        config = {
            "requirements": {"fast": {"models_catalog": {"tool_call": True}}},
            "gateway": {"api_key": ""},
        }
        result = generate_opencode_config(config)
        assert result["provider"]["gateway"]["options"]["apiKey"] == ""

    def test_no_gateway_section(self):
        """Missing gateway section defaults to empty api_key."""
        config = {
            "requirements": {"fast": {"models_catalog": {"tool_call": True}}},
        }
        result = generate_opencode_config(config)
        assert result["provider"]["gateway"]["options"]["apiKey"] == ""

    def test_no_requirements_section(self):
        """Missing requirements produces empty models dict."""
        config = {"gateway": {"api_key": "key"}}
        result = generate_opencode_config(config)
        assert result["provider"]["gateway"]["models"] == {}

    def test_empty_requirements(self):
        """Empty requirements dict produces empty models."""
        config = {"requirements": {}, "gateway": {"api_key": "key"}}
        result = generate_opencode_config(config)
        assert result["provider"]["gateway"]["models"] == {}


# =========================================================================
# main() — CLI interface
# =========================================================================

class TestMain:
    """main() handles file I/O and CLI args."""

    def test_stdout_output(self, minimal_config, capsys, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(minimal_config))

        exit_code = main(["-c", str(config_path)])
        assert exit_code == 0

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "provider" in result
        assert "fast" in result["provider"]["gateway"]["models"]

    def test_file_output(self, minimal_config, tmp_path, capsys):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(minimal_config))
        output_path = tmp_path / "opencode.json"

        exit_code = main(["-c", str(config_path), "-o", str(output_path)])
        assert exit_code == 0

        result = json.loads(output_path.read_text())
        assert "provider" in result

    def test_custom_indent(self, minimal_config, capsys, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(minimal_config))

        main(["-c", str(config_path), "--indent", "2"])
        captured = capsys.readouterr()
        # Check indentation: 2 spaces for first-level nesting
        assert '  "provider"' in captured.out

    def test_missing_config_file(self, tmp_path, capsys):
        exit_code = main(["-c", str(tmp_path / "nonexistent.json")])
        assert exit_code == 1

    def test_invalid_json(self, tmp_path, capsys):
        config_path = tmp_path / "bad.json"
        config_path.write_text("{not valid json")

        exit_code = main(["-c", str(config_path)])
        assert exit_code == 1

    def test_uses_default_config_path(self, capsys, monkeypatch, tmp_path):
        """When no -c is given, reads from config.json in cwd."""
        config = {
            "requirements": {"fast": {"models_catalog": {"tool_call": True}}},
            "gateway": {"api_key": "k"},
        }
        # Change to tmp_path and write config.json there
        monkeypatch.chdir(tmp_path)
        Path("config.json").write_text(json.dumps(config))

        exit_code = main([])
        assert exit_code == 0

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "fast" in result["provider"]["gateway"]["models"]
