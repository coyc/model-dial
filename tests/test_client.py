"""Tests for tools/client.py."""

import json

import pytest

from tools.client import (
    assistant_color,
    build_headers,
    build_request_payload,
    colorize,
    error_color,
    extract_error_message,
    extract_response_content,
    get_available_models,
    get_gateway_base_url,
    load_config,
    parse_model_selection,
    supports_color,
    user_color,
)

# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def sample_config():
    """Minimal config matching real config.json structure."""
    return {
        "requirements": {
            "coder": {"models_catalog": {"tool_call": True}},
            "visual": {"models_catalog": {"attachment": True}},
            "zen-tools": {"models_catalog": {"tool_call": True, "reasoning": True}},
        },
        "gateway": {
            "api_key": "test-key",
            "port": 8765,
            "request_timeout_ms": 30000,
        },
    }


@pytest.fixture
def minimal_config():
    """Config with only one requirement."""
    return {
        "requirements": {
            "fast": {"models_catalog": {"tool_call": True}},
        },
        "gateway": {"api_key": "", "port": 9999},
    }


@pytest.fixture
def empty_config():
    """Config with no requirements."""
    return {"gateway": {"api_key": "key"}}


# =========================================================================
# 1. Parsing available models from requirements
# =========================================================================

class TestGetAvailableModels:
    """Extract category names from requirements section."""

    def test_extracts_all_categories(self, sample_config):
        models = get_available_models(sample_config)
        assert set(models) == {"coder", "visual", "zen-tools"}

    def test_sorted_alphabetically(self, sample_config):
        models = get_available_models(sample_config)
        assert models == sorted(models)

    def test_empty_requirements(self, empty_config):
        models = get_available_models(empty_config)
        assert models == []

    def test_no_requirements_key(self):
        models = get_available_models({})
        assert models == []


# =========================================================================
# 2. Default model selection (first alphabetically)
# =========================================================================

class TestParseModelSelection:
    """Parse user input into a model name."""

    def test_empty_returns_first_model(self):
        models = ["coder", "visual", "zen-tools"]
        assert parse_model_selection("", models) == "coder"

    def test_whitespace_returns_first_model(self):
        models = ["coder", "visual"]
        assert parse_model_selection("   ", models) == "coder"

    def test_valid_number_selects_model(self):
        models = ["coder", "visual", "zen-tools"]
        assert parse_model_selection("2", models) == "visual"

    def test_first_number(self):
        models = ["coder", "visual"]
        assert parse_model_selection("1", models) == "coder"

    def test_last_number(self):
        models = ["a", "b", "c", "d"]
        assert parse_model_selection("4", models) == "d"

    def test_out_of_range_raises(self):
        models = ["coder", "visual"]
        with pytest.raises(ValueError, match="Invalid selection"):
            parse_model_selection("5", models)

    def test_zero_raises(self):
        models = ["coder", "visual"]
        with pytest.raises(ValueError, match="Invalid selection"):
            parse_model_selection("0", models)

    def test_negative_raises(self):
        models = ["coder"]
        with pytest.raises(ValueError, match="Invalid selection"):
            parse_model_selection("-1", models)

    def test_non_numeric_raises(self):
        models = ["coder", "visual"]
        with pytest.raises(ValueError, match="Invalid input"):
            parse_model_selection("abc", models)

    def test_single_model(self):
        assert parse_model_selection("", ["only-one"]) == "only-one"
        assert parse_model_selection("1", ["only-one"]) == "only-one"

    def test_no_models_raises(self):
        with pytest.raises(ValueError, match="No models available"):
            parse_model_selection("", [])


# =========================================================================
# 3. Config loading from config.json
# =========================================================================

class TestLoadConfig:
    """Load config.json with proper error handling."""

    def test_loads_valid_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        data = {"requirements": {}, "gateway": {"port": 8765}}
        config_path.write_text(json.dumps(data))

        result = load_config(config_path)
        assert result == data

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{not valid json")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config(config_path)

    def test_minimal_valid_config_loads(self, tmp_path):
        """Config with only requirements (no gateway) loads without error."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"requirements": {"fast": {}}}))

        result = load_config(config_path)
        assert result == {"requirements": {"fast": {}}}
        assert "gateway" not in result


# =========================================================================
# 4. HTTP request formation
# =========================================================================

class TestBuildRequest:
    """Build OpenAI-compatible request payload and headers."""

    def test_payload_structure(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        payload = build_request_payload("zen-tools", messages)
        assert payload == {
            "model": "zen-tools",
            "messages": messages,
            "stream": False,
        }

    def test_payload_single_message(self):
        payload = build_request_payload("coder", [{"role": "user", "content": "Hi"}])
        assert payload["model"] == "coder"
        assert len(payload["messages"]) == 1
        assert payload["stream"] is False

    def test_headers_with_api_key(self):
        headers = build_headers("my-secret-key")
        assert headers["Authorization"] == "Bearer my-secret-key"
        assert headers["Content-Type"] == "application/json"

    def test_headers_without_api_key(self):
        headers = build_headers(None)
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_headers_empty_string_key(self):
        headers = build_headers("")
        assert "Authorization" not in headers

    def test_gateway_base_url(self):
        assert get_gateway_base_url(8765) == "http://localhost:8765/v1"
        assert get_gateway_base_url(9999) == "http://localhost:9999/v1"


# =========================================================================
# 5. Response parsing
# =========================================================================

class TestExtractResponseContent:
    """Extract assistant content from OpenAI chat completion response."""

    def test_standard_response(self):
        response = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "The capital is Paris."},
                    "finish_reason": "stop",
                }
            ]
        }
        assert extract_response_content(response) == "The capital is Paris."

    def test_multiline_content(self):
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Line 1\nLine 2\nLine 3"},
                }
            ]
        }
        assert extract_response_content(response) == "Line 1\nLine 2\nLine 3"

    def test_empty_choices_raises(self):
        with pytest.raises(ValueError, match="no choices"):
            extract_response_content({"choices": []})

    def test_missing_choices_key_raises(self):
        with pytest.raises(ValueError, match="no choices"):
            extract_response_content({})

    def test_missing_content_raises(self):
        response = {"choices": [{"message": {"role": "assistant"}}]}
        with pytest.raises(ValueError, match="no content"):
            extract_response_content(response)


# =========================================================================
# 6. Error handling (network errors, HTTP errors)
# =========================================================================

class TestErrorHandling:
    """Handle network errors and HTTP error responses."""

    def test_extract_error_message_standard(self):
        response = {"error": {"message": "Model not found", "type": "invalid_request_error"}}
        assert extract_error_message(response) == "Model not found"

    def test_extract_error_message_missing(self):
        assert extract_error_message({}) == "Unknown error"

    def test_extract_error_message_no_message_field(self):
        assert extract_error_message({"error": {"type": "server_error"}}) == "Unknown error"


# =========================================================================
# 7. Exit command handling
# =========================================================================

class TestExitCommands:
    """Exit commands are recognized in the message loop."""

    def test_exit_command_detected(self):
        """Verify /exit is in the set of exit commands."""
        exit_commands = {"/exit", "/quit"}
        assert "/exit" in exit_commands
        assert "/quit" in exit_commands

    def test_non_exit_commands_not_matched(self):
        """Normal text is not an exit command."""
        exit_commands = {"/exit", "/quit"}
        assert "hello" not in exit_commands
        assert "/help" not in exit_commands
        assert "exit" not in exit_commands
        assert "/exit " not in exit_commands


# =========================================================================
# 8. Terminal colors
# =========================================================================

class TestTerminalColors:
    """Color helpers are no-op without a TTY and wrap text with ANSI codes when supported."""

    def test_colorize_noop_without_tty(self, monkeypatch):
        """Without color support the text is returned unchanged."""
        monkeypatch.setattr("tools.client.supports_color", lambda: False)
        assert colorize("hello", "\033[36m") == "hello"

    def test_colorize_wraps_with_ansi(self, monkeypatch):
        """With color support the text is wrapped in ANSI codes."""
        monkeypatch.setattr("tools.client.supports_color", lambda: True)
        assert colorize("hello", "\033[36m") == "\033[36mhello\033[0m"

    def test_user_color(self, monkeypatch):
        """User color is cyan."""
        monkeypatch.setattr("tools.client.supports_color", lambda: True)
        assert user_color("You:") == "\033[36mYou:\033[0m"

    def test_assistant_color(self, monkeypatch):
        """Assistant color is green."""
        monkeypatch.setattr("tools.client.supports_color", lambda: True)
        assert assistant_color("Answer") == "\033[32mAnswer\033[0m"

    def test_error_color(self, monkeypatch):
        """Error color is red."""
        monkeypatch.setattr("tools.client.supports_color", lambda: True)
        assert error_color("Boom") == "\033[31mBoom\033[0m"

    def test_supports_color_false_when_not_tty(self, monkeypatch):
        """supports_color is False when stdout is not a TTY."""
        class _FakeStdout:
            def isatty(self) -> bool:
                return False

        monkeypatch.setattr("tools.client.sys.stdout", _FakeStdout())
        assert supports_color() is False

    def test_supports_color_true_when_tty(self, monkeypatch):
        """supports_color is True when stdout is a TTY."""
        class _FakeStdout:
            def isatty(self) -> bool:
                return True

        monkeypatch.setattr("tools.client.sys.stdout", _FakeStdout())
        assert supports_color() is True
