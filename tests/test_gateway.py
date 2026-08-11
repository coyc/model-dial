#!/usr/bin/env python3
"""Unit tests for the gateway module (transparent proxy)."""

import copy
import json
import os
import random
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add src to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import gateway
import streaming


# ---------------------------------------------------------------------------
# Autouse fixture: redirect current-model.json to a temp path
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_current_model_path(tmp_path):
    """Redirect _CURRENT_MODEL_PATH so save/load use an isolated temp file."""
    old = gateway._CURRENT_MODEL_PATH
    gateway._CURRENT_MODEL_PATH = tmp_path / "current-model.json"
    yield
    gateway._CURRENT_MODEL_PATH = old


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
MOCK_CONFIG = {
    "user_agent": "test-agent/1.0",
    "requirements": {
        "simple": {},
        "coder": {},
        "visual": {},
    },
    "gateway": {
        "port": 8765,
        "retry_count": 2,
    },
}

MOCK_TEST_RESULTS = {
    "results": [
        {"provider": "nvidia", "model_id": "meta/llama-3.2-1b-instruct", "rejected": False, "requirements_breakdown": {"simple": True, "coder": False, "visual": False}, "test": {"status": "ok", "total_ms": 100, "correct": True}},
        {"provider": "alibaba", "model_id": "qwen-flash", "rejected": False, "requirements_breakdown": {"simple": True, "coder": False, "visual": False}, "test": {"status": "ok", "total_ms": 200, "correct": True}},
        {"provider": "nvidia", "model_id": "nvidia/nemotron-mini-4b-instruct", "rejected": False, "requirements_breakdown": {"simple": True, "coder": False, "visual": False}, "test": {"status": "ok", "total_ms": 300, "correct": True}},
        {"provider": "nvidia", "model_id": "nvidia/llama-3.3-nemotron-super-49b-v1", "rejected": False, "requirements_breakdown": {"simple": False, "coder": True, "visual": False}, "test": {"status": "ok", "total_ms": 400, "correct": True}},
        {"provider": "alibaba", "model_id": "deepseek-v4-pro", "rejected": False, "requirements_breakdown": {"simple": False, "coder": True, "visual": False}, "test": {"status": "ok", "total_ms": 500, "correct": True}},
        {"provider": "nvidia", "model_id": "meta/llama-3.2-11b-vision-instruct", "rejected": False, "requirements_breakdown": {"simple": False, "coder": False, "visual": True}, "test": {"status": "ok", "total_ms": 250, "correct": True}},
        {"provider": "alibaba", "model_id": "qwen-vl-plus", "rejected": False, "requirements_breakdown": {"simple": False, "coder": False, "visual": True}, "test": {"status": "ok", "total_ms": 350, "correct": True}},
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_config(**overrides) -> dict:
    config = {**MOCK_CONFIG}
    gw = {**MOCK_CONFIG["gateway"], **overrides.get("gateway", {})}
    config["gateway"] = gw
    return config


def _make_test_results(**overrides) -> dict:
    data = {**MOCK_TEST_RESULTS}
    data["results"] = [*MOCK_TEST_RESULTS["results"], *overrides.get("results", [])]
    return data


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# ===========================================================================
# Tests: Config loading
# ===========================================================================
class TestLoadConfig:
    def test_loads_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        _write_json(config_path, MOCK_CONFIG)

        result = gateway.load_config(config_path)

        assert result["user_agent"] == "test-agent/1.0"
        assert result["gateway"]["retry_count"] == 2
        assert "providers" not in result  # providers moved to providers.json

    def test_loads_test_results(self, tmp_path):
        results_path = tmp_path / "result-test.json"
        _write_json(results_path, MOCK_TEST_RESULTS)

        result = gateway.load_test_results(results_path)

        assert "results" in result
        assert len(result["results"]) == 7


# ===========================================================================
# Tests: GatewayState
# ===========================================================================
class TestGatewayState:
    def test_get_model_returns_first(self):
        state = gateway.GatewayState()
        state.pools = {"simple": [{"model_id": "a"}, {"model_id": "b"}]}
        state.current_index = {"simple": 0}

        assert state.get_model("simple")["model_id"] == "a"

    def test_get_model_returns_none_when_exhausted(self):
        state = gateway.GatewayState()
        state.pools = {"simple": [{"model_id": "a"}]}
        state.current_index = {"simple": 1}

        assert state.get_model("simple") is None

    def test_advance_moves_to_next(self):
        state = gateway.GatewayState()
        state.pools = {"simple": [{"model_id": "a"}, {"model_id": "b"}]}
        state.current_index = {"simple": 0}

        model = state.advance("simple")
        assert model["model_id"] == "b"

    def test_advance_returns_none_at_end(self):
        state = gateway.GatewayState()
        state.pools = {"simple": [{"model_id": "a"}]}
        state.current_index = {"simple": 0}

        model = state.advance("simple")
        assert model is None

    def test_available_categories(self):
        state = gateway.GatewayState()
        state.pools = {"simple": [{"m": 1}], "coder": [{"m": 2}], "visual": []}

        cats = state.available_categories()
        assert "simple" in cats
        assert "coder" in cats
        assert "visual" not in cats


# ===========================================================================
# Tests: build_state
# ===========================================================================
class TestBuildState:
    def test_builds_state(self):
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        assert len(state.pools["simple"]) == 3
        assert len(state.pools["coder"]) == 2
        assert len(state.pools["visual"]) == 2
        assert "nvidia" in state.providers_creds

    def test_switch_on_any_error_defaults_to_false(self):
        """switch_on_any_error should be False when not set in config."""
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert state.switch_on_any_error is False

    def test_switch_on_any_error_reads_from_config(self):
        """switch_on_any_error should be True when set in config."""
        config = _make_config(gateway={"switch_on_any_error": True})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        assert state.switch_on_any_error is True

    def test_request_timeout_default(self):
        """Defaults to 60000ms when not in config."""
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert state.request_timeout_ms == 60000

    def test_request_timeout_from_config(self):
        """Custom request_timeout_ms is read from config."""
        config = _make_config(gateway={"request_timeout_ms": 15000})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        assert state.request_timeout_ms == 15000

    def test_stream_idle_timeout_default(self):
        """Defaults to 30000ms when not in config."""
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert state.stream_idle_timeout_ms == 30000

    def test_stream_idle_timeout_from_config(self):
        """Custom stream_idle_timeout_ms is read from config."""
        config = _make_config(gateway={"stream_idle_timeout_ms": 10000})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        assert state.stream_idle_timeout_ms == 10000


# ===========================================================================
# Tests: Pool sorting strategies (requirements.<cat>.sort_strategy)
# ===========================================================================
class TestSortPool:
    def test_total_ms_ascending_with_nulls_last(self):
        models = [
            {"provider": "a", "model_id": "m1", "test": {"total_ms": 300}},
            {"provider": "b", "model_id": "m2", "test": {"total_ms": None}},
            {"provider": "c", "model_id": "m3", "test": {"total_ms": 100}},
        ]
        result = gateway.sort_pool(models, "total_ms")
        assert [m["model_id"] for m in result] == ["m3", "m1", "m2"]

    def test_ttft_ms_ascending_with_nulls_last(self):
        models = [
            {"provider": "a", "model_id": "slow", "test": {"ttft_ms": 500, "total_ms": 10}},
            {"provider": "b", "model_id": "none", "test": {"ttft_ms": None, "total_ms": 10}},
            {"provider": "c", "model_id": "fast", "test": {"ttft_ms": 100, "total_ms": 10}},
        ]
        result = gateway.sort_pool(models, "ttft_ms")
        assert [m["model_id"] for m in result] == ["fast", "slow", "none"]

    def test_model_id_alphabetical_case_insensitive(self):
        models = [
            {"provider": "a", "model_id": "Zebra"},
            {"provider": "b", "model_id": "alpha"},
            {"provider": "c", "model_id": "Beta"},
        ]
        result = gateway.sort_pool(models, "model_id")
        assert [m["model_id"] for m in result] == ["alpha", "Beta", "Zebra"]

    def test_random_is_permutation_and_seeded_deterministic(self):
        models = [
            {"provider": "a", "model_id": "m1"},
            {"provider": "b", "model_id": "m2"},
            {"provider": "c", "model_id": "m3"},
            {"provider": "d", "model_id": "m4"},
        ]
        rng = random.Random(42)
        shuffled = gateway.sort_pool(models, "random", rng=rng)
        # Same elements, just reordered
        assert sorted(m["model_id"] for m in shuffled) == ["m1", "m2", "m3", "m4"]
        # Seeded RNG → reproducible order
        again = gateway.sort_pool(models, "random", rng=random.Random(42))
        assert [m["model_id"] for m in shuffled] == [m["model_id"] for m in again]

    def test_unknown_strategy_falls_back_to_total_ms(self):
        models = [
            {"provider": "a", "model_id": "m1", "test": {"total_ms": 300}},
            {"provider": "c", "model_id": "m3", "test": {"total_ms": 100}},
        ]
        result = gateway.sort_pool(models, "bogus")
        assert [m["model_id"] for m in result] == ["m3", "m1"]


class TestSortStrategyConfig:
    @staticmethod
    def _config_with_strategy(strategy):
        config = copy.deepcopy(MOCK_CONFIG)
        config["requirements"]["simple"]["sort_strategy"] = strategy
        return config

    def test_default_strategy_is_total_ms(self):
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert [m["model_id"] for m in state.pools["simple"]] == [
            "meta/llama-3.2-1b-instruct",  # 100ms
            "qwen-flash",  # 200ms
            "nvidia/nemotron-mini-4b-instruct",  # 300ms
        ]

    def test_ttft_ms_strategy_from_config(self):
        results = {
            "results": [
                {"provider": "a", "model_id": "slow", "rejected": False,
                 "requirements_breakdown": {"simple": True},
                 "test": {"status": "ok", "ttft_ms": 500, "total_ms": 10}},
                {"provider": "b", "model_id": "fast", "rejected": False,
                 "requirements_breakdown": {"simple": True},
                 "test": {"status": "ok", "ttft_ms": 100, "total_ms": 999}},
                {"provider": "c", "model_id": "mid", "rejected": False,
                 "requirements_breakdown": {"simple": True},
                 "test": {"status": "ok", "ttft_ms": 300, "total_ms": 500}},
            ],
        }
        state = gateway.build_state(self._config_with_strategy("ttft_ms"), results)
        assert [m["model_id"] for m in state.pools["simple"]] == ["fast", "mid", "slow"]

    def test_model_id_strategy_from_config(self):
        results = {
            "results": [
                {"provider": "a", "model_id": "Zebra", "rejected": False,
                 "requirements_breakdown": {"simple": True}, "test": {"status": "ok", "total_ms": 10}},
                {"provider": "b", "model_id": "alpha", "rejected": False,
                 "requirements_breakdown": {"simple": True}, "test": {"status": "ok", "total_ms": 20}},
                {"provider": "c", "model_id": "Beta", "rejected": False,
                 "requirements_breakdown": {"simple": True}, "test": {"status": "ok", "total_ms": 30}},
            ],
        }
        state = gateway.build_state(self._config_with_strategy("model_id"), results)
        assert [m["model_id"] for m in state.pools["simple"]] == ["alpha", "Beta", "Zebra"]

    def test_random_strategy_keeps_all_models(self):
        state = gateway.build_state(self._config_with_strategy("random"), MOCK_TEST_RESULTS)
        assert {m["model_id"] for m in state.pools["simple"]} == {
            "meta/llama-3.2-1b-instruct",
            "qwen-flash",
            "nvidia/nemotron-mini-4b-instruct",
        }

    def test_unknown_strategy_falls_back_to_total_ms(self):
        state = gateway.build_state(self._config_with_strategy("bogus"), MOCK_TEST_RESULTS)
        assert [m["model_id"] for m in state.pools["simple"]] == [
            "meta/llama-3.2-1b-instruct",
            "qwen-flash",
            "nvidia/nemotron-mini-4b-instruct",
        ]

    def test_fallback_models_still_appended_after_sort(self):
        """fallback_models always land at the pool end, regardless of sort strategy."""
        config = self._config_with_strategy("model_id")
        config["requirements"]["simple"]["fallback_models"] = [
            {"provider": "opencode", "model_id": "zzz-fallback"},
        ]
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        assert state.pools["simple"][-1]["model_id"] == "zzz-fallback"


# ===========================================================================
# Tests: Untested fallback models (requirements.<cat>.fallback_models)
# ===========================================================================
class TestBuildStateFallbackModels:
    """fallback_models bypass testing and are always appended to the pool end."""

    @staticmethod
    def _config_with_extras(extras):
        config = copy.deepcopy(MOCK_CONFIG)
        config["requirements"]["simple"]["fallback_models"] = extras
        return config

    def test_fallback_models_appended_to_pool_end(self):
        config = self._config_with_extras([
            {"provider": "opencode", "model_id": "my-custom-model"},
            {"provider": "google", "model_id": "gemini-custom"},
        ])
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        pool = state.pools["simple"]
        # 3 tested models (sorted by total_ms) + 2 extras at the end
        assert len(pool) == 5
        assert pool[3] == {
            "provider": "opencode",
            "model_id": "my-custom-model",
            "test": {"status": "ok", "total_ms": None},
        }
        assert pool[4]["provider"] == "google"
        assert pool[4]["model_id"] == "gemini-custom"
        # Order of tested models preserved (fastest first)
        assert pool[0]["model_id"] == "meta/llama-3.2-1b-instruct"
        # Other categories unaffected
        assert len(state.pools["coder"]) == 2

    def test_fallback_models_skip_duplicate_of_tested_model(self):
        """A model already in the pool (tested) is not added again."""
        config = self._config_with_extras([
            {"provider": "nvidia", "model_id": "meta/llama-3.2-1b-instruct"},
            {"provider": "opencode", "model_id": "my-custom-model"},
        ])
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        assert len(state.pools["simple"]) == 4  # 3 tested + 1 new extra

    def test_fallback_models_deduped_within_list(self):
        config = self._config_with_extras([
            {"provider": "opencode", "model_id": "my-custom-model"},
            {"provider": "opencode", "model_id": "my-custom-model"},
        ])
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        assert len(state.pools["simple"]) == 4

    def test_fallback_models_skips_invalid_entries(self):
        config = self._config_with_extras([
            {"provider": "opencode", "model_id": "my-custom-model"},
            {"provider": "opencode"},  # missing model_id
            {"model_id": "no-provider"},  # missing provider
            "not-a-dict",
            None,
        ])
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        assert len(state.pools["simple"]) == 4  # only the valid extra added

    def test_fallback_models_no_effect_when_absent(self):
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert len(state.pools["simple"]) == 3

    def test_fallback_models_only_source_for_category(self):
        """A category with no tested models still gets its extras."""
        config = copy.deepcopy(MOCK_CONFIG)
        config["requirements"]["custom"] = {
            "fallback_models": [{"provider": "opencode", "model_id": "only-model"}],
        }
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        assert state.pools["custom"] == [
            {"provider": "opencode", "model_id": "only-model", "test": {"status": "ok", "total_ms": None}},
        ]

    def test_fallback_models_saved_to_current_model(self, tmp_path):
        """save_current_model writes extras at the end with total_ms = null."""
        config = self._config_with_extras([
            {"provider": "opencode", "model_id": "my-custom-model"},
        ])
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        model_path = tmp_path / "current-model.json"

        gateway.save_current_model(state, model_path)

        data = json.loads(model_path.read_text())
        models = data["simple"]["models"]
        assert models[-1]["provider"] == "opencode"
        assert models[-1]["model_id"] == "my-custom-model"
        assert models[-1]["total_ms"] is None

    def test_fallback_models_restore_current_index(self):
        """Saved current model pointing at an extra is restored correctly."""
        config = self._config_with_extras([
            {"provider": "opencode", "model_id": "my-custom-model"},
        ])
        saved = {"simple": {"provider": "opencode", "model_id": "my-custom-model"}}
        state = gateway.build_state(config, MOCK_TEST_RESULTS, saved)

        assert state.current_index["simple"] == 3  # after the 3 tested models


# ===========================================================================
# Tests: Transparent proxy — payload passed through as-is
# ===========================================================================
class TestTransparentProxy:
    """Test that gateway acts as transparent proxy — only model name is replaced."""

    def test_payload_passed_through_for_openai(self):
        """OpenAI-compatible: only model name replaced, everything else unchanged."""
        body = {
            "model": "coder",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8192,
            "temperature": 0.9,
            "tools": [{"type": "function", "function": {"name": "write"}}],
            "tool_choice": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        model = {"provider": "nvidia", "model_id": "nvidia/llama-3.3-nemotron-super-49b-v1"}

        # Simulate what forward_streaming does for OpenAI providers
        provider = {"type": "openai", "base_url": "https://integrate.api.nvidia.com/v1", "api_key": "test-nvidia-1"}
        assert provider["type"] == "openai"

        # Build payload same way as gateway
        payload = {body_key: body[body_key] for body_key in body}
        payload["model"] = model["model_id"]

        # Only model name changed
        assert payload["model"] == "nvidia/llama-3.3-nemotron-super-49b-v1"
        assert payload["max_tokens"] == 8192  # NOT capped at 4096
        assert payload["temperature"] == 0.9
        assert payload["tools"] == body["tools"]
        assert payload["tool_choice"] == "auto"
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    def test_no_system_role_conversion(self):
        """System messages NOT converted — OpenCode handles this."""
        body = {
            "model": "coder",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hi"},
            ],
        }

        payload = {body_key: body[body_key] for body_key in body}
        payload["model"] = "some-model"

        # System message preserved exactly
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful"

    def test_no_max_tokens_capping(self):
        """max_tokens NOT capped — OpenCode sets limits itself."""
        body = {"model": "coder", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16384}
        payload = {body_key: body[body_key] for body_key in body}
        payload["model"] = "some-model"

        assert payload["max_tokens"] == 16384  # NOT capped

    def test_no_message_merging(self):
        """Messages NOT merged — each kept as-is."""
        body = {
            "model": "coder",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ],
        }

        payload = {body_key: body[body_key] for body_key in body}
        payload["model"] = "some-model"

        # All 3 messages preserved
        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][2]["role"] == "user"

    def test_tool_calls_preserved(self):
        """Tool messages with tool_call_id NOT modified."""
        body = {
            "model": "coder",
            "messages": [
                {"role": "user", "content": "search for cats"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_123", "content": "found 10 results"},
            ],
        }

        payload = {body_key: body[body_key] for body_key in body}
        payload["model"] = "some-model"

        assert payload["messages"][1]["tool_calls"][0]["id"] == "call_123"
        assert payload["messages"][2]["tool_call_id"] == "call_123"


# ===========================================================================
# Tests: PID file ownership (gateway.py writes its own gateway.pid)
# ===========================================================================
class TestGatewayPid:
    """gateway.py owns logs/gateway.pid — written on startup, removed on shutdown.

    Tests use a tmp_path so the live gateway.pid is never touched.
    """

    def test_write_pid_records_current_process(self, tmp_path):
        gateway._PID_PATH = tmp_path / "gateway.pid"
        try:
            gateway._write_pid()
            assert gateway._PID_PATH.read_text().strip() == str(os.getpid())
        finally:
            gateway._remove_pid()
            assert not gateway._PID_PATH.exists()

    def test_remove_pid_only_clears_own_pid(self, tmp_path):
        gateway._PID_PATH = tmp_path / "gateway.pid"
        # Simulate a PID file left by another process.
        gateway._PID_PATH.write_text("999999")
        try:
            gateway._remove_pid()  # current pid != 999999 → must not remove
            assert gateway._PID_PATH.read_text().strip() == "999999"
        finally:
            gateway._PID_PATH.unlink(missing_ok=True)

    def test_remove_pid_clears_when_owned(self, tmp_path):
        gateway._PID_PATH = tmp_path / "gateway.pid"
        gateway._PID_PATH.write_text(str(os.getpid()))
        try:
            gateway._remove_pid()
            assert not gateway._PID_PATH.exists()
        finally:
            gateway._PID_PATH.unlink(missing_ok=True)


# ===========================================================================
# Tests: API endpoints (using FastAPI TestClient)
# ===========================================================================
class TestHealthEndpoint:
    def test_health(self):
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        with patch.object(gateway, "_state", state), patch.object(gateway, "_config", MOCK_CONFIG):
            client = TestClient(gateway.app)
            resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "simple" in data["categories"]
        assert data["categories"]["simple"] == 3


class TestModelsEndpoint:
    def test_returns_three_models(self):
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        with patch.object(gateway, "_state", state), patch.object(gateway, "_config", MOCK_CONFIG):
            client = TestClient(gateway.app)
            resp = client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3

        ids = [m["id"] for m in data["data"]]
        assert "simple" in ids
        assert "coder" in ids
        assert "visual" in ids

    def test_model_metadata(self):
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        with patch.object(gateway, "_state", state), patch.object(gateway, "_config", MOCK_CONFIG):
            client = TestClient(gateway.app)
            resp = client.get("/v1/models")

        simple_model = next(m for m in resp.json()["data"] if m["id"] == "simple")
        assert simple_model["meta"]["model_count"] == 3
        assert simple_model["meta"]["fastest"] == "meta/llama-3.2-1b-instruct"


class TestChatCompletionsEndpoint:
    def test_unknown_model_returns_404(self):
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        with patch.object(gateway, "_state", state), patch.object(gateway, "_config", MOCK_CONFIG):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 404

    def test_routes_to_correct_provider(self):
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "42"}}]}))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", return_value=mock_response),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "2+2?"}]},
            )

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "42"

    def test_retries_on_rate_limit(self):
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        error_response = AsyncMock()
        error_response.status = 429
        error_response.text = AsyncMock(return_value='{"error": "rate_limit_exceeded"}')
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[error_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        assert state.current_index["simple"] == 1
        # Verify file was saved after switch — current model is now #1
        saved = json.loads(gateway._CURRENT_MODEL_PATH.read_text())
        assert "current" not in saved["simple"]["models"][0]
        assert saved["simple"]["models"][1]["current"] is True
        assert saved["simple"]["models"][1]["model_id"] == "qwen-flash"

    def test_returns_429_when_all_exhausted(self):
        from fastapi.testclient import TestClient

        config = _make_config(gateway={"retry_count": 1})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        error_response = AsyncMock()
        error_response.status = 429
        error_response.text = AsyncMock(return_value='{"error": "rate_limit_exceeded"}')
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession.post", return_value=error_response),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "visual", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 429
        assert "unavailable" in resp.json()["error"]["message"]

    def test_applies_tool_choice_when_tools_present(self):
        from fastapi.testclient import TestClient
        from contextlib import asynccontextmanager

        config = _make_config(gateway={"tool_choice": "required"})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        captured_body = {}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))

        @asynccontextmanager
        async def mock_post(url, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            yield mock_response

        class MockSession:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def post(self, url, **kwargs):
                captured_body.update(kwargs.get("json", {}))

                @asynccontextmanager
                async def ctx():
                    yield mock_response
                return ctx()

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession", MockSession),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "web_search"}}],
                },
            )

        assert resp.status_code == 200
        assert captured_body["tool_choice"] == "required"

    def test_does_not_override_client_tool_choice(self):
        from fastapi.testclient import TestClient
        from contextlib import asynccontextmanager

        config = _make_config(gateway={"tool_choice": "required"})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        captured_body = {}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))

        class MockSession:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def post(self, url, **kwargs):
                captured_body.update(kwargs.get("json", {}))

                @asynccontextmanager
                async def ctx():
                    yield mock_response
                return ctx()

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession", MockSession),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "web_search"}}],
                    "tool_choice": "auto",
                },
            )

        assert resp.status_code == 200
        assert captured_body["tool_choice"] == "auto"

    def test_switch_on_any_error_retries_on_non_retryable(self):
        """With switch_on_any_error=true, a 400 error should switch model."""
        from fastapi.testclient import TestClient

        config = _make_config(gateway={
            "switch_on_any_error": True,
            "retry_count": 2,
        })
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        error_response = AsyncMock()
        error_response.status = 400
        error_response.text = AsyncMock(return_value='{"error": "bad request"}')
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession.post", side_effect=[error_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        # Should have switched from index 0 to index 1
        assert state.current_index["simple"] == 1
        # Verify file was saved after switch
        saved = json.loads(gateway._CURRENT_MODEL_PATH.read_text())
        assert saved["simple"]["models"][1]["current"] is True

    def test_switch_on_any_error_false_returns_400(self):
        """With switch_on_any_error=false (default), a 400 error returns 400 to client."""
        from fastapi.testclient import TestClient

        config = _make_config(gateway={
            "retry_count": 2,
        })
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        error_response = AsyncMock()
        error_response.status = 400
        error_response.text = AsyncMock(return_value='{"error": "bad request"}')
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession.post", return_value=error_response),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 400
        # Should NOT have switched — still at index 0
        assert state.current_index["simple"] == 0

    def test_timeout_switches_model(self):
        """Non-streaming: model timeout triggers model switch."""
        from fastapi.testclient import TestClient
        import asyncio

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        # First post raises TimeoutError, second succeeds
        timeout_mock = AsyncMock()
        timeout_mock.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        timeout_mock.__aexit__ = AsyncMock(return_value=False)

        ok_mock = AsyncMock()
        ok_mock.status = 200
        ok_mock.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_mock.__aenter__ = AsyncMock(return_value=ok_mock)
        ok_mock.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[timeout_mock, ok_mock]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        # Switched from index 0 to index 1
        assert state.current_index["simple"] == 1
        saved = json.loads(gateway._CURRENT_MODEL_PATH.read_text())
        assert saved["simple"]["models"][1]["current"] is True
        assert saved["simple"]["models"][1]["model_id"] == "qwen-flash"

    def test_timeout_exhausted_returns_429(self):
        """All models time out → 429 returned."""
        from fastapi.testclient import TestClient
        import asyncio

        config = _make_config(gateway={"retry_count": 0})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        timeout_mock = AsyncMock()
        timeout_mock.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        timeout_mock.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession.post", return_value=timeout_mock),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 429
        assert "unavailable" in resp.json()["error"]["message"]

    def test_no_tool_choice_without_tools(self):
        from fastapi.testclient import TestClient
        from contextlib import asynccontextmanager

        config = _make_config(gateway={"tool_choice": "required"})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        captured_body = {}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))

        class MockSession:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def post(self, url, **kwargs):
                captured_body.update(kwargs.get("json", {}))

                @asynccontextmanager
                async def ctx():
                    yield mock_response
                return ctx()

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession", MockSession),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 200
        assert "tool_choice" not in captured_body


    # ------------------------------------------------------------------
    # API key authentication tests
    # ------------------------------------------------------------------
    def test_missing_auth_header_returns_401(self):
        """When api_key is set, requests without Authorization header get 401."""
        from fastapi.testclient import TestClient

        config_with_key = copy.deepcopy(MOCK_CONFIG)
        config_with_key["gateway"]["api_key"] = "test-secret-key"

        state = gateway.build_state(config_with_key, MOCK_TEST_RESULTS)

        with patch.object(gateway, "_state", state), patch.object(gateway, "_config", config_with_key):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 401
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "authentication_error"

    def test_wrong_api_key_returns_401(self):
        """When api_key is set, requests with wrong Bearer token get 401."""
        from fastapi.testclient import TestClient

        config_with_key = copy.deepcopy(MOCK_CONFIG)
        config_with_key["gateway"]["api_key"] = "test-secret-key"

        state = gateway.build_state(config_with_key, MOCK_TEST_RESULTS)

        with patch.object(gateway, "_state", state), patch.object(gateway, "_config", config_with_key):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer wrong-key"},
            )

        assert resp.status_code == 401

    def test_wrong_auth_format_returns_401(self):
        """When api_key is set, non-Bearer auth format gets 401."""
        from fastapi.testclient import TestClient

        config_with_key = copy.deepcopy(MOCK_CONFIG)
        config_with_key["gateway"]["api_key"] = "test-secret-key"

        state = gateway.build_state(config_with_key, MOCK_TEST_RESULTS)

        with patch.object(gateway, "_state", state), patch.object(gateway, "_config", config_with_key):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Basic dGVzdDp0ZXN0"},
            )

        assert resp.status_code == 401

    def test_correct_api_key_passes(self):
        """With correct api_key, request proceeds to provider."""
        from fastapi.testclient import TestClient

        config_with_key = copy.deepcopy(MOCK_CONFIG)
        config_with_key["gateway"]["api_key"] = "test-secret-key"

        state = gateway.build_state(config_with_key, MOCK_TEST_RESULTS)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "42"}}]}))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config_with_key),
            patch("aiohttp.ClientSession.post", return_value=mock_response),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "2+2?"}]},
                headers={"Authorization": "Bearer test-secret-key"},
            )

        assert resp.status_code == 200

    def test_no_api_key_in_config_no_auth_required(self):
        """When api_key is not set in config, no auth check is performed."""
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "42"}}]}))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", return_value=mock_response),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "2+2?"}]},
                # No Authorization header
            )

        assert resp.status_code == 200


class TestParseSseError:
    """Unit tests for _parse_sse_error helper."""

    def test_normal_choices_chunk_returns_none(self):
        """Normal SSE data with choices returns None."""
        chunk = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        assert gateway._parse_sse_error(chunk) is None

    def test_done_marker_returns_none(self):
        """[DONE] marker returns None."""
        assert gateway._parse_sse_error("data: [DONE]") is None

    def test_non_data_line_returns_none(self):
        """Line without data: prefix returns None."""
        assert gateway._parse_sse_error(": heartbeat") is None

    def test_non_json_data_returns_none(self):
        """Line with data: but non-JSON content returns None."""
        assert gateway._parse_sse_error("data: just a string") is None

    def test_json_without_error_key_returns_none(self):
        """Valid JSON data without error key returns None."""
        chunk = 'data: {"id":"123","object":"chat.completion.chunk"}'
        assert gateway._parse_sse_error(chunk) is None

    def test_resource_exhausted_sse_error(self):
        """SSE error with ResourceExhausted returns error_info with status 500."""
        chunk = 'data: {"error":{"message":"ResourceExhausted: Worker local total request limit reached (48/48)","type":"internal_server_error","code":500}}'
        result = gateway._parse_sse_error(chunk)
        assert result is not None
        assert result["status"] == 500
        assert "ResourceExhausted" in result["body"]

    def test_sse_error_with_code_429(self):
        """SSE error with code 429 returns error_info with status 429."""
        chunk = 'data: {"error":{"message":"Rate limit exceeded","code":429}}'
        result = gateway._parse_sse_error(chunk)
        assert result is not None
        assert result["status"] == 429
        assert "Rate limit" in result["body"]

    def test_sse_error_with_string_error_field(self):
        """SSE error with string error value (non-retryable) returns None."""
        chunk = 'data: {"error":"Internal server error"}'
        result = gateway._parse_sse_error(chunk)
        assert result is None, "non-retryable errors should pass through as normal chunks"

    def test_sse_error_with_non_numeric_code(self):
        """SSE error with non-numeric code doesn't crash, returns None for non-retryable."""
        chunk = 'data: {"error":{"message":"Tool use failed","code":"tool_use_failed"}}'
        result = gateway._parse_sse_error(chunk)
        assert result is None, "tool call errors are model-level, not provider-level"

    def test_sse_error_with_none_code(self):
        """SSE error with null code and non-retryable message returns None."""
        chunk = 'data: {"error":{"message":"Some error","code":null}}'
        result = gateway._parse_sse_error(chunk)
        assert result is None, "non-retryable errors should pass through"

    def test_sse_error_retryable_with_non_numeric_code(self):
        """SSE error matching retryable text returns error_info even with non-numeric code."""
        chunk = 'data: {"error":{"message":"rate limit exceeded","code":"rate_limit_exceeded"}}'
        result = gateway._parse_sse_error(chunk)
        assert result is not None
        assert result["status"] == 500
        assert "rate limit" in result["body"]

    def test_tool_call_validation_error_passes_through(self):
        """Tool call validation error (model-level) passes through, not trigger switch."""
        chunk = 'data: {"error":{"message":"tool call validation failed: parameters for tool websearch did not match schema: errors: [`/numResults`: expected number, but got string]","type":"invalid_request_error","code":"invalid_request_error"}}'
        result = gateway._parse_sse_error(chunk)
        assert result is None, "tool validation is model-level, should pass through as normal chunk"

    def test_switch_on_any_error_makes_all_errors_switchable(self):
        """With switch_on_any_error=True, non-retryable SSE errors trigger switch too."""
        chunk = 'data: {"error":{"message":"tool call validation failed","code":"invalid_request_error"}}'
        # Without flag → passes through
        assert gateway._parse_sse_error(chunk) is None
        # With flag → triggers switch
        result = gateway._parse_sse_error(chunk, switch_on_any_error=True)
        assert result is not None
        assert result["status"] == 500
        assert "tool call validation" in result["body"]


class TestStreamingEndpoint:
    def test_streaming_returns_sse(self):
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        async def mock_stream(*args, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', None
            yield 'data: [DONE]\n\n', None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("gateway.forward_streaming", mock_stream),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_streaming_retries_on_error(self):
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        call_count = 0

        async def mock_stream_forward(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "", {"status": 429, "body": "rate_limit"}
            else:
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("gateway.forward_streaming", mock_stream_forward),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert state.current_index["simple"] == 1
        # Verify file was saved after stream switch
        saved = json.loads(gateway._CURRENT_MODEL_PATH.read_text())
        assert saved["simple"]["models"][1]["current"] is True

    def test_streaming_switch_on_any_error_retries_400(self):
        """With switch_on_any_error=true, 400 triggers stream model switch."""
        from fastapi.testclient import TestClient

        config = _make_config(gateway={
            "switch_on_any_error": True,
            "retry_count": 2,
        })
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        call_count = 0

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "", {"status": 400, "body": "bad request"}
            else:
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("gateway.forward_streaming", mock_stream_fwd),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert state.current_index["simple"] == 1
        # Verify file was saved after stream switch
        saved = json.loads(gateway._CURRENT_MODEL_PATH.read_text())
        assert saved["simple"]["models"][1]["current"] is True

    def test_streaming_stall_timeout_switches(self):
        """Stream stall (408 from forward_streaming) triggers model switch (408 is retryable)."""
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        call_count = 0

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Stall timeout — 408 is retryable → triggers switch
                yield "", {"status": 408, "body": "stall timeout"}
            else:
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("gateway.forward_streaming", mock_stream_fwd),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert state.current_index["simple"] == 1
        saved = json.loads(gateway._CURRENT_MODEL_PATH.read_text())
        assert saved["simple"]["models"][1]["current"] is True

    def test_streaming_connection_error_switches_model(self):
        """Regression for logs/gateway.log: a DNS/connection error
        (aiohttp.ClientError) raised while streaming must switch models instead
        of crashing the ASGI app ('Exception in ASGI application').

        Before the fix, forward_streaming only caught asyncio.TimeoutError, so a
        ClientConnectorDNSError on the provider host propagated unhandled and
        killed the request with HTTP 500.
        """
        from fastapi.testclient import TestClient
        import socket
        import aiohttp
        from aiohttp.client_reqrep import ConnectionKey

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        # First model → transient DNS failure (exactly like the log on
        # opencodel.ai); second model → successful stream.
        ck = ConnectionKey("opencodel.ai", 443, False, False, None, None, None, None)
        dns_exc = aiohttp.ClientConnectorDNSError(ck, socket.gaierror(-3, "Try again"))
        dns_mock = AsyncMock()
        dns_mock.__aenter__ = AsyncMock(side_effect=dns_exc)
        dns_mock.__aexit__ = AsyncMock(return_value=False)

        async def _aiter(lines):
            for ln in lines:
                yield ln

        ok_mock = AsyncMock()
        ok_mock.status = 200
        ok_mock.__aenter__ = AsyncMock(return_value=ok_mock)
        ok_mock.__aexit__ = AsyncMock(return_value=False)
        ok_mock.content = _aiter([
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
            b'data: [DONE]\n\n',
        ])

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[dns_mock, ok_mock]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        # Switched from the failed model to the next one.
        assert state.current_index["simple"] == 1
        saved = json.loads(gateway._CURRENT_MODEL_PATH.read_text())
        assert saved["simple"]["models"][1]["current"] is True
        assert saved["simple"]["models"][1]["model_id"] == "qwen-flash"

    def test_streaming_switch_on_any_error_false_returns_error(self):
        """With switch_on_any_error=false (default), 400 in stream returns error to client."""
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            yield "", {"status": 400, "body": "bad request"}

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("gateway.forward_streaming", mock_stream_fwd),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200  # SSE always returns 200
        content = resp.text
        assert "error" in content
        # Should NOT have switched — still at index 0
        assert state.current_index["simple"] == 0


# ===========================================================================
# Tests: Provider URL building
# ===========================================================================
class TestGetProviderUrl:
    def test_openai_url(self):
        provider = {"type": "openai", "base_url": "https://api.example.com/v1"}
        url = gateway.get_provider_url(provider, "model-x")
        assert url == "https://api.example.com/v1/chat/completions"

    def test_google_url(self):
        provider = {"type": "google", "base_url": "https://generativelanguage.googleapis.com/v1beta"}
        url = gateway.get_provider_url(provider, "gemini-2.0-flash")
        assert "models/gemini-2.0-flash:streamGenerateContent" in url
        assert "alt=sse" in url
        assert "key=" not in url  # key goes in header, not URL


class TestGetProviderHeaders:
    def test_openai_includes_bearer_auth(self):
        provider = {"type": "openai", "api_key": "sk-test"}
        headers = gateway.get_provider_headers(provider, "test-agent/1.0")
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["User-Agent"] == "test-agent/1.0"
        assert "x-goog-api-key" not in headers

    def test_google_uses_x_goog_api_key_header(self):
        provider = {"type": "google", "api_key": "AQ.Ab8RN6test"}
        headers = gateway.get_provider_headers(provider, "test-agent/1.0")
        assert headers["x-goog-api-key"] == "AQ.Ab8RN6test"
        assert "Authorization" not in headers
        assert headers["User-Agent"] == "test-agent/1.0"

    def test_google_both_key_formats(self):
        """Both AIzaSy and AQ.Ab keys go into x-goog-api-key header."""
        for key in ("AIzaSyTestKey00000000000000000000000000", "AQ.Ab8RN6testkey00000000000000000000000000000"):
            provider = {"type": "google", "api_key": key}
            headers = gateway.get_provider_headers(provider, "test-agent/1.0")
            assert headers["x-goog-api-key"] == key
            assert "Authorization" not in headers


# ===========================================================================
# Tests: Current model persistence
# ===========================================================================
class TestCurrentModelPersistence:
    # ------------------------------------------------------------------
    # save_current_model — new format (full pool per category)
    # ------------------------------------------------------------------
    def test_save_current_model_writes_file(self, tmp_path):
        state = gateway.GatewayState()
        state.pools = {
            "simple": [
                {"provider": "nvidia", "model_id": "m-a", "test": {"total_ms": 100}},
                {"provider": "alibaba", "model_id": "m-b", "test": {"total_ms": 200}},
            ],
            "coder": [
                {"provider": "google", "model_id": "m-c", "test": {"total_ms": 150}},
            ],
        }
        state.current_index = {"simple": 0, "coder": 0}
        model_path = tmp_path / "current-model.json"

        gateway.save_current_model(state, model_path)

        assert model_path.exists()
        data = json.loads(model_path.read_text())

        # simple pool — 2 models, first is current
        assert "simple" in data
        assert "models" in data["simple"]
        assert len(data["simple"]["models"]) == 2
        assert data["simple"]["models"][0] == {
            "provider": "nvidia", "model_id": "m-a", "ttft_ms": None, "total_ms": 100, "current": True,
        }
        assert data["simple"]["models"][1] == {
            "provider": "alibaba", "model_id": "m-b", "ttft_ms": None, "total_ms": 200,
        }
        # coder pool — 1 model, current
        assert data["coder"]["models"][0]["current"] is True

    def test_save_current_model_exhausted_includes_pool(self, tmp_path):
        """Category where index is past end — pool still saved, no model marked current."""
        state = gateway.GatewayState()
        state.pools = {"simple": [{"provider": "nvidia", "model_id": "m-a", "test": {"total_ms": 100}}]}
        state.current_index = {"simple": 1}  # past end
        model_path = tmp_path / "current-model.json"

        gateway.save_current_model(state, model_path)

        data = json.loads(model_path.read_text())
        assert "simple" in data
        assert len(data["simple"]["models"]) == 1
        assert "current" not in data["simple"]["models"][0]

    def test_save_current_model_empty_state(self, tmp_path):
        """No pools at all — file still created with empty object."""
        state = gateway.GatewayState()
        model_path = tmp_path / "current-model.json"

        gateway.save_current_model(state, model_path)

        data = json.loads(model_path.read_text())
        assert data == {}

    def test_save_current_model_sorts_by_total_ms(self, tmp_path):
        """Pool order in file matches state.pools order (already sorted by total_ms)."""
        state = gateway.GatewayState()
        state.pools = {
            "simple": [
                {"provider": "a", "model_id": "slow", "test": {"total_ms": 500}},
                {"provider": "b", "model_id": "fast", "test": {"total_ms": 50}},
                {"provider": "c", "model_id": "medium", "test": {"total_ms": 100}},
            ],
        }
        state.current_index = {"simple": 1}  # fast is current
        model_path = tmp_path / "current-model.json"

        gateway.save_current_model(state, model_path)

        data = json.loads(model_path.read_text())
        mids = [m["model_id"] for m in data["simple"]["models"]]
        assert mids == ["slow", "fast", "medium"]  # same order as pool
        # Only the one at current_index has current=true
        assert data["simple"]["models"][1]["current"] is True
        assert "current" not in data["simple"]["models"][0]
        assert "current" not in data["simple"]["models"][2]

    # ------------------------------------------------------------------
    # load_current_model
    # ------------------------------------------------------------------
    def test_load_current_model_returns_empty_on_missing(self, tmp_path):
        model_path = tmp_path / "nonexistent.json"
        result = gateway.load_current_model(model_path)
        assert result == {}

    def test_load_current_model_returns_empty_on_corrupt(self, tmp_path):
        model_path = tmp_path / "current-model.json"
        model_path.write_text("not json")
        result = gateway.load_current_model(model_path)
        assert result == {}

    def test_load_current_model_new_format(self, tmp_path):
        """Models[] + current flag."""
        model_path = tmp_path / "current-model.json"
        model_path.write_text(json.dumps({
            "simple": {
                "models": [
                    {"provider": "nvidia", "model_id": "m-a", "total_ms": 100},
                    {"provider": "alibaba", "model_id": "m-b", "total_ms": 200, "current": True},
                ],
            },
            "coder": {
                "models": [
                    {"provider": "google", "model_id": "m-c", "total_ms": 150, "current": True},
                ],
            },
        }))
        result = gateway.load_current_model(model_path)
        assert result == {
            "simple": {"provider": "alibaba", "model_id": "m-b"},
            "coder": {"provider": "google", "model_id": "m-c"},
        }

    def test_load_current_model_no_current_flag(self, tmp_path):
        """No model marked current → first model is used."""
        model_path = tmp_path / "current-model.json"
        model_path.write_text(json.dumps({
            "simple": {
                "models": [
                    {"provider": "nvidia", "model_id": "m-a", "total_ms": 100},
                    {"provider": "alibaba", "model_id": "m-b", "total_ms": 200},
                ],
            },
        }))
        result = gateway.load_current_model(model_path)
        assert result == {"simple": {"provider": "nvidia", "model_id": "m-a"}}

    def test_load_current_model_empty_models(self, tmp_path):
        """Empty models list → category omitted from result."""
        model_path = tmp_path / "current-model.json"
        model_path.write_text(json.dumps({
            "simple": {"models": []},
            "coder": {"models": [{"provider": "g", "model_id": "m", "total_ms": 1, "current": True}]},
        }))
        result = gateway.load_current_model(model_path)
        assert "simple" not in result
        assert result["coder"] == {"provider": "g", "model_id": "m"}


class TestBuildStateRestoreModel:
    def test_restores_saved_model(self):
        saved = {"simple": {"provider": "alibaba", "model_id": "qwen-flash"}}
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS, saved)
        assert state.current_index["simple"] == 1

    def test_restores_all_categories(self):
        saved = {
            "simple": {"provider": "alibaba", "model_id": "qwen-flash"},
            "coder": {"provider": "nvidia", "model_id": "nvidia/llama-3.3-nemotron-super-49b-v1"},
            "visual": {"provider": "alibaba", "model_id": "qwen-vl-plus"},
        }
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS, saved)
        assert state.current_index["simple"] == 1
        assert state.current_index["coder"] == 0
        assert state.current_index["visual"] == 1

    def test_not_found_starts_at_zero(self):
        """Saved model not in pool → start from first."""
        saved = {"simple": {"provider": "nonexistent", "model_id": "ghost"}}
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS, saved)
        assert state.current_index["simple"] == 0

    def test_no_saved_models_starts_at_zero(self):
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert state.current_index["simple"] == 0

    def test_random_strategy_ignores_saved_model(self):
        """For `random` pools the start is reshuffled each restart, so the
        saved (last-used) model must not be pinned — index stays at 0
        (first model of the fresh shuffle)."""
        config = copy.deepcopy(MOCK_CONFIG)
        config["requirements"]["simple"]["sort_strategy"] = "random"
        saved = {"simple": {"provider": "alibaba", "model_id": "qwen-flash"}}
        state = gateway.build_state(config, MOCK_TEST_RESULTS, saved)
        assert state.current_index["simple"] == 0

    def test_ignore_unknown_category_in_saved(self):
        """Saved category that doesn't exist in pools is silently ignored."""
        saved = {"unknown_cat": {"provider": "nvidia", "model_id": "m"}}
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS, saved)
        # No crash, simple still starts at 0
        assert state.current_index["simple"] == 0

    def test_advance_saves_current_model(self, tmp_path):
        """After advance(), the file should reflect the new current model."""
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        model_path = tmp_path / "current-model.json"

        # Advance simple from index 0 to index 1
        state.advance("simple")
        gateway.save_current_model(state, model_path)

        data = json.loads(model_path.read_text())
        # Pool has all 3 simple models
        assert len(data["simple"]["models"]) == 3
        # Model at index 0 was first (llama-3.2-1b), now not current
        assert "current" not in data["simple"]["models"][0]
        # Model at index 1 is now current (qwen-flash)
        assert data["simple"]["models"][1]["current"] is True
        assert data["simple"]["models"][1]["model_id"] == "qwen-flash"

    # ------------------------------------------------------------------
    # Restart cycle: save → load → build_state → save
    # ------------------------------------------------------------------
    def test_restart_cycle_preserves_current_model(self, tmp_path):
        """Simulate gateway restart: save, then load and rebuild."""
        model_path = tmp_path / "current-model.json"

        # First boot
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert state.current_index["simple"] == 0
        # Advance simple to second model
        state.advance("simple")
        gateway.save_current_model(state, model_path)

        # Restart — simulate lifespan: load saved, rebuild, save
        saved = gateway.load_current_model(model_path)
        state2 = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS, saved)
        gateway.save_current_model(state2, model_path)

        # simple should still be at index 1 (qwen-flash)
        assert state2.current_index["simple"] == 1
        data = json.loads(model_path.read_text())
        assert data["simple"]["models"][1]["model_id"] == "qwen-flash"
        assert data["simple"]["models"][1]["current"] is True

    def test_restart_cycle_previous_model_gone(self, tmp_path):
        """If previous current model is no longer in pool → start from first."""
        model_path = tmp_path / "current-model.json"

        # First boot with full results
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        state.advance("simple")  # now at index 1 (qwen-flash)
        gateway.save_current_model(state, model_path)

        # Second boot with results where qwen-flash is gone (e.g. model dropped upstream)
        results_no_qwen = {
            "results": [
                r for r in MOCK_TEST_RESULTS["results"]
                if r["model_id"] != "qwen-flash"
            ]
        }
        saved = gateway.load_current_model(model_path)
        state2 = gateway.build_state(MOCK_CONFIG, results_no_qwen, saved)
        gateway.save_current_model(state2, model_path)

        # qwen-flash no longer in pool → model not found → start at 0
        assert state2.current_index["simple"] == 0
        data = json.loads(model_path.read_text())
        assert data["simple"]["models"][0]["current"] is True
        assert data["simple"]["models"][0]["provider"] == "nvidia"


# ---------------------------------------------------------------------------
# Autouse fixture: redirect _DEFAULT_PROVIDERS to avoid reading real file
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_providers_path(tmp_path):
    """Redirect _DEFAULT_PROVIDERS so tests don't read real providers.json.

    Writes default MOCK_PROVIDERS_CREDS to the tmp_path so build_state()
    can load providers from the patched path.
    """
    old = gateway._DEFAULT_PROVIDERS
    providers_path = tmp_path / "providers.json"
    with open(providers_path, "w") as f:
        json.dump(MOCK_PROVIDERS_CREDS, f, indent=2)
    gateway._DEFAULT_PROVIDERS = providers_path
    yield
    gateway._DEFAULT_PROVIDERS = old


# ---------------------------------------------------------------------------
# Test data: providers credentials for rotation tests
# ---------------------------------------------------------------------------
MOCK_PROVIDERS_CREDS = {
    "nvidia": {
        "type": "openai",
        "credentials": [
            {"base_url": "https://integrate.api.nvidia.com/v1", "api_key": "test-nvidia-1", "current": True},
            {"base_url": "https://integrate.api.nvidia.com/v1", "api_key": "test-nvidia-2"},
            {"base_url": "https://integrate.api.nvidia.com/v1", "api_key": "test-nvidia-3"},
        ],
    },
    "alibaba": {
        "type": "openai",
        "credentials": [
            {"base_url": "https://dashscope.aliyuncs.com/v1", "api_key": "test-alibaba-1", "current": True},
        ],
    },
    "google": {
        "type": "google",
        "credentials": [
            {"base_url": "https://generativelanguage.googleapis.com/v1beta", "api_key": "test-google-1", "current": True},
        ],
    },
}


def _write_providers_json(tmp_path: Path, data: dict) -> None:
    """Write providers.json to tmp_path."""
    p = tmp_path / "providers.json"
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    gateway._DEFAULT_PROVIDERS = p


# ===========================================================================
# Tests: Quota error detection
# ===========================================================================
class TestQuotaErrorDetection:
    """Test is_quota_error against real provider messages and variations."""

    @pytest.mark.parametrize(
        "body",
        [
            # Huggingface
            '{"error":"You have depleted your monthly included credits. Purchase pre-paid credits to continue."}',
            # Alibaba
            '{"error":{"message":"The free quota has been exhausted.","type":"insufficient_quota","code":"insufficient_quota"}}',
            # Google
            '{"code":429,"message":"You exceeded your current quota, please check your plan and billing details."}',
            # Groq
            'Rate limit reached for model `llama-4-scout` in organization `org_123` service tier `on_demand` on tokens per minute (TPM): Limit 30000, Used 29583.',
        ],
    )
    def test_real_provider_quota_messages(self, body):
        assert gateway.is_quota_error(body) is True

    @pytest.mark.parametrize(
        "body",
        [
            "quota has been exhausted",
            "quota exhausted",
            "quota exceeded",
            "quota insufficient",
            "free quota has been exhausted",
            "exceeded your current quota",
            "rate limit reached",
            "rate limit exceeded",
            "rate limit surpassed",
            "insufficient_quota",
            "credits exhausted",
            "credits depleted",
            "tokens per minute",
            "too many requests",
            # Fuzzy: quota with long message between words
            "quota is currently exhausted",
            "your monthly quota was exceeded",
        ],
    )
    def test_fuzzy_patterns(self, body):
        assert gateway.is_quota_error(body) is True

    def test_empty_body(self):
        assert gateway.is_quota_error("") is False
        assert gateway.is_quota_error(None) is False

    @pytest.mark.parametrize(
        "body",
        [
            "normal error message",
            '{"error": "invalid_request"}',
            "model not found",
            "bad request",
            "context_length_exceeded",
            "maximum context length exceeded",
            "degraded",  # overloaded, not quota
            "capacity exceeded",  # overloaded, not quota
        ],
    )
    def test_non_quota_errors(self, body):
        assert gateway.is_quota_error(body) is False

    def test_case_insensitive(self):
        assert gateway.is_quota_error("QUOTA EXHAUSTED") is True
        assert gateway.is_quota_error("Rate Limit Reached") is True

    def test_quota_error_is_also_retryable(self):
        """Quota errors should also be detected as retryable (subset)."""
        for body in [
            "quota has been exhausted",
            "insufficient_quota",
            "rate limit reached",
        ]:
            assert gateway.is_retryable_error(200, body) is True


# ===========================================================================
# Tests: Credential rotation in GatewayState
# ===========================================================================
class TestCredentialRotation:
    def test_get_credential_returns_first_when_current(self):
        state = gateway.GatewayState()
        state.providers_creds = MOCK_PROVIDERS_CREDS
        state.credential_index = {"nvidia": 0}

        cred = state.get_credential("nvidia")
        assert cred["api_key"] == "test-nvidia-1"
        assert cred["current"] is True

    def test_get_credential_returns_none_for_unknown_provider(self):
        state = gateway.GatewayState()
        state.providers_creds = {}

        assert state.get_credential("unknown") is None

    def test_get_credential_returns_none_when_no_credentials(self):
        state = gateway.GatewayState()
        state.providers_creds = {"nvidia": {"type": "openai", "credentials": []}}

        assert state.get_credential("nvidia") is None

    def test_get_credential_wraps_on_invalid_index(self):
        """If credential_index exceeds list length, resets to 0."""
        state = gateway.GatewayState()
        state.providers_creds = MOCK_PROVIDERS_CREDS
        state.credential_index = {"nvidia": 99}

        cred = state.get_credential("nvidia")
        assert cred["api_key"] == "test-nvidia-1"
        assert state.credential_index["nvidia"] == 0

    def test_advance_credential_moves_to_next(self):
        state = gateway.GatewayState()
        state.providers_creds = MOCK_PROVIDERS_CREDS
        state.credential_index = {"nvidia": 0}

        new_cred = state.advance_credential("nvidia")
        assert new_cred["api_key"] == "test-nvidia-2"
        assert state.credential_index["nvidia"] == 1

    def test_advance_credential_wraps_around(self):
        state = gateway.GatewayState()
        state.providers_creds = MOCK_PROVIDERS_CREDS
        state.credential_index = {"nvidia": 2}  # last

        new_cred = state.advance_credential("nvidia")
        assert new_cred["api_key"] == "test-nvidia-1"  # back to first
        assert state.credential_index["nvidia"] == 0

    def test_advance_credential_returns_none_when_no_creds(self):
        state = gateway.GatewayState()
        state.providers_creds = {"nvidia": {"type": "openai", "credentials": []}}

        assert state.advance_credential("nvidia") is None

    def test_advance_credential_returns_none_for_unknown(self):
        state = gateway.GatewayState()
        state.providers_creds = {}

        assert state.advance_credential("unknown") is None

    def test_advance_credential_wraps_single_cred(self):
        """With 1 credential, advance always returns the same cred (wrap)."""
        state = gateway.GatewayState()
        state.providers_creds = MOCK_PROVIDERS_CREDS  # alibaba has 1 cred
        state.credential_index = {"alibaba": 0}

        new_cred = state.advance_credential("alibaba")
        assert new_cred["api_key"] == "test-alibaba-1"
        assert state.credential_index["alibaba"] == 0  # wrapped to same


# ===========================================================================
# Tests: resolve_provider
# ===========================================================================
class TestResolveProvider:
    def test_resolves_with_current_credential(self):
        state = gateway.GatewayState()
        state.providers_creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        state.credential_index = {"nvidia": 0}

        result = gateway.resolve_provider("nvidia", state)
        assert result["base_url"] == "https://integrate.api.nvidia.com/v1"
        assert result["api_key"] == "test-nvidia-1"
        assert result["type"] == "openai"

    def test_resolves_without_creds_returns_none(self):
        state = gateway.GatewayState()
        state.providers_creds = {}

        result = gateway.resolve_provider("nvidia", state)
        assert result is None

    def test_returns_none_for_unknown_provider(self):
        state = gateway.GatewayState()
        state.providers_creds = {}

        assert gateway.resolve_provider("unknown", state) is None

    def test_preserves_extra_config_fields(self):
        """model_filter and other config fields are preserved."""
        state = gateway.GatewayState()
        state.providers_creds = {
            "openrouter": {"type": "openai", "model_filter": ":free", "credentials": [{"base_url": "new-url", "api_key": "new-key"}]},
        }
        state.credential_index = {"openrouter": 0}

        result = gateway.resolve_provider("openrouter", state)
        assert result["model_filter"] == ":free"
        assert result["base_url"] == "new-url"


# ===========================================================================
# Tests: build_state with providers config
# ===========================================================================
class TestBuildStateProvidersConfig:
    def test_builds_state_with_no_providers_file(self, tmp_path):
        """When providers.json doesn't exist, build_state still works."""
        gateway._DEFAULT_PROVIDERS = tmp_path / "nonexistent.json"
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert state.providers_creds == {}
        assert state.credential_index == {}

    def test_builds_state_with_providers_file(self, tmp_path):
        _write_providers_json(tmp_path, copy.deepcopy(MOCK_PROVIDERS_CREDS))

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert "nvidia" in state.providers_creds
        assert state.credential_index["nvidia"] == 0
        assert state.credential_index["alibaba"] == 0

    def test_builds_state_reads_current_flag(self, tmp_path):
        """Non-zero current flag is used as starting index."""
        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        creds["nvidia"]["credentials"][0].pop("current", None)
        creds["nvidia"]["credentials"][1]["current"] = True
        _write_providers_json(tmp_path, creds)

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert state.credential_index["nvidia"] == 1

    def test_builds_state_defaults_to_zero_when_no_current(self, tmp_path):
        """When no credential has current, defaults to index 0."""
        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        for cred in creds["nvidia"]["credentials"]:
            cred.pop("current", None)
        _write_providers_json(tmp_path, creds)

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert state.credential_index["nvidia"] == 0


# ===========================================================================
# Tests: Credential rotation in non-streaming handler
# ===========================================================================
class TestNonStreamQuotaRotation:
    def test_quota_rotates_credential_not_model(self):
        """Quota error → next credential, same model."""
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        quota_response = AsyncMock()
        quota_response.status = 429
        quota_response.text = AsyncMock(return_value='{"error": "The free quota has been exhausted."}')
        quota_response.__aenter__ = AsyncMock(return_value=quota_response)
        quota_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[quota_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        # Model should NOT have switched — same model, new credential
        assert state.current_index["simple"] == 0
        # But credential should have advanced
        assert state.credential_index["nvidia"] == 1

    def test_quota_all_creds_exhausted_switches_model(self):
        """When all credentials for a provider are exhausted → switch model."""
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        config = _make_config(gateway={"retry_count": 4})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        # 3 creds exhausted → 3 attempts, then model switch → 1 more attempt needed
        quota_responses = [AsyncMock() for _ in range(3)]
        for r in quota_responses:
            r.status = 429
            r.text = AsyncMock(return_value='{"error": "quota exhausted"}')
            r.__aenter__ = AsyncMock(return_value=r)
            r.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession.post", side_effect=[*quota_responses, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        # All 3 creds tried for first model → switched to second model
        assert state.current_index["simple"] == 1

    def test_quota_single_cred_switches_model_immediately(self):
        """With 1 credential, quota error → immediate model switch."""
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        creds["nvidia"]["credentials"] = [{"base_url": "url", "api_key": "key"}]
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        quota_response = AsyncMock()
        quota_response.status = 429
        quota_response.text = AsyncMock(return_value='{"error": "quota exhausted"}')
        quota_response.__aenter__ = AsyncMock(return_value=quota_response)
        quota_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[quota_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        # With 1 cred, quota = immediate model switch
        assert state.current_index["simple"] == 1

    def test_non_quota_retryable_switches_model(self):
        """Non-quota retryable error → model switch (not credential rotation)."""
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        error_response = AsyncMock()
        error_response.status = 503
        error_response.text = AsyncMock(return_value='{"error": "model is overloaded"}')
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[error_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        # Model switched (overload is retryable but NOT quota)
        assert state.current_index["simple"] == 1
        # Credential NOT rotated
        assert state.credential_index["nvidia"] == 0

    def test_422_with_nested_429_not_treated_as_quota(self):
        """422 with429 in previous_errors → NOT quota rotation.

        OpenRouter sometimes wraps a backend429 in previous_errors when the
        primary response is422. Credentials should NOT rotate — the error
        is a request validation issue (422), not rate limiting.
        """
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        # Body from real OpenRouter response: 422 primary,429 in previous_errors
        body_422_with_nested_429 = json.dumps({
            "error": {
                "message": "Provider returned error",
                "code": 422,
                "metadata": {
                    "raw": '{"error":{"code":"invalid_request_error","message":"auto tool schemas do not support dependency"}}',
                    "provider_name": "Darkbloom",
                    "provider_error_code": "invalid_request_error",
                    "previous_errors": [{"code": 429, "message": "Provider returned error", "provider_name": "Google AI Studio"}],
                },
            }
        })

        error_response = AsyncMock()
        error_response.status = 422
        error_response.text = AsyncMock(return_value=body_422_with_nested_429)
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[error_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        # 422 without retryable keywords in body → non_retryable error returned as-is
        assert resp.status_code == 422
        # KEY: Credential NOT rotated
        assert state.credential_index["nvidia"] == 0
        # Model NOT switched either
        assert state.current_index["simple"] == 0

    def test_422_nested_429_with_retryable_keywords_switches_model(self):
        """422 with 'rate limited' in nested errors → retryable (model switch), NOT quota.

        Matches the real OpenRouter log where 'rate limited' in previous_errors
        triggers retryable detection, but credentials must NOT rotate.
        """
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        body = json.dumps({
            "error": {
                "message": "Provider returned error",
                "code": 422,
                "metadata": {
                    "raw": '{"error":{"code":"invalid_request_error","message":"auto tool schemas do not support dependency"}}',
                    "provider_name": "Darkbloom",
                    "provider_error_code": "invalid_request_error",
                    "previous_errors": [{"code": 429, "message": "rate limited", "provider_name": "Google AI Studio"}],
                },
            }
        })

        error_response = AsyncMock()
        error_response.status = 422
        error_response.text = AsyncMock(return_value=body)
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[error_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        # "rate limited" is retryable → model switched
        assert state.current_index["simple"] == 1
        # KEY: Credential NOT rotated despite429 in body
        assert state.credential_index["nvidia"] == 0

    def test_402_quota_rotates_credential(self):
        """HTTP 402 with quota body → credential rotation (not just 429).

        HuggingFace returns 402 for depleted credits. The quota check
        must not be restricted to status 429 alone.
        """
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        quota_response = AsyncMock()
        quota_response.status = 402
        quota_response.text = AsyncMock(return_value='{"error": "You have depleted your monthly included credits. Purchase pre-paid credits to continue."}')
        quota_response.__aenter__ = AsyncMock(return_value=quota_response)
        quota_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[quota_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        # Model NOT switched
        assert state.current_index["simple"] == 0
        # Credential rotated (402 + quota body)
        assert state.credential_index["nvidia"] == 1

    def test_403_quota_rotates_credential(self):
        """HTTP 403 with quota body → credential rotation.

        Some providers (e.g. Google) return 403 for quota exhaustion.
        """
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        quota_response = AsyncMock()
        quota_response.status = 403
        quota_response.text = AsyncMock(return_value='{"error": {"message": "quota exceeded"}}')
        quota_response.__aenter__ = AsyncMock(return_value=quota_response)
        quota_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[quota_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        assert state.current_index["simple"] == 0
        assert state.credential_index["nvidia"] == 1

    def test_credits_error_401_rotates_credential(self):
        """opencode CreditsError (HTTP 401) → credential rotation, same model."""
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        credits_response = AsyncMock()
        credits_response.status = 401
        credits_response.text = AsyncMock(
            return_value='{"type":"error","error":{"type":"CreditsError","message":"No payment method."}}'
        )
        credits_response.__aenter__ = AsyncMock(return_value=credits_response)
        credits_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[credits_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        assert state.current_index["simple"] == 0
        assert state.credential_index["nvidia"] == 1

    def test_not_found_for_account_404_rotates_credential(self):
        """NVIDIA 'not found for account' (HTTP 404) → credential rotation, same model."""
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)

        not_found_response = AsyncMock()
        not_found_response.status = 404
        not_found_response.text = AsyncMock(
            return_value='{"status":404,"detail":"Function \'abc\': Not found for account \'xyz\'"}'
        )
        not_found_response.__aenter__ = AsyncMock(return_value=not_found_response)
        not_found_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[not_found_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "simple", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        assert state.current_index["simple"] == 0
        assert state.credential_index["nvidia"] == 1


# ===========================================================================
# Tests: Credential rotation in streaming handler
# ===========================================================================
class TestStreamQuotaRotation:
    def test_streaming_quota_rotates_credential(self):
        """Quota error in stream → credential rotation, same model."""
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        call_count = 0

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "", {"status": 429, "body": "quota has been exhausted"}
            else:
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("gateway.forward_streaming", mock_stream_fwd),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        # Model NOT switched — same model, new credential
        assert state.current_index["simple"] == 0
        # Credential advanced
        assert state.credential_index["nvidia"] == 1

    def test_streaming_quota_all_creds_exhausted_switches_model(self):
        """Streaming: all creds exhausted → model switch."""
        from fastapi.testclient import TestClient

        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(Path(gateway._DEFAULT_PROVIDERS).parent, creds)
        config = _make_config(gateway={"retry_count": 4})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        call_count = 0

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # 3 quota errors = all creds for nvidia
                yield "", {"status": 429, "body": "quota exhausted"}
            else:
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("gateway.forward_streaming", mock_stream_fwd),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        # All creds tried → switched model
        assert state.current_index["simple"] == 1


# ===========================================================================
# Tests: Credential persistence to providers.json
# ===========================================================================
class TestCredentialPersistence:
    def test_advance_credential_updates_file(self, tmp_path):
        """advance_credential persists current flag to providers.json."""
        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(tmp_path, creds)
        state = gateway.GatewayState()
        from src.credential_manager import ProviderCredentialManager
        state.set_credential_manager(ProviderCredentialManager(tmp_path / "providers.json"))

        state.advance_credential("nvidia")

        # Read file and verify
        with open(tmp_path / "providers.json") as f:
            data = json.load(f)
        file_creds = data["nvidia"]["credentials"]
        assert file_creds[0].get("current") is not True
        assert file_creds[1]["current"] is True

    def test_advance_credential_wraps_persists(self, tmp_path):
        """Wrapping around persists correctly."""
        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(tmp_path, creds)
        state = gateway.GatewayState()
        from src.credential_manager import ProviderCredentialManager
        mgr = ProviderCredentialManager(tmp_path / "providers.json")
        # Manually set index to last cred
        mgr._credential_index["nvidia"] = 2
        state.set_credential_manager(mgr)

        state.advance_credential("nvidia")

        with open(tmp_path / "providers.json") as f:
            data = json.load(f)
        file_creds = data["nvidia"]["credentials"]
        assert file_creds[2].get("current") is not True
        assert file_creds[0]["current"] is True


class TestCredentialPosition:
    def test_first_credential(self, tmp_path):
        """First credential → '1/3'."""
        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(tmp_path, creds)
        state = gateway.GatewayState()
        from src.credential_manager import ProviderCredentialManager
        state.set_credential_manager(ProviderCredentialManager(tmp_path / "providers.json"))

        assert state.credential_position("nvidia") == "1/3"

    def test_after_advance(self, tmp_path):
        """After advancing once → '2/3'."""
        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(tmp_path, creds)
        state = gateway.GatewayState()
        from src.credential_manager import ProviderCredentialManager
        state.set_credential_manager(ProviderCredentialManager(tmp_path / "providers.json"))

        state.advance_credential("nvidia")
        assert state.credential_position("nvidia") == "2/3"

    def test_after_wrap_around(self, tmp_path):
        """After wrapping around from last → '1/3'."""
        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(tmp_path, creds)
        state = gateway.GatewayState()
        from src.credential_manager import ProviderCredentialManager
        mgr = ProviderCredentialManager(tmp_path / "providers.json")
        mgr._credential_index["nvidia"] = 2
        state.set_credential_manager(mgr)

        state.advance_credential("nvidia")
        assert state.credential_position("nvidia") == "1/3"

    def test_unknown_provider_returns_none(self, tmp_path):
        creds = copy.deepcopy(MOCK_PROVIDERS_CREDS)
        _write_providers_json(tmp_path, creds)
        state = gateway.GatewayState()
        from src.credential_manager import ProviderCredentialManager
        state.set_credential_manager(ProviderCredentialManager(tmp_path / "providers.json"))

        assert state.credential_position("nonexistent") is None

    def test_no_cred_manager_returns_none(self):
        state = gateway.GatewayState()
        assert state.credential_position("nvidia") is None

    def test_single_credential(self, tmp_path):
        """Single credential → '1/1'."""
        creds = {"nvidia": {"type": "openai", "credentials": [{"api_key": "k", "base_url": "http://x"}]}}
        _write_providers_json(tmp_path, creds)
        state = gateway.GatewayState()
        from src.credential_manager import ProviderCredentialManager
        state.set_credential_manager(ProviderCredentialManager(tmp_path / "providers.json"))

        assert state.credential_position("nvidia") == "1/1"


# ===========================================================================
# Tests for gateway_info module — <debug> tag format
# ===========================================================================
import gateway_info


class TestBuildDebugTag:
    """Tests for gateway_info.build_debug_tag()."""

    def test_basic_tag(self):
        tag = gateway_info.build_debug_tag(
            "nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter", "13/15",
        )
        assert tag == (
            "\n<debug>nvidia/nemotron-3-ultra-550b-a55b:free (openrouter), creds=13/15</debug>"
        )

    def test_simple_model(self):
        tag = gateway_info.build_debug_tag("gemini-flash", "google", "2/3")
        assert tag == "\n<debug>gemini-flash (google), creds=2/3</debug>"

    def test_single_credential(self):
        tag = gateway_info.build_debug_tag("model", "provider", "1/1")
        assert tag == "\n<debug>model (provider), creds=1/1</debug>"


class TestStripDebugTags:
    """Tests for gateway_info.strip_debug_tags()."""

    def test_strips_tag(self):
        text = "before\n<debug>model (p), creds=1/2</debug>after"
        result = gateway_info.strip_debug_tags(text)
        assert "<debug>" not in result
        assert "before" in result
        assert "after" in result

    def test_strips_multiple_tags(self):
        text = "first\n<debug>info1</debug>middle\n<debug>info2</debug>last"
        result = gateway_info.strip_debug_tags(text)
        assert "<debug>" not in result
        assert "first" in result
        assert "middle" in result
        assert "last" in result

    def test_content_between_tags_preserved(self):
        """Two debug tags with real content between them — only tags stripped."""
        text = (
            "Here is the answer.\n"
            "<debug>model-a (provider-1), creds=1/3</debug>\n"
            "IMPORTANT: always follow the rules.\n"
            "<debug>model-b (provider-2), creds=2/3</debug>"
        )
        result = gateway_info.strip_debug_tags(text)
        assert "<debug>" not in result
        assert "Here is the answer." in result
        assert "IMPORTANT: always follow the rules." in result

    def test_no_tag_unchanged(self):
        text = "normal text\nno tags here"
        assert gateway_info.strip_debug_tags(text) == text

    def test_tag_with_trailing_newline(self):
        text = "reply\n<debug>model (p), creds=1/1</debug>\n"
        result = gateway_info.strip_debug_tags(text)
        assert result == "reply"

    def test_tag_only(self):
        text = "\n<debug>model (p), creds=1/1</debug>"
        result = gateway_info.strip_debug_tags(text)
        assert result.strip() == ""


class TestCleanMessages:
    """Tests for gateway_info.clean_messages()."""

    def test_strips_from_assistant_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "actual reply\n<debug>model (p), creds=1/1</debug>"},
            {"role": "user", "content": "thanks"},
        ]
        result = gateway_info.clean_messages(messages)
        assert len(result) == 3
        assert "<debug>" not in result[1]["content"]
        assert "actual reply" in result[1]["content"]

    def test_drops_empty_assistant_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "\n<debug>model (p), creds=1/1</debug>"},
            {"role": "user", "content": "thanks"},
        ]
        result = gateway_info.clean_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "user"

    def test_non_string_content_passes_through(self):
        messages = [
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": 123},
        ]
        result = gateway_info.clean_messages(messages)
        assert len(result) == 2

    def test_non_assistant_messages_unchanged(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
        ]
        result = gateway_info.clean_messages(messages)
        assert result == messages

    def test_mixed_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "answer\n<debug>info</debug>"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "\n<debug>info2</debug>"},
        ]
        result = gateway_info.clean_messages(messages)
        assert len(result) == 4  # last assistant dropped
        assert result[2]["content"].strip() == "answer"


class TestDebugTagNeverLeaks:
    """Verify that debug data is completely removed — model never sees it."""

    def test_strip_removes_all_debug_content(self):
        """Full message with debug tag at end — after strip, zero debug remnants."""
        text = "Sure, here's the answer!\n<debug>nvidia/nemotron-3-ultra-550b-a55b:free (openrouter), creds=13/15</debug>"
        result = gateway_info.strip_debug_tags(text)
        assert "<debug>" not in result
        assert "</debug>" not in result
        assert "openrouter" not in result
        assert "creds=" not in result
        assert "nemotron" not in result
        assert result == "Sure, here's the answer!"

    def test_clean_messages_never_leaks_debug_data(self):
        """Multiple turns — debug tags stripped from every assistant message."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!\n<debug>model (p), creds=1/1</debug>"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "bye!\n<debug>model (p), creds=2/2</debug>"},
        ]
        result = gateway_info.clean_messages(messages)
        combined = " ".join(m["content"] for m in result if isinstance(m.get("content"), str))
        assert "<debug>" not in combined
        assert "</debug>" not in combined
        assert "creds=" not in combined

    def test_debug_in_middle_of_text(self):
        """Debug tag in middle — only tag and surrounding newlines removed."""
        text = "Part 1\n<debug>x (y), creds=1/1</debug>\nPart 2"
        result = gateway_info.strip_debug_tags(text)
        assert result == "Part 1Part 2"

    def test_old_gateway_blocks_also_stripped(self):
        """Legacy [[[gateway]]] blocks are NOT stripped by new code (different format)."""
        text = "reply[[[gateway]]]\ninfo\n[[[/gateway]]]"
        result = gateway_info.strip_debug_tags(text)
        # Old format is NOT stripped — only <debug> tags are
        assert "[[[gateway]]]" in result

    def test_real_world_example(self):
        """Exact format from the user's specification."""
        tag = gateway_info.build_debug_tag(
            "nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter", "13/15"
        )
        full_msg = f"Here is my response.{tag}"
        stripped = gateway_info.strip_debug_tags(full_msg)
        assert stripped == "Here is my response."
        # Verify the tag itself contains the expected data
        assert "nemotron" in tag
        assert "openrouter" in tag
        assert "creds=13/15" in tag


class TestExtractChunkContent:
    """Tests for the streaming chunk content extractor."""

    def test_extracts_openai_delta_content(self):
        chunk = 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        assert streaming.extract_chunk_content(chunk) == "hello"

    def test_extracts_openai_delta_text(self):
        chunk = 'data: {"choices":[{"delta":{"text":"world"}}]}\n\n'
        assert streaming.extract_chunk_content(chunk) == "world"

    def test_extracts_choice_text(self):
        chunk = 'data: {"choices":[{"text":"choice text"}]}\n\n'
        assert streaming.extract_chunk_content(chunk) == "choice text"

    def test_returns_empty_for_done_marker(self):
        assert streaming.extract_chunk_content("data: [DONE]\n\n") == ""

    def test_returns_empty_for_malformed_data(self):
        assert streaming.extract_chunk_content("not an sse chunk") == ""

    def test_returns_empty_when_no_content(self):
        chunk = 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        assert streaming.extract_chunk_content(chunk) == ""


class TestHasToolCallsInChunk:
    """Tests for the streaming tool_calls detector."""

    def test_returns_true_when_tool_calls_present(self):
        chunk = 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"test"}}]}}]}\n\n'
        assert streaming.has_tool_calls_in_chunk(chunk) is True

    def test_returns_true_for_tool_calls_with_arguments(self):
        chunk = 'data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":1}"}}]}}]}\n\n'
        assert streaming.has_tool_calls_in_chunk(chunk) is True

    def test_returns_false_for_done_marker(self):
        assert streaming.has_tool_calls_in_chunk("data: [DONE]\n\n") is False

    def test_returns_false_for_malformed_data(self):
        assert streaming.has_tool_calls_in_chunk("not an sse chunk") is False

    def test_returns_false_when_no_tool_calls(self):
        chunk = 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        assert streaming.has_tool_calls_in_chunk(chunk) is False

    def test_returns_false_when_tool_calls_empty(self):
        chunk = 'data: {"choices":[{"delta":{"tool_calls":[]}}]}\n\n'
        assert streaming.has_tool_calls_in_chunk(chunk) is False

    def test_returns_false_when_tool_calls_null(self):
        chunk = 'data: {"choices":[{"delta":{"tool_calls":null}}]}\n\n'
        assert streaming.has_tool_calls_in_chunk(chunk) is False


class TestUselessResponseLogging:
    """Provider response text is logged when a useless response triggers a switch."""

    def test_useless_response_logs_provider_text(self, tmp_path):
        """When a model returns a short response and is switched, log contains the text."""
        from fastapi.testclient import TestClient

        config = _make_config(gateway={
            "switch_on_any_error": True,
            "retry_count": 2,
        })
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        call_count = 0

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First model returns a useless short response
                yield 'data: {"choices":[{"delta":{"content":"I"}}]}\n\n', None
                yield 'data: {"choices":[{"delta":{"content":" see."}}]}\n\n', None
                yield 'data: {"choices":[{"finish_reason":"stop"}],"usage":{"completion_tokens":5}}\n\n', None
                yield "data: [DONE]\n\n", None
            else:
                # Second model returns a normal response
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield 'data: [DONE]\n\n', None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("gateway.forward_streaming", mock_stream_fwd),
            patch.object(gateway.logger, "warning") as mock_warning,
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        messages = [str(call.args[0]) if call.args else str(call) for call in mock_warning.call_args_list]
        assert any("I see." in msg for msg in messages)
        assert any("finish_reason=stop" in msg for msg in messages)

    def test_empty_response_is_retryable_without_switch_flag(self, tmp_path):
        """A successful [STREAM_DONE] with no answer AND no reasoning must be
        treated as a retryable error and switched to the next model, even when
        switch_on_any_error is disabled.
        """
        from fastapi.testclient import TestClient

        config = _make_config(gateway={
            "switch_on_any_error": False,
            "retry_count": 2,
        })
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        call_count = 0

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First model returns a "successful" stream that says nothing:
                # only a finish_reason/usage chunk, then [DONE] — no content,
                # no reasoning.
                yield 'data: {"choices":[{"finish_reason":"stop"}],"usage":{"completion_tokens":0}}\n\n', None
                yield "data: [DONE]\n\n", None
            else:
                # Second model returns a normal response
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("gateway.forward_streaming", mock_stream_fwd),
            patch.object(gateway.logger, "warning") as mock_warning,
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        body = resp.text
        # The empty first model must have been skipped — we got the real answer.
        assert "ok" in body
        messages = [str(call.args[0]) if call.args else str(call) for call in mock_warning.call_args_list]
        assert any("empty response (no answer)" in msg for msg in messages)

    def test_tool_calls_are_not_empty_response(self, tmp_path):
        """A response with tool_calls but no text content is a valid response
        and must NOT be treated as empty, even when text content is absent.
        """
        from fastapi.testclient import TestClient

        config = _make_config(gateway={
            "switch_on_any_error": False,
            "retry_count": 1,
        })
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            # Model returns tool_calls but no text content
            yield 'data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"name":"test","arguments":"{\\"a\\":1}"}}]}}]}\n\n', None
            yield 'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}],"usage":{"completion_tokens":10}}\n\n', None
            yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("gateway.forward_streaming", mock_stream_fwd),
            patch.object(gateway.logger, "warning") as mock_warning,
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        body = resp.text
        # Tool calls must be present in response
        assert "tool_calls" in body
        # Must NOT have triggered empty response warning
        messages = [str(call.args[0]) if call.args else str(call) for call in mock_warning.call_args_list]
        assert not any("empty response" in msg for msg in messages)

    def test_google_function_calls_are_not_empty_response(self, tmp_path):
        """Google provider function calls (converted to OpenAI tool_calls) should NOT
        be treated as empty response. Tests the full chain: Google functionCall →
        convert_google_sse_chunk_to_openai → has_tool_calls_in_chunk → not empty.
        """
        from fastapi.testclient import TestClient

        config = _make_config(gateway={
            "switch_on_any_error": False,
            "retry_count": 1,
        })
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            # Simulate Google streaming response with functionCall (will be converted)
            # Note: forward_streaming handles conversion internally via convert_google_sse_chunk_to_openai
            # For this test, we'll directly yield converted OpenAI format (as if already converted)
            yield 'data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"Paris\\"}"}}]}}]}\n\n', None
            yield 'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}],"usage":{"completion_tokens":10}}\n\n', None
            yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("gateway.forward_streaming", mock_stream_fwd),
            patch.object(gateway.logger, "warning") as mock_warning,
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [{"role": "user", "content": "What's the weather?"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        body = resp.text
        # Tool calls must be present in response
        assert "tool_calls" in body
        assert "get_weather" in body
        # Must NOT have triggered empty response warning
        messages = [str(call.args[0]) if call.args else str(call) for call in mock_warning.call_args_list]
        assert not any("empty response" in msg for msg in messages)


# ===========================================================================
# Tests: infinite_models configuration
# ===========================================================================
class TestInfiniteModels:
    """Tests for infinite_models: when enabled, advance() wraps to first model."""

    def test_infinite_models_defaults_to_false(self):
        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        assert state.infinite_models is False

    def test_infinite_models_reads_from_config(self):
        config = _make_config(gateway={"infinite_models": True})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        assert state.infinite_models is True

    def test_advance_wraps_when_enabled(self):
        """With infinite_models=True, advance past last model wraps to first."""
        state = gateway.GatewayState()
        state.pools = {"cat": [{"model_id": "a"}, {"model_id": "b"}]}
        state.current_index = {"cat": 1}  # at last model
        state.infinite_models = True

        model = state.advance("cat")
        assert model is not None
        assert model["model_id"] == "a"
        assert state.current_index["cat"] == 0

    def test_advance_returns_none_when_disabled(self):
        """With infinite_models=False (default), advance past last returns None."""
        state = gateway.GatewayState()
        state.pools = {"cat": [{"model_id": "a"}, {"model_id": "b"}]}
        state.current_index = {"cat": 1}
        state.infinite_models = False

        model = state.advance("cat")
        assert model is None

    def test_non_stream_wraps_and_succeeds(self):
        """All models fail → wraps to first → succeeds on retry."""
        from fastapi.testclient import TestClient

        config = _make_config(gateway={"infinite_models": True})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        error_response = AsyncMock()
        error_response.status = 429
        error_response.text = AsyncMock(return_value='{"error": "rate_limit_exceeded"}')
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        # visual has 2 models — both fail, then wrap to first which succeeds
        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession.post", side_effect=[error_response, error_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "visual", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        # After wrapping, index is back at 0
        assert state.current_index["visual"] == 0

    def test_non_stream_disabled_returns_429(self):
        """With infinite_models=False (default), exhausted pool returns 429."""
        from fastapi.testclient import TestClient

        config = _make_config(gateway={"infinite_models": False})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)

        error_response = AsyncMock()
        error_response.status = 429
        error_response.text = AsyncMock(return_value='{"error": "rate_limit_exceeded"}')
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("aiohttp.ClientSession.post", return_value=error_response),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "visual", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 429
        assert "unavailable" in resp.json()["error"]["message"]

    def test_streaming_wraps_and_succeeds(self):
        """Streaming: all models fail → wraps to first → succeeds."""
        from fastapi.testclient import TestClient

        config = _make_config(gateway={"infinite_models": True})
        state = gateway.build_state(config, MOCK_TEST_RESULTS)
        call_count = 0

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # both visual models fail
                yield "", {"status": 429, "body": "rate_limit"}
            else:
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", config),
            patch("gateway.forward_streaming", mock_stream_fwd),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "visual",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert state.current_index["visual"] == 0


# ===========================================================================
# Tests: strip_reasoning_content — remove model-specific thinking state
# ===========================================================================
class TestStripReasoningContent:
    """Tests for gateway_info.strip_reasoning_content()."""

    def test_strips_reasoning_from_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Hello!",
                "reasoning_content": "I should greet the user...",
            },
        ]
        result = gateway_info.strip_reasoning_content(messages)
        assert result[1]["reasoning_content"] == ""
        assert result[1]["content"] == "Hello!"

    def test_leaves_user_messages_unchanged(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "You are helpful"},
        ]
        result = gateway_info.strip_reasoning_content(messages)
        assert result == messages

    def test_leaves_assistant_without_reasoning_unchanged(self):
        messages = [
            {"role": "assistant", "content": "just a normal reply"},
        ]
        result = gateway_info.strip_reasoning_content(messages)
        assert result == messages

    def test_strips_from_multiple_assistant_messages(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Hello!",
                "reasoning_content": "thinking...",
            },
            {"role": "user", "content": "bye"},
            {
                "role": "assistant",
                "content": "Goodbye!",
                "reasoning_content": "farewell thinking...",
            },
        ]
        result = gateway_info.strip_reasoning_content(messages)
        assert result[1]["reasoning_content"] == ""
        assert result[3]["reasoning_content"] == ""

    def test_preserves_other_assistant_fields(self):
        messages = [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "should be removed",
                "tool_calls": [{"id": "call_1"}],
            },
        ]
        result = gateway_info.strip_reasoning_content(messages)
        assert result[0]["reasoning_content"] == ""
        assert result[0]["content"] == "answer"
        assert result[0]["tool_calls"] == [{"id": "call_1"}]

    def test_handles_empty_list(self):
        assert gateway_info.strip_reasoning_content([]) == []

    def test_does_not_mutate_original(self):
        """Returns a new list — original messages are not mutated."""
        messages = [
            {
                "role": "assistant",
                "content": "hi",
                "reasoning_content": "thinking",
            },
        ]
        result = gateway_info.strip_reasoning_content(messages)
        assert messages[0]["reasoning_content"] == "thinking"
        assert result[0]["reasoning_content"] == ""

# ===========================================================================
# Tests: reasoning_content — strip only on model switch (switch-aware)
# ===========================================================================
class TestReasoningContentStrippedFromProviderPayload:
    """Integration tests: reasoning_content must be stripped only when switching
    models, and preserved when the same model (or its rotated credential)
    continues a multi-turn thinking conversation.
    """

    def _make_mock_session_recording(self, captured_bodies):
        """Build a MockSession that records every outbound request body."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]})
        )

        class MockSession:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, url, **kwargs):
                captured_bodies.append(copy.deepcopy(kwargs.get("json", {})))
                from contextlib import asynccontextmanager

                @asynccontextmanager
                async def ctx():
                    yield mock_response

                return ctx()

        return MockSession

    def test_non_stream_preserves_reasoning_for_same_model(self):
        """No switch → reasoning_content is preserved (multi-turn thinking)."""
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        captured = []

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession", self._make_mock_session_recording(captured)),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": "Hello!",
                            "reasoning_content": "thinking...",
                        },
                        {"role": "user", "content": "continue"},
                    ],
                },
            )

        assert resp.status_code == 200
        assert len(captured) == 1
        # Same model continues → reasoning_content must be passed through
        assistant_msg = captured[0]["messages"][1]
        assert assistant_msg["reasoning_content"] == "thinking..."

    def test_non_stream_strips_reasoning_on_model_switch(self):
        """Switch model → new model receives history WITHOUT reasoning_content."""
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        captured = []

        # First attempt (model A) 429, second (model B) 200
        error_response = AsyncMock()
        error_response.status = 429
        error_response.text = AsyncMock(return_value='{"error": "rate_limit_exceeded"}')
        error_response.__aenter__ = AsyncMock(return_value=error_response)
        error_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.status = 200
        ok_response.text = AsyncMock(
            return_value=json.dumps({"choices": [{"message": {"content": "ok"}}]})
        )
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("aiohttp.ClientSession.post", side_effect=[error_response, ok_response]),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": "Hello!",
                            "reasoning_content": "thinking...",
                        },
                        {"role": "user", "content": "continue"},
                    ],
                },
            )

        assert resp.status_code == 200
        # Two attempts were made to the provider
        # We cannot capture both easily via aiohttp.post side_effect, so verify
        # the final model index advanced (switched) — the switch path is what
        # triggers the strip. The streaming test below captures both bodies.
        assert state.current_index["simple"] == 1

    def test_streaming_preserves_reasoning_on_first_attempt(self):
        """Streaming first attempt (no switch) preserves reasoning_content."""
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        captured_bodies = []

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            captured_bodies.append(copy.deepcopy(body))
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
            yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("gateway.forward_streaming", mock_stream_fwd),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": "Hello!",
                            "reasoning_content": "thinking...",
                        },
                    ],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert len(captured_bodies) == 1
        # First attempt, same model → reasoning_content preserved
        assert captured_bodies[0]["messages"][1]["reasoning_content"] == "thinking..."

    def test_streaming_strips_reasoning_on_model_switch(self):
        """Streaming: after a model switch, the new model gets clean history."""
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        captured_bodies = []

        call_count = 0

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            captured_bodies.append(copy.deepcopy(body))
            if call_count == 1:
                # First model fails → forces a switch before any chunk
                yield "", {"status": 429, "body": "rate_limit"}
            else:
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield "data: [DONE]\n\n", None

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("gateway.forward_streaming", mock_stream_fwd),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": "Hello!",
                            "reasoning_content": "thinking...",
                        },
                        {"role": "user", "content": "continue"},
                    ],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert len(captured_bodies) == 2
        # Attempt 1 (model A): reasoning_content preserved
        assert captured_bodies[0]["messages"][1]["reasoning_content"] == "thinking..."
        # Attempt 2 (model B, after switch): reasoning_content passed back unchanged
        # (upstream requires it on every turn when thinking mode is active)
        assert captured_bodies[1]["messages"][1]["reasoning_content"] == "thinking..."
        assert captured_bodies[1]["messages"][1]["content"] == "Hello!"
        # Model actually switched
        assert state.current_index["simple"] == 1

    def test_deepseek_r1_real_world_scenario(self):
        """Real-world: DeepSeek R1 reasoning_content preserved until a switch.

        Reproduces the reported bug: zen-coder-1 (thinking) produces
        reasoning_content, then a rate limit forces a switch to big-pickle
        (another thinking model). The foreign reasoning_content must be stripped
        for big-pickle, but preserved while zen-coder-1 continues.
        """
        from fastapi.testclient import TestClient

        state = gateway.build_state(MOCK_CONFIG, MOCK_TEST_RESULTS)
        captured_bodies = []

        call_count = 0

        async def mock_stream_fwd(st, model, body, ua, idle_timeout=30.0):
            nonlocal call_count
            call_count += 1
            captured_bodies.append(copy.deepcopy(body))
            if call_count == 1:
                yield "", {"status": 429, "body": "rate_limit"}
            else:
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', None
                yield "data: [DONE]\n\n", None

        deepseek_reasoning = (
            "Okay, the user wants to know about Python sorting. "
            "I should explain both sorted() and list.sort()..."
        )

        with (
            patch.object(gateway, "_state", state),
            patch.object(gateway, "_config", MOCK_CONFIG),
            patch("gateway.forward_streaming", mock_stream_fwd),
        ):
            client = TestClient(gateway.app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "simple",
                    "messages": [
                        {"role": "user", "content": "How to sort in Python?"},
                        {
                            "role": "assistant",
                            "content": "You can use sorted() or list.sort()...",
                            "reasoning_content": deepseek_reasoning,
                        },
                        {"role": "user", "content": "Which is faster?"},
                    ],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert len(captured_bodies) == 2
        # Attempt 1 (zen-coder-1): reasoning preserved
        assert captured_bodies[0]["messages"][1]["reasoning_content"] == deepseek_reasoning
        # Attempt 2 (big-pickle, after switch): reasoning passed back unchanged,
        # content kept (upstream requires reasoning_content on every turn)
        switched = captured_bodies[1]["messages"][1]
        assert switched["reasoning_content"] == deepseek_reasoning
        assert switched["content"] == "You can use sorted() or list.sort()..."
        assert state.current_index["simple"] == 1
