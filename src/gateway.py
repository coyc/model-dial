#!/usr/bin/env python3
"""
Transparent proxy gateway that routes requests to the fastest available model
from each category. Categories are defined in config.json → requirements.

Only modifies:
  - Model name (virtual → actual provider model ID)
  - Error handling (retryable errors → switch model)

Everything else is passed through as-is.
"""

import argparse
import asyncio
import atexit
import json
import logging
import os
import random
import sys
import time
import uuid

from credential_manager import ProviderCredentialManager
from error_detection import is_quota_error, is_retryable_error
from error_response import extract_error_message
from convert_google import GoogleConverter
from streaming import parse_chunk_metadata, extract_chunk_content, has_tool_calls_in_chunk, is_useless_response, parse_sse_error, forward_streaming
from gateway_info import build_debug_tag, clean_messages
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.json"
_DEFAULT_TEST_RESULTS = _PROJECT_ROOT / "result-test.json"
_LOG_DIR = _PROJECT_ROOT / "logs"
_PID_PATH = _LOG_DIR / "gateway.pid"
_CURRENT_MODEL_PATH = _LOG_DIR / "current-model.json"
_DEFAULT_PROVIDERS = _PROJECT_ROOT / "providers.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("gateway")


def setup_logging(log_dir: Path | None = None) -> None:
    """Configure rotating log: today + yesterday only."""
    d = log_dir or _LOG_DIR
    d.mkdir(parents=True, exist_ok=True)

    handler = TimedRotatingFileHandler(
        d / "gateway.log",
        when="midnight",
        interval=1,
        backupCount=1,
    )
    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)


# ---------------------------------------------------------------------------
# Gateway state
# ---------------------------------------------------------------------------
class GatewayState:
    """Holds model pools, current positions per category, and credential rotation state."""

    def __init__(self) -> None:
        self.pools: dict[str, list[dict]] = {}  # category → sorted model list
        self.current_index: dict[str, int] = {}  # category → index
        self._cred_manager: ProviderCredentialManager | None = None
        self.switch_on_any_error: bool = False
        self.show_debug_tag: bool = False
        self.infinite_models: bool = False
        self.request_timeout_ms: int = 60000
        self.stream_idle_timeout_ms: int = 30000
        # Per-category count of consecutive quota failures for the CURRENT model.
        # Used to detect that every credential for a model is exhausted so we can
        # advance to the next model. Lives on the process-wide state singleton,
        # so it survives across the requests a client makes while retrying.
        self._cred_tried: dict[str, int] = {}

    @property
    def providers_creds(self) -> dict[str, dict]:
        """Provider config dict — read-only proxy to credential manager."""
        if self._cred_manager is None:
            return {}
        return self._cred_manager.providers_creds

    @providers_creds.setter
    def providers_creds(self, value: dict) -> None:
        """Allow setting providers_creds directly (for tests and backward compat).

        Creates a ProviderCredentialManager in-memory from the given dict.
        No file persistence — use set_credential_manager() for real usage.
        """
        if self._cred_manager is None:
            # Create a dummy manager (won't persist since no path)
            self._cred_manager = ProviderCredentialManager.__new__(ProviderCredentialManager)
            self._cred_manager._providers_path = Path("/dev/null")
            self._cred_manager._providers_creds = value
            self._cred_manager._credential_index = {}
        else:
            self._cred_manager._providers_creds = value
            self._cred_manager._credential_index = {}

    @property
    def credential_index(self) -> dict[str, int]:
        """Credential index per provider — read-only proxy to credential manager."""
        if self._cred_manager is None:
            return {}
        return self._cred_manager.credential_index

    @credential_index.setter
    def credential_index(self, value: dict[str, int]) -> None:
        """Allow setting credential_index directly (for tests and backward compat)."""
        if self._cred_manager is None:
            self._cred_manager = ProviderCredentialManager.__new__(ProviderCredentialManager)
            self._cred_manager._providers_path = Path("/dev/null")
            self._cred_manager._providers_creds = {}
            self._cred_manager._credential_index = value
        else:
            self._cred_manager._credential_index = value

    def set_credential_manager(self, manager: ProviderCredentialManager) -> None:
        """Set the credential manager (production path)."""
        self._cred_manager = manager

    def get_model(self, category: str) -> dict | None:
        """Get current model for category, or None if exhausted."""
        pool = self.pools.get(category, [])
        idx = self.current_index.get(category, 0)
        if idx >= len(pool):
            return None
        return pool[idx]

    def advance(self, category: str) -> dict | None:
        """Move to next model in category. Returns new model or None.
        
        When wrap_models is enabled and the pool is exhausted, wraps around
        to the first model instead of returning None.
        """
        pool = self.pools.get(category, [])
        if not pool:
            return None
        idx = self.current_index.get(category, 0) + 1
        if idx >= len(pool):
            if self.infinite_models:
                idx = 0
            else:
                self.current_index[category] = idx
                # A model change invalidates the per-model credential-exhaustion counter.
                self._cred_tried[category] = 0
                return None
        self.current_index[category] = idx
        # A model change invalidates the per-model credential-exhaustion counter.
        self._cred_tried[category] = 0
        return self.get_model(category)

    def credential_count(self, provider_name: str) -> int:
        """Return the number of credentials configured for a provider."""
        if self._cred_manager is None:
            return 0
        return self._cred_manager.credential_count(provider_name)

    def reset_cred_tried(self, category: str) -> None:
        """Reset the per-model quota-failure counter for a category."""
        self._cred_tried[category] = 0

    def inc_cred_tried(self, category: str) -> None:
        """Record one more quota failure for the current model of a category."""
        self._cred_tried[category] = self._cred_tried.get(category, 0) + 1

    def get_cred_tried(self, category: str) -> int:
        """Return how many credentials have failed for the current model."""
        return self._cred_tried.get(category, 0)

    def available_categories(self) -> list[str]:
        """Return categories that have at least one model."""
        return [cat for cat, pool in self.pools.items() if pool]

    def get_credential(self, provider_name: str) -> dict | None:
        """Get current credential (base_url, api_key) for provider."""
        if self._cred_manager is None:
            return None
        return self._cred_manager.get_credential(provider_name)

    def advance_credential(self, provider_name: str) -> dict | None:
        """Move to next credential for provider, wrap around. Persists to file."""
        if self._cred_manager is None:
            return None
        return self._cred_manager.advance_credential(provider_name)

    def credential_position(self, provider_name: str) -> str | None:
        """Return 1-based position string like "2/5" or None if no creds."""
        if self._cred_manager is None:
            return None
        total = self._cred_manager.credential_count(provider_name)
        if total == 0:
            return None
        idx = self.credential_index.get(provider_name, 0)
        return f"{idx + 1}/{total}"

    # ------------------------------------------------------------------
    # Provider URL / header helpers (used by streaming.py forward_streaming)
    # ------------------------------------------------------------------
    def _resolve_provider(self, provider_name: str) -> dict | None:
        """Get provider config with current credential applied."""
        prov_data = self.providers_creds.get(provider_name)
        if not prov_data:
            return None
        cred = self.get_credential(provider_name)
        if cred:
            return {**prov_data, "base_url": cred["base_url"], "api_key": cred["api_key"]}
        return prov_data

    @staticmethod
    def _get_provider_url(provider: dict, model_id: str) -> str:
        """Get the full URL for a request."""
        base = provider["base_url"].rstrip("/")
        ptype = provider.get("type", "openai")
        if ptype == "google":
            return f"{base}/models/{model_id}:streamGenerateContent?alt=sse"
        return f"{base}/chat/completions"

    @staticmethod
    def _get_provider_headers(provider: dict, user_agent: str) -> dict:
        """Get headers for provider request."""
        ptype = provider.get("type", "openai")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        }
        if ptype == "google":
            headers["x-goog-api-key"] = provider["api_key"]
        else:
            headers["Authorization"] = f"Bearer {provider['api_key']}"
        return headers


# ---------------------------------------------------------------------------
# Current model persistence (logs/current-model.json)
# ---------------------------------------------------------------------------
def save_current_model(state: GatewayState, path: Path | None = None) -> None:
    """Save full model pool per category to logs/current-model.json.

    For each category stores the complete sorted/selected model list
    (provider, model_id, ttft_ms, total_ms) with the current model marked by
    ``current: true``. This makes it possible to see at a glance how
    the gateway filtered and sorted models from result-test.json.
    """
    target = path or _CURRENT_MODEL_PATH
    data: dict[str, dict] = {}
    for cat in state.pools:
        pool = state.pools[cat]
        idx = state.current_index.get(cat, 0)
        models_list = []
        for i, m in enumerate(pool):
            entry: dict = {
                "provider": m["provider"],
                "model_id": m["model_id"],
                "ttft_ms": m.get("test", {}).get("ttft_ms"),
                "total_ms": m.get("test", {}).get("total_ms"),
            }
            if i == idx:
                entry["current"] = True
            models_list.append(entry)
        data[cat] = {"models": models_list}
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(data, f, indent=2)


def load_current_model(path: Path | None = None) -> dict:
    """Load saved current model from logs/current-model.json.

    Format: ``{"simple": {"models": [{"provider": ..., "model_id": ..., "current": true}, ...]}}``

    Returns ``{cat: {"provider": ..., "model_id": ...}}`` (the current model
    per category) or empty dict.
    """
    target = path or _CURRENT_MODEL_PATH
    if not target.exists():
        return {}
    try:
        with open(target) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    result: dict[str, dict[str, str]] = {}
    for cat, cat_data in data.items():
        if not isinstance(cat_data, dict):
            continue
            # New format: {"models": [{"provider": ..., "model_id": ..., "current": true}, ...]}
        if "models" in cat_data:
            for m in cat_data["models"]:
                if m.get("current"):
                    result[cat] = {"provider": m["provider"], "model_id": m["model_id"]}
                    break
            else:
                # No explicit current → first model is current
                if cat_data["models"]:
                    result[cat] = {
                        "provider": cat_data["models"][0]["provider"],
                        "model_id": cat_data["models"][0]["model_id"],
                    }
    return result


def load_config(path: Path | None = None) -> dict:
    """Load config.json."""
    src = path or _DEFAULT_CONFIG
    with open(src) as f:
        return json.load(f)


def load_test_results(path: Path | None = None) -> dict:
    """Load result-test.json.

    Expected format (from test-models.py):
      {
        "results": [
          {
            "provider": "...",
            "model_id": "...",
            "rejected": false,
            "requirements_breakdown": {"simple": true, "coder": false, ...},
            "test": {
              "status": "ok",
              "ttft_ms": 123,
              "total_ms": 456,
              "correct": true,
              ...
            },
            "capabilities": {...}
          },
          ...
        ]
      }
    """
    src = path or _DEFAULT_TEST_RESULTS
    with open(src) as f:
        return json.load(f)


def load_providers_config(path: Path | None = None) -> ProviderCredentialManager:
    """Create a ProviderCredentialManager from providers.json.

    Returns a manager instance. If the file doesn't exist, returns a manager
    backed by a non-existent path (empty credentials).
    """
    src = path or _DEFAULT_PROVIDERS
    return ProviderCredentialManager(src)


# ---------------------------------------------------------------------------
# Pool sorting strategies (config → requirements.<pool>.sort_strategy)
# ---------------------------------------------------------------------------
_SORT_STRATEGIES = ("ttft_ms", "total_ms", "model_id", "random")
_DEFAULT_SORT_STRATEGY = "total_ms"


def sort_pool(models: list[dict], strategy: str, rng: random.Random | None = None) -> list[dict]:
    """Return *models* ordered according to *strategy*.

    Strategies:
      - ``total_ms`` (default): ascending total response time; untested
        (``total_ms`` is ``None``) models sink to the end.
      - ``ttft_ms``: ascending time-to-first-token; untested (``ttft_ms`` is
        ``None``) models sink to the end.
      - ``model_id``: case-insensitive alphabetical by model id.
      - ``random``: shuffled order. Pass a seeded ``random.Random`` via *rng*
        for deterministic results (used in tests); otherwise the process-wide
        RNG is used and the order changes on each gateway start.

    Unknown strategies fall back to ``total_ms``.
    """
    if strategy == "model_id":
        return sorted(models, key=lambda m: m.get("model_id", "").lower())
    if strategy == "ttft_ms":
        return sorted(models, key=lambda m: m.get("test", {}).get("ttft_ms") or float("inf"))
    if strategy == "random":
        shuffled = list(models)
        (rng or random).shuffle(shuffled)
        return shuffled
    # total_ms (default) — unknown strategies also land here.
    return sorted(models, key=lambda m: m.get("test", {}).get("total_ms") or float("inf"))


def _resolve_sort_strategy(config: dict, category: str) -> str:
    """Read ``sort_strategy`` for *category*, validating against known values."""
    req = config.get("requirements", {}).get(category, {})
    strategy = req.get("sort_strategy", _DEFAULT_SORT_STRATEGY)
    if strategy not in _SORT_STRATEGIES:
        logger.warning(
            f"[SORT] category '{category}': unknown sort_strategy '{strategy}', "
            f"falling back to '{_DEFAULT_SORT_STRATEGY}'"
        )
        return _DEFAULT_SORT_STRATEGY
    return strategy


def build_state(config: dict, test_results: dict, saved_models: dict | None = None) -> GatewayState:
    """Build GatewayState from config and test results.

    Categories are derived dynamically from ``config["requirements"]`` keys.
    Uses ``requirements_breakdown`` from each model entry to determine
    which categories a model belongs to (a model can serve multiple
    categories). Models that failed tests (status != "ok") or are rejected
    are excluded. Each category pool is sorted according to its
    ``sort_strategy`` (config → requirements.<category>.sort_strategy),
    defaulting to ``total_ms`` (fastest first).

    Untested fallback models (``requirements.<category>.fallback_models``) bypass
    the test pipeline entirely: each ``{provider, model_id}`` entry is always
    appended to the end of its category pool. They rotate exactly like tested
    models (same credential rotation and model switching).

    If *saved_models* is provided (from logs/current-model.json), restore
    current_index to point at the saved model for each category. If a saved
    model is not found in the pool, start from the first model (index 0).

    Credential rotation state is loaded from providers.json. Each provider's
    ``current: true`` flag determines the starting credential index.
    """
    state = GatewayState()

    # Load provider config from providers.json (single source of truth)
    cred_manager = load_providers_config()
    state.set_credential_manager(cred_manager)
    gw = config.get("gateway", {})
    state.switch_on_any_error = gw.get("switch_on_any_error", False)
    state.show_debug_tag = gw.get("show_debug_tag", False)
    state.infinite_models = gw.get("infinite_models", False)
    state.request_timeout_ms = gw.get("request_timeout_ms", 60000)
    state.stream_idle_timeout_ms = gw.get("stream_idle_timeout_ms", 30000)

    # Build per-category pools from requirements_breakdown
    all_results = test_results.get("results", [])
    categories = list(config.get("requirements", {}).keys())
    buckets: dict[str, list[dict]] = {cat: [] for cat in categories}
    for m in all_results:
        if m.get("rejected") or m.get("test", {}).get("status") != "ok":
            continue
        breakdown = m.get("requirements_breakdown", {})
        for cat in categories:
            if breakdown.get(cat):
                buckets[cat].append(m)

    for cat in categories:
        strategy = _resolve_sort_strategy(config, cat)
        models = sort_pool(buckets[cat], strategy)
        state.pools[cat] = models
        state.current_index[cat] = 0

    # Mix in untested fallback models (config → requirements.<cat>.fallback_models).
    # These are always appended to the end of the category pool — no testing,
    # no capability check. Duplicates (already tested or repeated in the list)
    # and malformed entries are skipped.
    for cat in categories:
        extras = config.get("requirements", {}).get(cat, {}).get("fallback_models", [])
        if not extras:
            continue
        pool = state.pools[cat]
        existing = {(m["provider"], m["model_id"]) for m in pool}
        for entry in extras:
            if not isinstance(entry, dict):
                logger.warning(f"[FALLBACK_MODELS] {cat}: skipping invalid entry {entry!r}")
                continue
            provider = entry.get("provider")
            model_id = entry.get("model_id")
            if not provider or not model_id:
                logger.warning(                    f"[FALLBACK_MODELS] {cat}: skipping entry missing provider/model_id: {entry!r}")
                continue
            if (provider, model_id) in existing:
                continue
            if provider not in state.providers_creds:
                logger.warning(
                    f"[FALLBACK_MODELS] {cat}: provider '{provider}' has no credentials in providers.json"
                )
            pool.append({
                "provider": provider,
                "model_id": model_id,
                "test": {"status": "ok", "total_ms": None},
            })
            existing.add((provider, model_id))

    if saved_models:
        for cat, saved in saved_models.items():
            if cat not in state.pools:
                continue
            # For `random` the pool is reshuffled on every start, so the saved
            # model would land at an arbitrary position. Keep the fresh random
            # start (index 0) instead of pinning to the last-used model.
            if _resolve_sort_strategy(config, cat) == "random":
                continue
            for idx, m in enumerate(state.pools[cat]):
                if m.get("provider") == saved.get("provider") and m.get("model_id") == saved.get("model_id"):
                    state.current_index[cat] = idx
                    break
            # Not found → keep index 0 (first model)

    return state


# ---------------------------------------------------------------------------
# Error handling helper — shared by non-stream and stream handlers
# ---------------------------------------------------------------------------
def _handle_provider_error(
    state: GatewayState,
    category: str,
    provider_name: str,
    model_id: str,
    status: int,
    body_text: str,
) -> tuple[str, str]:
    """Handle provider error: rotate credentials or switch model.

    Returns: (action, reason) where:
    - action is "retry", "switch", or "error"
    - reason is a short description
    """
    
    if is_quota_error(body_text) and status in {429, 402, 403}:
        error_msg = body_text[:500].replace('\n', ' ')
        logger.warning(
            f"[QUOTA] {category}: {provider_name} (HTTP {status})\n  Message: {error_msg}"
        )

        cred_count = state.credential_count(provider_name)

        if cred_count == 0:
            # No credentials configured → switch model directly.
            state.advance(category)
            save_current_model(state)
            return "switch", "quota: no credentials configured"

        # Rotate to the next credential and record the failure against the
        # current model. Once every credential for this model has failed we
        # advance to the next model. This counter lives on the process-wide
        # state singleton, so it survives across the client's retries.
        state.advance_credential(provider_name)
        state.inc_cred_tried(category)
        tried = state.get_cred_tried(category)

        if tried >= cred_count:
            # Every credential for this model is exhausted → switch model.
            state.advance(category)  # also resets the per-model counter
            save_current_model(state)
            return "switch", "quota: all credentials exhausted"

        # Some credentials remain → rotate and retry.
        return "retry", "quota"

    if state.switch_on_any_error or is_retryable_error(status, body_text):
        state.advance(category)
        save_current_model(state)
        return "switch", "retryable_error"

    return "error", "non_retryable"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and test results on startup."""
    global _state, _config
    config = load_config()
    test_results = load_test_results()
    _config = config

    # Restore persisted current-model state (survives restarts)
    saved = load_current_model()
    _state = build_state(config, test_results, saved)
    save_current_model(_state)

    log_level = config.get("gateway", {}).get("log_level", "normal")
    if log_level == "debug":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    cats = _state.available_categories()
    for cat in cats:
        model = _state.get_model(cat)
        if model:
            logger.info(f"[START] {cat} -> {model['model_id']} ({model['provider']})")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Model Gateway", version="2.0.0", lifespan=lifespan)

_state: GatewayState | None = None
_config: dict | None = None


def get_state() -> GatewayState:
    assert _state is not None, "Gateway not initialized"
    return _state


def get_config() -> dict:
    assert _config is not None, "Gateway not initialized"
    return _config


@app.get("/health")
async def health():
    """Health check endpoint."""
    state = get_state()
    return {
        "status": "ok",
        "categories": {cat: len(pool) for cat, pool in state.pools.items()},
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list. Returns our 3 virtual models."""
    state = get_state()
    data = []
    for cat in state.available_categories():
        models = state.pools[cat]
        data.append({
            "id": cat,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "gateway",
            "meta": {"model_count": len(models), "fastest": models[0]["model_id"] if models else None},
        })
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    body = await request.json()
    requested_model = body.get("model", "simple")
    is_stream = body.get("stream", False)

    logger.debug(f"[INCOMING] model={requested_model}, stream={is_stream}, body_keys={list(body.keys())}")

    # Debug: log message structure
    messages = body.get("messages", [])
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            logger.debug(f"[MSG {i}] role={role}, content=str({len(content)} chars): {content[:200].replace(chr(10), '\\n')}")
        elif isinstance(content, list):
            types = [item.get("type", "?") for item in content]
            logger.debug(f"[MSG {i}] role={role}, content=list({len(content)} items, types={types})")
            for j, item in enumerate(content):
                item_type = item.get("type", "?")
                if item_type == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        mime = url.split(";")[0].split(":")[1]
                        data_len = len(url.split(",", 1)[1]) if "," in url else 0
                        logger.debug(f"  [ITEM {j}] type=image_url, mime={mime}, base64_len={data_len}")
                    else:
                        logger.debug(f"  [ITEM {j}] type=image_url, url={url[:100]}")
                elif item_type == "text":
                    text = item.get("text", "")
                    logger.debug(f"  [ITEM {j}] type=text, text={text[:200]}")
                else:
                    logger.debug(f"  [ITEM {j}] type={item_type}, keys={list(item.keys())}")
        else:
            logger.debug(f"[MSG {i}] role={role}, content={type(content)}")

    state = get_state()
    config = get_config()
    user_agent = config.get("user_agent", "model-gateway/1.0")

    # API key authentication (optional — if set, all requests must include it)
    api_key = config.get("gateway", {}).get("api_key")
    if api_key:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth.removeprefix("Bearer ") != api_key:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid or missing API key", "type": "authentication_error"}},
            )

    # Apply tool_choice from config if tools are present and not already set
    tools = body.get("tools")
    if tools and "tool_choice" not in body:
        gw_tool_choice = config.get("gateway", {}).get("tool_choice")
        if gw_tool_choice:
            body["tool_choice"] = gw_tool_choice

    if requested_model not in state.pools:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"Unknown model: {requested_model}", "type": "invalid_request"}},
        )

    category = requested_model

    if is_stream:
        return await _handle_stream(state, body, category, user_agent)
    else:
        return await _handle_non_stream(state, body, category, user_agent)


async def _handle_non_stream(state: GatewayState, body: dict, category: str, user_agent: str):
    """Handle non-streaming request with retry logic.

    If a model times out (request_timeout_ms), it is treated as a switchable
    error — the gateway advances to the next model in the pool.

    On quota/rate-limit errors, credentials are rotated first (same model).
    When all credentials for a provider are exhausted (wrapped around), the
    next model is tried.
    """
    timeout_sec = state.request_timeout_ms / 1000 if state.request_timeout_ms > 0 else None
    http_timeout = aiohttp.ClientTimeout(total=timeout_sec)

    # reasoning_content is model-specific internal state produced by extended
    # thinking models, but when thinking mode is active the upstream API requires
    # it to be passed back on EVERY request (DeepSeek/Console: "The
    # reasoning_content in the thinking mode must be passed back to the API").
    # The client already stores it in the conversation history for prior thinking
    # turns and sends it to us, so we forward it verbatim — stripping or blanking
    # it (even on a model switch) makes the upstream provider reject the request.
    # It is preserved across model switches and credential rotations alike.
    while True:
        model = state.get_model(category)
        if model is None:
            logger.error(f"[EXHAUSTED] {category}: all models reached limit")
            return JSONResponse(
                status_code=429,
                content={"error": {"message": f"All models in '{category}' category are unavailable", "type": "rate_limit_error"}},
            )

        provider_name = model["provider"]

        logger.debug(f"[REQUEST] {category} -> {model['model_id']} ({provider_name})")

        provider = state._resolve_provider(provider_name)
        if not provider:
            return JSONResponse(status_code=502, content={"error": {"message": f"Unknown provider: {provider_name}"}})

        model_id = model["model_id"]
        url = state._get_provider_url(provider, model_id)
        headers = state._get_provider_headers(provider, user_agent)

        ptype = provider.get("type", "openai")
        if ptype == "google":
            payload = GoogleConverter.openai_to_provider(body, model_id)
        else:
            payload = {body_key: body[body_key] for body_key in body}
            payload["model"] = model_id

        try:
            async with aiohttp.ClientSession(timeout=http_timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp_body = await resp.text()

                    if resp.status == 200:
                        resp_json = json.loads(resp_body)
                        if ptype == "google":
                            resp_json = GoogleConverter.provider_to_openai(resp_json)
                        # A successful response means the current model/credential
                        # is healthy — clear the per-model quota-failure counter.
                        state.reset_cred_tried(category)
                        return JSONResponse(content=resp_json)

                    action, reason = _handle_provider_error(state, category, provider_name, model_id, resp.status, resp_body)

                    err_msg = extract_error_message(resp_body)
                    if action == "retry":
                        cred_pos = state.credential_position(provider_name)
                        logger.warning(
                            f"[SWITCH_CRED] {category}: {model['model_id']} ({provider_name}) -> "
                            f"next credential {cred_pos} ({reason}) — {err_msg}"
                        )
                        continue
                    elif action == "switch":
                        next_model = state.get_model(category)
                        if next_model:
                            logger.warning(
                                f"[SWITCH_MODEL] {category}: {model['model_id']} ({provider_name}) -> "
                                f"{next_model['model_id']} ({next_model['provider']}) "
                                f"({reason}) — {err_msg}"
                            )
                        else:
                            logger.error(f"[EXHAUSTED] {category}: {model['model_id']} ({provider_name}) was last model ({reason}) — {err_msg}\n{resp_body[:500]}")
                        continue
                    else:
                        logger.error(f"[ERROR] {category}: {model['model_id']} ({provider_name}) failed: HTTP {resp.status}\n{resp_body[:1000]}")
                        try:
                            return JSONResponse(status_code=resp.status, content=json.loads(resp_body))
                        except json.JSONDecodeError:
                            return JSONResponse(status_code=resp.status, content={"error": {"message": resp_body[:500]}})
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            # Timeout / network error → treat as switchable
            old_model = model['model_id']
            old_provider = model['provider']
            old_pos = state.credential_position(provider_name) or "1/1"
            
            next_model = state.advance(category)
            save_current_model(state)
            error_type = "timeout" if isinstance(e, asyncio.TimeoutError) else "connection"
            err_detail = str(e) or error_type
            
            if next_model:
                logger.warning(
                    f"[SWITCH_MODEL] {category}: {model['model_id']} ({model['provider']}) -> "
                    f"{next_model['model_id']} ({next_model['provider']}) "
                    f"({err_detail})"
                )
            else:
                logger.error(f"[EXHAUSTED] {category}: {model['model_id']} ({model['provider']}) was last model ({err_detail})")

    logger.error(f"[EXHAUSTED] {category}: all models exhausted")
    return JSONResponse(
        status_code=429,
        content={"error": {"message": f"All models in '{category}' category are unavailable", "type": "rate_limit_error"}},
    )


async def _handle_stream(state: GatewayState, body: dict, category: str, user_agent: str):
    """Handle streaming request with retry logic.

    On quota/rate-limit errors, credentials are rotated first (same model).
    When all credentials for a provider are exhausted (wrapped around), the
    next model is tried.
    """
    # Strip gateway info blocks from conversation history so the model
    # never sees them in subsequent requests.
    if state.show_debug_tag and body.get("messages"):
        body["messages"] = clean_messages(body["messages"])

    async def generate():
        # reasoning_content is model-specific internal state from extended thinking
        # models, but when thinking mode is active the upstream API requires it to
        # be passed back on EVERY request (DeepSeek/Console: "The
        # reasoning_content in the thinking mode must be passed back to the API").
        # The client already includes it in the conversation history for prior
        # thinking turns, so we forward it verbatim across model switches and
        # credential rotations — stripping or blanking it makes the upstream
        # provider reject the request.
        while True:
            model = state.get_model(category)
            if model is None:
                error_chunk = {"error": {"message": f"All models in '{category}' category are unavailable", "type": "rate_limit_error"}}
                yield f"data: {json.dumps(error_chunk)}\n\n"
                return

            provider_name = model["provider"]

            logger.debug(f"[REQUEST] {category} -> {model['model_id']} ({provider_name}) [stream]")

            chunk_count = 0
            finish_reason = None
            completion_tokens = None
            response_text_parts: list[str] = []
            has_tool_calls = False
            idle_timeout = state.stream_idle_timeout_ms / 1000 if state.stream_idle_timeout_ms > 0 else 0
            async for chunk_str, error_info in forward_streaming(state, model, body, user_agent, idle_timeout):
                if error_info is None:
                    chunk_count += 1
                    # Debug: log chunk content (increased from 200 to capture more context)
                    logger.debug(f"[CHUNK {chunk_count}] {chunk_str[:2000]}")

                    # Track metadata for useless response detection
                    fr, ct = parse_chunk_metadata(chunk_str)
                    if fr:
                        finish_reason = fr
                    if ct is not None:
                        completion_tokens = ct

                    # Accumulate visible content for diagnostics
                    content = extract_chunk_content(chunk_str)
                    if content:
                        response_text_parts.append(content)
                    
                    # Track tool_calls presence
                    if has_tool_calls_in_chunk(chunk_str):
                        has_tool_calls = True

                    # Ensure proper SSE format
                    chunk_str = chunk_str.rstrip("\n") + "\n\n"
                    yield chunk_str
                else:
                    err_status = error_info.get("status", 500)
                    status = int(err_status) if isinstance(err_status, (int, str)) else 500
                    body_text = str(error_info.get("body", ""))

                    action, reason = _handle_provider_error(state, category, provider_name, model['model_id'], status, body_text)

                    err_msg = extract_error_message(body_text)
                    if action == "retry":
                        cred_pos = state.credential_position(provider_name)
                        logger.warning(
                            f"[SWITCH_CRED] {category}: {model['model_id']} ({provider_name}) -> "
                            f"next credential {cred_pos} ({reason}) [stream] — {err_msg}"
                        )
                        break
                    elif action == "switch":
                        next_model = state.get_model(category)
                        if next_model:
                            logger.warning(
                                f"[SWITCH_MODEL] {category}: {model['model_id']} ({provider_name}) -> "
                                f"{next_model['model_id']} ({next_model['provider']}) "
                                f"({reason}) [stream] — {err_msg}"
                            )
                        else:
                            logger.error(f"[EXHAUSTED] {category}: {model['model_id']} ({provider_name}) was last model ({reason}) [stream] — {err_msg}\n{body_text[:500]}")
                        break
                    else:
                        logger.error(f"[ERROR] {category}: {model['model_id']} ({provider_name}) failed: HTTP {status}\n{body_text[:1000]} [stream]")
                        error_chunk = {"error": {"message": f"Provider error: {status}", "type": "provider_error"}}
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        return
            else:
                # Stream completed without errors
                # A successful response means the current model/credential is
                # healthy — clear the per-model quota-failure counter.
                state.reset_cred_tried(category)
                response_text = "".join(response_text_parts).strip()
                tail = response_text[-500:] if len(response_text) > 500 else response_text
                logger.debug(
                    f"[STREAM_DONE] {category}: {model['model_id']} ({model['provider']}) "
                    f"sent {chunk_count} chunks, {len(response_text)} chars"
                    + (f" | tail={tail!r}" if tail else "")
                )

                # Decide whether the (successful) stream must still be treated
                # as a retryable failure and the gateway should move on to the
                # next model. Two cases:
                #   1. The model returned no answer (even if [STREAM_DONE] was
                #      reached). That is a provider malfunction — neither text
                #      nor tool_calls — so it is always retryable.
                #   2. The response is "useless" (too short / context overflow),
                #      which is only acted on when switch_on_any_error is set.
                switch_reason = None
                if not response_text and not has_tool_calls:
                    switch_reason = "empty response (no answer)"
                elif state.switch_on_any_error and is_useless_response(finish_reason, completion_tokens):
                    switch_reason = f"useless response: finish_reason={finish_reason}, tokens={completion_tokens}"

                if switch_reason:
                    next_model = state.advance(category)
                    save_current_model(state)

                    preview = f" | text={response_text[:200]!r}" if response_text else ""
                    if next_model:
                        logger.warning(
                            f"[SWITCH_MODEL] {category}: {model['model_id']} ({provider_name}) -> "
                            f"{next_model['model_id']} ({next_model['provider']}) "
                            f"({switch_reason}){preview}"
                        )
                    else:
                        logger.error(
                            f"[EXHAUSTED] {category}: {model['model_id']} ({provider_name}) was last model "
                            f"({switch_reason}){preview}"
                        )
                    
                    # For empty responses, add a short delay before retrying to
                    # protect against temporary provider glitches (otherwise we
                    # burn through 5-7 models per second on a transient failure).
                    if "empty response" in switch_reason:
                        await asyncio.sleep(1.0)
                    
                    # Don't return — continue and retry with next model
                    continue

                # Inject debug tag as the last chunk (only when we actually
                # return this response to the client)
                if state.show_debug_tag:
                    cred_pos = state.credential_position(provider_name) or "1/1"
                    debug_tag = build_debug_tag(model["model_id"], provider_name, cred_pos)
                    debug_chunk = {"choices": [{"delta": {"content": debug_tag}, "index": 0}]}
                    yield f"data: {json.dumps(debug_chunk)}\n\n"

                return

        error_chunk = {"error": {"message": f"All models in '{category}' category are unavailable", "type": "rate_limit_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Re-exports for backward compatibility (tests import these via `import gateway`)
# ---------------------------------------------------------------------------
_parse_sse_error = parse_sse_error
# Standalone wrappers for functions that became GatewayState methods
def resolve_provider(provider_name: str, state: GatewayState) -> dict | None:
    return state._resolve_provider(provider_name)
def get_provider_url(provider: dict, model_id: str) -> str:
    return GatewayState._get_provider_url(provider, model_id)
def get_provider_headers(provider: dict, user_agent: str) -> dict:
    return GatewayState._get_provider_headers(provider, user_agent)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _write_pid() -> None:
    """Write the current process PID to gateway.pid (single source of truth).

    Owns its own PID file so that ``gateway.py`` stays tracked regardless of how
    it was started (gateway.sh, docker restart, or a direct ``python3``).
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _PID_PATH.write_text(str(os.getpid()))


def _remove_pid() -> None:
    """Remove gateway.pid if it belongs to us (clean exit / shutdown)."""
    try:
        current = _PID_PATH.read_text().strip()
    except FileNotFoundError:
        return
    if current == str(os.getpid()):
        try:
            _PID_PATH.unlink()
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible model gateway")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--test-results", type=Path, help="Path to result-test.json")
    parser.add_argument("--port", type=int, help="Override port")
    args = parser.parse_args()

    global _DEFAULT_CONFIG, _DEFAULT_TEST_RESULTS
    if args.config:
        _DEFAULT_CONFIG = args.config
    if args.test_results:
        _DEFAULT_TEST_RESULTS = args.test_results

    setup_logging()
    _write_pid()
    atexit.register(_remove_pid)
    logger.info("Starting gateway...")

    config = load_config(args.config)
    port = args.port or config.get("gateway", {}).get("port", 8765)

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    finally:
        _remove_pid()


if __name__ == "__main__":
    main()
