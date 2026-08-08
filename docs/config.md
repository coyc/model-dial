# Configuration

Reference for **`config.json`** — the file that controls gateway and testing settings (requirements, categories, test parameters, gateway options incl. optional API key).

With Docker, this file is created automatically on first run (from the example templates). Running without Docker? See [Local Setup →](local-setup.md). For provider credentials, see [Providers →](providers.md).

## Config & Requirements

### `config.json`

```json
{
    "user_agent": "opencode/1.17.4",
    "models_catalog_url": "https://models.dev/api.json",
    "test_models": {
        "prompt": "Calculate 2+2. Reply ONLY with valid JSON: {\"answer\": <number>, \"reasoning\": \"<one word>\"}. No explanation, no markdown, no code fences. Be quick.",
        "expected": "4",
        "concurrency": 10,
        "timeout": 30,
        "max_tokens": 1024
    },
    "gateway": {
        "api_key": "change-me",
        "port": 8765,
        "switch_on_any_error": true,
        "infinite_models": false,
        "show_debug_tag": false,
        "request_timeout_ms": 60000,
        "stream_idle_timeout_ms": 30000,
        "tool_choice": "auto",
        "log_level": "normal"
    }
}
```

### Fields reference (`config.json`)

| Field | Description |
|-------|-------------|
| `user_agent` | User-Agent header for API requests |
| `models_catalog_url` | URL for the model capabilities catalog |
| `test_models.prompt` | Default prompt for testing |
| `test_models.expected` | Expected answer (for validation) |
| `test_models.concurrency` | Max concurrent requests per provider |
| `test_models.timeout` | Timeout for model test requests in seconds |
| `test_models.max_tokens` | Max tokens in test response |
| `gateway.api_key` | Optional API key. If set, all requests must include `Authorization: Bearer <key>` |
| `gateway.port` | Port for the gateway server (default: 8765) |
| `gateway.switch_on_any_error` | If `true`, switch model on any HTTP error (not just rate limits/quota). |
| `gateway.infinite_models` | If `true`, the gateway never returns 429 — when all models in a category are exhausted, it wraps around to the first model and keeps retrying. Default: `false` |
| `gateway.show_debug_tag` | If `true`, append a `<debug>` tag at the end of each streaming response showing current model/provider/credentials. Stripped from conversation history before forwarding to providers. Streaming only. Default: `false` |
| `gateway.request_timeout_ms` | Timeout for gateway requests in ms (default: 60000) |
| `gateway.stream_idle_timeout_ms` | Idle timeout for streaming in ms (default: 30000) |
| `gateway.log_level` | `"normal"` (errors + switches) or `"debug"` (all requests) |
| `gateway.tool_choice` | `"auto"`, `"required"`, or `"none"` — applied when tools are present |

### Requirements (capability filtering)

Each key in the `"requirements"` section of `config.json` is a **virtual model name** — a single name that represents a *pool* of real models matching the configured requirements. You reference it as `"model": "fast"` (e.g. in OpenCode), and the gateway routes the request to the fastest real model in that pool. No key means no such virtual model exists.

The requirements themselves define which capabilities a model needs to join each pool. Models are checked against a **model catalog** (`models.dev/api.json`) before testing:

```json
{
    "requirements": {
        "fast": {
            "models_catalog": {
                "tool_call": true,
                "limit": { "context": 60000, "output": 8000 }
            },
            "model_id": {
                "providers": [],
                "include": ["flash", "fast", "speed", "quick", "turbo", "realtime", "highspeed", "instant", "rapid", "swift", "haiku"],
                "exclude": ["nano", "micro", "tiny", "safeguard"],
                "min_params": null
            },
            "test": {
                "ttft_ms": 1500,
                "total_ms": 2000
            },
            "sort_strategy": "ttft_ms",
            "fallback_models": [
                {"provider": "opencode", "model_id": "my-private-flash"}
            ]
        },
        "coder": {
            "models_catalog": {
                "tool_call": true,
                "reasoning": true,
                "limit": { "context": 200000, "output": 16000 }
            },
            "model_id": {
                "providers": [],
                "include": [],
                "exclude": ["nano", "micro", "tiny", "lite", "small"],
                "min_params": "30b"
            },
            "test": {
                "ttft_ms": null,
                "total_ms": null
            }
        },
        "visual": {
            "models_catalog": {
                "tool_call": true,
                "reasoning": true,
                "attachment": true,
                "modalities": { "input": ["text", "image"] },
                "limit": { "context": 60000, "output": 16000 }
            },
            "model_id": {
                "providers": [],
                "include": [],
                "exclude": ["nano", "micro", "flash", "tiny", "lite", "small"],
                "min_params": "8b"
            },
            "test": {
                "ttft_ms": null,
                "total_ms": null
            }
        },
        "planner": {
            "models_catalog": {
                "tool_call": true,
                "reasoning": true,
                "limit": { "context": 900000, "output": 120000 }
            },
            "model_id": {
                "include": ["deepseek-v4-pro", "nemotron-3-ultra", "glm-5.2"]
            }
        }
    }
}
```

Each pool has five sub-sections:

#### `models_catalog`

Capability-based checks against the model catalog. Each key is a required capability — the model must satisfy all of them to join the pool. Checked recursively:

| Type | Example | Rule |
|------|---------|------|
| `bool` | `"tool_call": true` | Model must have this flag set to `true` |
| `int`/`float` | `"limit": {"context": 32000}` | Model's value must be ≥ required |
| `list` | `"modalities": {"input": ["text", "image"]}` | All required items must be present in model's list |
| `dict` | `"limit": {...}` | Recursively check nested keys |
| `string` | `"family": "claude"` | Prefix match — `"claude"` matches `"claude-sonnet"` |

#### `model_id`

Name-based filters that apply directly to the model ID:

| Filter | Type | Description |
|--------|------|-------------|
| `include` | `list[str]` | Required substrings (case-insensitive). If **at least one** substring is found in the model's ID, the model passes this check. If **none** match, the model is **rejected** for this group. Useful for requiring a specific keyword in the name (e.g. `["vision"]`). Set to `[]` or omit to disable. |
| `exclude` | `list[str]` or `null` | If **any** substring (case-insensitive) is found in the model's ID, the model is **rejected** for this group — even if it matched `include`. Useful for blocking obviously small/limited models by name pattern. |
| `min_params` | `str` or `int` or `null` | Minimum parameter count (in billions). Parsed automatically from the model ID via pattern like `70b`, `8B`, `550b-a55b`. If the ID contains no parameter pattern (e.g. `deepseek-r1:free`), the check is **skipped** — the model is kept. Set to `null` to disable. |
| `providers` | `list[str]` or `null` | Allowed providers (exact match). If the model's provider is **not** in the list, the model is **rejected** for this group. Useful for limiting a group to specific providers (e.g. `["opencode", "google"]`). Set to `[]` or omit to disable. |

Examples of name-based filtering in action:

| Model ID | `include: ["vision"]` | `exclude: ["nano","micro","flash"]` | `min_params: "30b"` | Result |
|----------|----------------------|--------------------------------------|----------------------|--------|
| `llama-3.2-3b` | ❌ no "vision" | ✅ passes | ❌ 3 < 30 | Blocked by include + min_params |
| `gemma-2-2b-it` | ❌ no "vision" | ✅ passes | ❌ 2 < 30 | Blocked by include + min_params |
| `gemini-2.0-flash` | ❌ no "vision" | ❌ "flash" matches | ⏭️ no param count | Blocked by exclude |
| `llama-3.3-70b-versatile` | ❌ no "vision" | ✅ passes | ✅ 70 ≥ 30 | Blocked by include |
| `qwen3-vision-nano` | ✅ "vision" found | ❌ "nano" matches | ⏭️ no param count | Blocked by exclude (exclude wins) |
| `qwen3-vision-max` | ✅ "vision" found | ✅ passes | ⏭️ no param count | Passes all |
| `deepseek-r1:free` | ❌ no "vision" | ✅ passes | ⏭️ no param count | Blocked by include |
| `nemotron-550b-a55b` | ❌ no "vision" | ✅ passes | ✅ 550 ≥ 30 | Blocked by include |

Models that don't appear in the catalog are **rejected** (`rejected: true`, `reject_reason: "no catalog data"`) — they won't be tested. Models that fail all category requirements (`requirements_breakdown` all `false`) are also skipped, saving time and API calls.

#### `test`

Post-test performance thresholds applied after models are tested. Controls which pool a tested model ends up in:

| Field | Description |
|-------|-------------|
| `ttft_ms` | Max Time-To-First-Token in ms. If exceeded, the category is set to `false` in `requirements_breakdown` — the model is excluded from that category's pool. |
| `total_ms` | Max total response time in ms. Same behavior as `ttft_ms`. |

`null` or absent = no filtering for that metric. When a model's measured values exceed thresholds, only that specific category is set to `false` in `requirements_breakdown` — the model can still pass other categories that don't have those constraints.

#### `sort_strategy`

Controls how the gateway **orders** the tested models in a pool (and thus in `logs/current-model.json`) at startup. It does **not** include/exclude models — only the position of each model in the rotation.

| Value | Behavior |
|-------|----------|
| `total_ms` | Ascending total response time (fastest first). Untested models (`total_ms` is `null`) sink to the end. **Default** when the key is absent or unrecognized. |
| `ttft_ms` | Ascending time-to-first-token (lowest latency to first token first). Untested models (`ttft_ms` is `null`) sink to the end. Best for latency-sensitive pools. |
| `model_id` | Case-insensitive alphabetical order by `model_id`. |
| `random` | Models are shuffled. The order changes on every gateway start (uses the process RNG), giving each model a fair share of "first pick" over restarts. |

Unknown values log a warning and fall back to `total_ms`.

```json
{
    "requirements": {
        "fast":   { "sort_strategy": "ttft_ms" },
        "coder":  { "sort_strategy": "total_ms" },
        "visual": { "sort_strategy": "model_id" },
        "planner": { "sort_strategy": "random" }
    }
}
```

> Note: `fallback_models` are always appended **after** the sorted tested models, regardless of `sort_strategy` — they act as fallback models used only once the faster tested models fail.

#### `fallback_models`

Untested models always added to the pool **without running the test pipeline**. Each entry is a `{provider, model_id}` pair that is appended to the end of the category's model pool on gateway startup (and thus written to `logs/current-model.json`). These models rotate exactly like tested ones: the gateway applies the same credential rotation and model switching on errors.

> **Important:** `model_id` must be the **full model name** exactly as expected by the provider's API (e.g., `gemini-2.5-pro-preview`, not `gemini-2.5`). Short names or aliases will cause API requests to fail.

```json
{
    "requirements": {
        "coder": {
            "fallback_models": [
                {"provider": "opencode", "model_id": "my-private-coder"},
                {"provider": "google", "model_id": "gemini-2.5-pro-preview"}
            ]
        }
    }
}
```

| Field | Description |
|-------|-------------|
| `provider` | Provider ID from `providers.json` (must have credentials configured, otherwise the gateway logs a warning and the model fails at request time) |
| `model_id` | The model ID sent to the provider API |

Rules:
- Entries are appended **after** the sorted tested models — they always sit at the end of the pool and are only used once faster tested models fail.
- Duplicates (a model that already passed testing, or repeated in the list) are skipped.
- Malformed entries (missing `provider` or `model_id`) are skipped with a warning.
- Since these models are never tested, `total_ms` is saved as `null` in `logs/current-model.json`.

#### Example

| Pool | Key Requirements | Description |
|----------|-----------------|-------------|
| **fast** | `tool_call`, `limit.context ≥ 60000`, `limit.output ≥ 8000`, name matches `flash/fast/speed/quick/turbo/…`, `ttft_ms ≤ 1500`, `total_ms ≤ 2000` | Speed-optimized models for low-latency chat |
| **coder** | `tool_call`, `reasoning`, `limit.context ≥ 200000`, `limit.output ≥ 16000`, `min_params: 30b` | Reasoning-capable models with large context for coding |
| **visual** | `tool_call`, `reasoning`, `attachment`, `modalities.input` includes `image`, `limit.context ≥ 60000`, `limit.output ≥ 16000`, `min_params: 8b` | Multimodal models with vision and attachment support |
| **planner** | `tool_call`, `reasoning`, `limit.context ≥ 900000`, `limit.output ≥ 120000`, `include: ["deepseek-v4-pro", "nemotron-3-ultra", "glm-5.2"]` | Long-context models for complex multi-step planning |
