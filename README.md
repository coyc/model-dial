# Model Dial

> Use pool of the fastest capable AI models as single model. Model switched automatically if error.

<div align="center">
  <img src="docs/architecture.svg" alt="Architecture Overview" width="820"/>
</div>

A Python tool to fetch available models from AI providers, check their capabilities against a model catalog, test latency + answer quality of eligible models, and serve the fastest per category through an OpenAI-compatible gateway.

## Why
If you have multiple models (even from different providers), but they're not always stable (connection issues, queue load, strict rate limits, etc.), manually switching models each time is inconvenient, especially if you work with multiple agents.

Model-Dial allows you to create a pool of models with specified characteristics (or even a separate pool for each agent) and use this pool as a single model. Under the hood, Model-Dial automatically handles switching models and providers if any issues arise.

As a result, you get a stable, working model with the specified requirements.

## Features
- Fetches model lists from any providers in `providers.json` (OpenAI-compatible + Google)
- Checks each model's capabilities against a **model catalog** — rejects models without catalog data
- Filters by **per-category requirements** (tool_call, reasoning, vision, context limits, etc.) before testing
- Tests qualifying models with a simple prompt via streaming
- Measures TTFT (Time To First Token) and total response time
- Validates answers (configurable prompt and expected value in response)
- Each model's `requirements_breakdown` (from catalog check) determines category membership — no separate classification  needed
- **OpenAI-compatible gateway** that routes requests to the fastest available model per category
- Auto-switches to next model on rate limit / quota / errors
- Generates JSON reports for analysis

## Quick Start

### Docker Setup (Recommended)

```bash
cd model-dial
make up
```

1. **Edit `providers.json`** — add your real AI providers and API keys (delete unused providers)
2. **Test models** — run `make test` to verify models (it restarts the gateway automatically)
3. **Monitor gateway** — use `make state` for live updates (current models, logs)
4. **Configure models in your app** — run `make config-opencode` to generate provider config for OpenCode (see [docs/opencode.md](docs/opencode.md))

**All commands:** `make help` — shows all available commands grouped by category

> **Windows users:** Makefile requires `make` (available in Git Bash or WSL). Alternatively, use `docker compose` commands directly.

### Local Setup

For users who prefer to run without Docker, see [Local Setup →](docs/local-setup.md).

## Configuration

Three config files control Model Dial (full reference with all fields and examples in `docs/`):

- **`config.json`** — gateway and testing settings (requirements, categories, test parameters, gateway options incl. optional API key) — [docs/config.md](docs/config.md)
- **`providers.json`** — provider credentials (API keys) and types (single source of truth for provider access) — [docs/providers.md](docs/providers.md)
- **`deny.json`** — optional list of models to block from testing — [docs/deny.md](docs/deny.md)

## Usage

Once the gateway is running, it routes requests to the fastest available models based on your configuration.

**Refresh models periodically** to keep the gateway using the latest and fastest models:

```bash
make test
```

This command:
- Fetches current model lists from your configured providers
- Tests their latency and capabilities
- Updates and restarts the gateway with the fastest models per category

## Auto-testing

Model tests also run automatically every hour by default. Configure in [docker-compose.yml](docker-compose.yml):

```yaml
environment:
  SCHEDULER_INTERVAL: 60            # every N minutes (default: 60)
  # OR
  SCHEDULER_TIMES: "08:00 14:00 20:00"  # specific times (mutually exclusive with INTERVAL)
  SCHEDULER_ARGS: "--concurrency 5"     # optional: extra args for run.sh
```

- **`SCHEDULER_INTERVAL`** — minutes between test runs. Comment out to disable.
- **`SCHEDULER_TIMES`** — space-separated `HH:MM` list. Overrides interval mode.

Scheduler output appears in docker logs with `[scheduler]` prefix (use `make logs` for watch logs).

## Gateway

**Transparent proxy** that routes requests to the fastest available model in each category (defined dynamically in `config.json` → `requirements`). Only modifies the model name — everything else (messages, tools, parameters) is passed through as-is. Supports **multi-key credential rotation**: when a model hits a rate/quota limit, the gateway tries the next API key for the same model before switching to a next model in category.

### Start / Stop / Manage

```bash
make start                     # start gateway
make stop                      # stop gateway
make restart                   # restart gateway
make state                     # show current state (models, credentials) + live logs
make switch CATEGORY=fast      # switch to next model in category + restart
make reset                     # remove saved state and restart (return to first model and credentials)
make rotate PROVIDER=nvidia    # rotate API key for provider + restart
```

Running without Docker? See [manage the gateway](docs/local-setup.md#manage-the-gateway).

### Connect from OpenCode

Run `make config-opencode` to generate the provider config for your current `config.json` requirements. The utility outputs a ready-to-use JSON snippet for `opencode.jsonc`.

Full documentation with examples and field mapping: [docs/opencode.md](docs/opencode.md).

### API

The gateway exposes standard OpenAI endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available categories (from `config.json` → `requirements`) |
| `POST /v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `GET /health` | Health check |

Request example:
```bash
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coder",
    "messages": [{"role": "user", "content": "Write a Python quicksort"}],
    "stream": true
  }'
```

### How it works

1. On startup, loads `result-test.json` and groups successful models into per-category pools using `requirements_breakdown`, sorted by speed (fastest first)
2. Loads provider credentials from `providers.json` and initializes rotation state
3. Client sends request with `model: "fast"` (or any category from config)
4. Gateway replaces the category name with actual provider model ID (e.g., `"qwen3-flash"` on Alibaba)
5. Forwards the request as-is — no modification to messages, tools, or parameters
6. Streams response back as-is
7. On **quota/rate-limit error** → rotates to next API key for the same model (same provider)
8. When **all credentials exhausted** (wrapped around) → switches to the next model
9. On other retryable errors (503, 408, etc.) → switches to the next model directly
10. If all models exhausted → returns `429`: "All models in 'X' category are unavailable"

`logs/current-model.json` stores the current per-category model pools (persisted across restarts).


## More Info

| Document | Description |
|----------|-------------|
| [docs/local-setup.md](docs/local-setup.md) | Run Model Dial without Docker — install, configure, and manage the gateway locally |
| [docs/config.md](docs/config.md) | Reference for `config.json` — gateway settings, test parameters, requirements, categories |
| [docs/providers.md](docs/providers.md) | Reference for `providers.json` — provider credentials, types, and credential rotation |
| [docs/deny.md](docs/deny.md) | Reference for `deny.json` — blocking specific models from being tested |
| [docs/logs.md](docs/logs.md) | Gateway logs — formats, log levels (`normal` / `debug`), and daily rotation |
| [docs/opencode.md](docs/opencode.md) | Connect OpenCode to the Model Dial gateway
