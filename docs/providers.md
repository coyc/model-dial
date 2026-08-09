# Providers

Single source of truth for **`providers.json`** — the file that controls provider credentials (API keys) and types.

With Docker, this file is created automatically on first run (from the example templates). Running without Docker? See [Local Setup →](local-setup.md).

## `providers.json`

Supports **multiple API keys per provider** with automatic rotation on quota exhaustion.

```json
{
    "nvidia": {
        "type": "openai",
        "credentials": [
            {"base_url": "https://integrate.api.nvidia.com/v1", "api_key": "nvapi-xxx"},
            {"base_url": "https://integrate.api.nvidia.com/v1", "api_key": "nvapi-yyy"},
            {"base_url": "https://integrate.api.nvidia.com/v1", "api_key": "nvapi-zzz"}
        ]
    },
    "openrouter": {
        "type": "openai",
        "model_filter": ":free",
        "credentials": [
            {"base_url": "https://openrouter.ai/api/v1", "api_key": "sk-or-xxx"}
        ]
    },
    "google": {
        "type": "google",
        "credentials": [
            {"base_url": "https://generativelanguage.googleapis.com/v1beta", "api_key": "AIza-xxx"}
        ]
    }
}
```

**Provider names**: top-level key = provider ID, matching the [models.dev catalog](https://models.dev/api.json) — any provider from it can be added.

> **Note:** Currently only `"openai"` (OpenAI-compatible) and `"google"` types are supported. Other API formats are planned for future releases.

| Field | Description |
|-------|-------------|
| `type` | Provider type: `"openai"` (OpenAI-compatible) or `"google"` (native Google API) |
| `model_filter` | Optional: filter models by substring in ID (e.g. `":free"` for OpenRouter) |
| `credentials` | Array of credential objects for this provider |
| `credentials[].base_url` | API base URL (can differ per credential, e.g. for Alibaba regions) |
| `credentials[].api_key` | API key |
| `credentials[].current` | Optional. If `true`, this credential is used on startup (only one per provider). If omitted, the first credential is used automatically |

**Credential rotation**: When a provider returns a quota/rate-limit error, both the gateway and the model test runner automatically rotate to the next credential for the same model. Only when all credentials are exhausted does the gateway switch to the next model (test runner marks the model as failed). Credentials cycle by wrap-around (never exhausted — repeats from the first). The `current` flag in `providers.json` is always kept in sync with the active credential.
