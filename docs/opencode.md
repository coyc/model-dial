# Connect from OpenCode

Add the generated provider to `opencode.jsonc`. The config is produced automatically from your `config.json` requirements — run:

```bash
make config-opencode
```

This prints the provider config to stdout. Copy the output into your `opencode.jsonc`:

```jsonc
{
  "provider": {
    "gateway": {
      // ... generated config here
    }
  }
}
```

Or save directly to a file:

```bash
python3 tools/generate_opencode_config.py -o opencode-provider.json
```

By default the output is indented with 2 spaces. Override it with option:

```bash
python3 tools/generate_opencode_config.py --indent 4
make config-opencode INDENT=4
```

## Example output

Given this in `config.json`:

```json
{
  "requirements": {
    "fast": {
      "models_catalog": {
        "tool_call": true,
        "limit": { "context": 60000, "output": 8000 }
      }
    },
    "coder": {
      "models_catalog": {
        "tool_call": true,
        "reasoning": true,
        "limit": { "context": 200000, "output": 32000 }
      }
    },
    "visual": {
      "models_catalog": {
        "tool_call": true,
        "reasoning": true,
        "attachment": true,
        "modalities": { "input": ["text", "image"] },
        "limit": { "context": 60000, "output": 16000 }
      }
    },
    "planner": {
      "models_catalog": {
        "tool_call": true,
        "reasoning": true,
        "limit": { "context": 900000, "output": 120000 }
      }
    }
  },
  "gateway": {
    "api_key": "your-key"
  }
}
```

The utility generates:

```json
{
    "provider": {
        "gateway": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Model-Dial Gateway",
            "options": {
                "baseURL": "http://localhost:8765/v1",
                "apiKey": "your-key"
            },
            "models": {
                "fast": {
                    "name": "Fast (auto)",
                    "tool_call": true,
                    "limit": { "context": 60000, "output": 8000 }
                },
                "coder": {
                    "name": "Coder (auto)",
                    "tool_call": true,
                    "reasoning": true,
                    "limit": { "context": 200000, "output": 32000 }
                },
                "visual": {
                    "name": "Visual (auto)",
                    "tool_call": true,
                    "reasoning": true,
                    "attachment": true,
                    "limit": { "context": 60000, "output": 16000 },
                    "modalities": {
                        "input": ["text", "image"],
                        "output": ["text"]
                    }
                },
                "planner": {
                    "name": "Planner (auto)",
                    "tool_call": true,
                    "reasoning": true,
                    "limit": { "context": 900000, "output": 120000 }
                }
            }
        }
    }
}
```

## Field mapping

The utility maps each requirement's `models_catalog` to OpenCode model capabilities:

| `models_catalog` field | OpenCode model field | Rule |
|------------------------|---------------------|------|
| `tool_call: true` | `tool_call: true` | Only included when `true` |
| `reasoning: true` | `reasoning: true` | Only included when `true` |
| `attachment: true` | `attachment: true` | Only included when `true` |
| `modalities` | `modalities` | Input copied as-is, output always `["text"]` |
| `limit` | `limit` | Copied as-is |

## Important notes

**`tool_call: true` is required** — without it OpenCode treats the model as tool-incapable, which causes incorrect message ordering when tools are used (e.g., `tool` → `user` without `assistant` between them gets sent as-is, and some providers reject it).

**Capability fields per category** must match what the gateway's requirements guarantee for that pool — see `config.json` → `requirements`. Over-declaring (e.g. `attachment: true` on `fast`) will cause errors when the gateway routes to a model that doesn't support it.

**`modalities` with `image` input** is required for visual categories — otherwise OpenCode blocks images before they reach the gateway.

**API key**: If `gateway.api_key` is set in `config.json`, the gateway checks `Authorization: Bearer <key>` on every incoming request. If the key is empty or not set, no auth is required (keep `apiKey` in OpenCode config matching whatever is in `config.json`).

> **Regenerate after pool changes**: When you add, remove, or modify a model pool in `config.json` (under `requirements`), you must regenerate the OpenCode config by re-running `make config-opencode` — otherwise OpenCode will still use the stale provider definition.
