# Deny List

Reference for **`deny.json`** — a file in the project root to block specific models from ever being tested, regardless of their catalog data.

See also: **[Config](config.md)** for requirement-based filtering, **[Providers](providers.md)** for credential management.

## `deny.json`

```json
{
    "deny": [
        {"model_id": "error-model"},
        {"model_id": "qwen/qwen3-32b", "provider": "groq"}
    ]
}
```

| Field | Description |
|-------|-------------|
| `model_id` | Model ID to block (matched exactly) |
| `provider` | If set, only blocks this (provider, model_id) pair. If `null` or absent, blocks the model globally across all providers. |

Models matching a deny entry are marked `rejected: true` with `reject_reason: "denied"`. Their capabilities are still recorded (for debugging) but `requirements_breakdown` is skipped since they won't be tested.

If the file doesn't exist — no models are denied.
