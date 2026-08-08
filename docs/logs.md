# Logs

Logs go to `logs/gateway.log` with daily rotation (keeps today + yesterday).

Two modes controlled by `gateway.log_level` in [config.json](config.md):

## normal (default)

Only errors, switches, start/stop:

```
2026-07-08 12:00:01 [INFO] Gateway started. Categories: {'fast': 41, 'coder': 29, 'visual': 24, 'planner': 7}
2026-07-08 12:00:05 [WARNING] [SWITCH_MODEL] fast: qwen3-flash → qwen3-plus (HTTP 429, attempt 1/2)
2026-07-08 12:00:05 [ERROR] [EXHAUSTED] visual: all models reached limit
```

## debug

All requests including incoming, streaming, completions:

```
2026-07-08 12:00:05 [INFO] [INCOMING] model=fast, stream=True, body_keys=[...]
2026-07-08 12:00:05 [DEBUG] [REQUEST] fast → qwen3-flash (alibaba) [stream]
2026-07-08 12:00:06 [DEBUG] [STREAM_DONE] fast: qwen3-flash sent 42 chunks
```

> You can monitor all logs together using `make logs` command.
