# Local Setup (without Docker)

For users who prefer to run Model Dial without Docker.

## Install

```bash
cd model-dial
cp docker/config.example.json config.json
cp docker/providers.example.json providers.json
pip3 install --break-system-packages -r requirements.txt
```

1. **Fill `providers.json`** — keep only the providers you use, with real API keys; delete the rest (they're examples with dummy keys)
2. **Set `gateway.api_key`** in `config.json` — the key the gateway will require
3. **Run `./run.sh`** — scans providers, tests models, and auto-starts the gateway at the end
4. **Configure models in your app** — example: [Connect from OpenCode](../README.md#connect-from-opencode)

Config file reference: **[providers.md](providers.md)**, **[config.md](config.md)**, **[deny.md](deny.md)**

## Refresh models

Run periodically to keep the gateway using the latest and fastest models:

```bash
./run.sh
```

## Manage the gateway

```bash
./gateway.sh start              # start (background, PID in logs/gateway.pid)
./gateway.sh stop               # stop
./gateway.sh restart            # restart
./gateway.sh state              # show current state (PID, models, credentials)
./gateway.sh switch <category>  # switch to next model in category + restart
./gateway.sh reset              # remove saved state and restart
./gateway.sh rotate nvidia      # rotate API key for nvidia → next credential + restart
./gateway.sh help               # show all commands
```
