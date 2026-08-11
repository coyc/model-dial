#!/usr/bin/env bash
#
# run.sh — models-tester
#
# Fetches models from providers, checks requirements against catalog,
# tests each eligible model, saves results to result-*.json.
#
# Pipeline: fetch -> check-requirements -> test -> gateway reset
#
# Usage:
#   ./run.sh                                   — full run
#   ./run.sh --providers groq,nvidia           — specific providers only
#   ./run.sh --concurrency 5                   — fewer parallel requests
#   ./run.sh --only-fetch                      — step 1 only (fetch models)
#   ./run.sh --only-requirements               — step 2 only (check requirements, uses existing result-fetch.json)
#   ./run.sh --only-test                       — step 3 only (test models, uses existing result-requirements.json)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/src"
FETCH_OUTPUT="$SCRIPT_DIR/result-fetch.json"
CHECKED_OUTPUT="$SCRIPT_DIR/result-requirements.json"
TEST_OUTPUT="$SCRIPT_DIR/result-test.json"
LOCK_FILE="$SCRIPT_DIR/logs/run.lock"
cd "$SCRIPT_DIR"

# --- Concurrency guard ---
# Only one run.sh (manual `make test` or the scheduler) may write the
# result-*.json files at a time — a second run truncates them mid-write
# via `>` and can crash the gateway when it reads a half-written file.
mkdir -p "$(dirname "$LOCK_FILE")"
if [[ -f "$LOCK_FILE" ]]; then
  existing_pid="$(tr -d '[:space:]' < "$LOCK_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Error: another run.sh is already running (PID $existing_pid, lock: $LOCK_FILE)." >&2
    echo "Wait for it to finish, or check its progress before retrying." >&2
    exit 1
  fi
  echo "Removing stale lock file (PID ${existing_pid:-unknown} no longer running)." >&2
  rm -f "$LOCK_FILE"
fi
echo "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# Checks
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found" >&2
  exit 1
fi

if ! python3 -c "import aiohttp, fastapi, uvicorn" 2>/dev/null; then
  echo "Installing dependencies..." >&2
  pip3 install --break-system-packages -r requirements.txt 2>&1 | tail -1 >&2
fi

# Parse flags
ONLY_FETCH=false
ONLY_REQUIREMENTS=false
ONLY_TEST=false
FETCH_ARGS=()
TEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-fetch)        ONLY_FETCH=true ;;
    --only-requirements) ONLY_REQUIREMENTS=true ;;
    --only-test)         ONLY_TEST=true ;;
    --providers)
      shift || { echo "Error: --providers requires a value" >&2; exit 1; }
      FETCH_ARGS+=("--providers" "$1")
      ;;
    --providers=*)
      FETCH_ARGS+=("--providers" "${1#*=}")
      ;;
    --concurrency)
      shift || { echo "Error: --concurrency requires a value" >&2; exit 1; }
      TEST_ARGS+=("--concurrency" "$1")
      ;;
    --concurrency=*)
      TEST_ARGS+=("--concurrency" "${1#*=}")
      ;;
    --test-timeout)
      shift || { echo "Error: --test-timeout requires a value" >&2; exit 1; }
      TEST_ARGS+=("--test-timeout" "$1")
      ;;
    --test-timeout=*)
      TEST_ARGS+=("--test-timeout" "${1#*=}")
      ;;
    *)
      echo "Error: unknown parameter '$1'" >&2
      echo "Available: --providers, --concurrency, --test-timeout, --only-fetch, --only-requirements, --only-test" >&2
      exit 1
      ;;
  esac
  shift
done

# 1. Fetch models
if [[ "$ONLY_REQUIREMENTS" == false && "$ONLY_TEST" == false ]]; then
  echo "=== [1/3] Fetching models ===" >&2
  python3 "$SRC/fetch-models.py" "${FETCH_ARGS[@]}" 2>/dev/null > "$FETCH_OUTPUT"
  echo "  Saved to $FETCH_OUTPUT" >&2

  if [[ "$ONLY_FETCH" == true ]]; then
    echo "  (--only-fetch — done)" >&2
    exit 0
  fi
fi

# 2. Check requirements
if [[ "$ONLY_TEST" == false ]]; then
  echo "=== [2/3] Checking requirements ===" >&2
  python3 "$SRC/check-requirements.py" --input "$FETCH_OUTPUT" > "$CHECKED_OUTPUT"
  echo "  Results in $CHECKED_OUTPUT" >&2

  if [[ "$ONLY_REQUIREMENTS" == true ]]; then
    echo "  (--only-requirements — done)" >&2
    exit 0
  fi
fi

# Ensure checked output exists for test
if [[ ! -s "$CHECKED_OUTPUT" ]]; then
  echo "[error] $CHECKED_OUTPUT not found or empty. Run check-requirements first." >&2
  exit 1
fi

# 3. Test models
echo "=== [3/3] Testing models ===" >&2
python3 "$SRC/test-models.py" --input "$CHECKED_OUTPUT" "${TEST_ARGS[@]}" > "$TEST_OUTPUT"
echo "  Results in $TEST_OUTPUT" >&2

if [[ "$ONLY_TEST" == true ]]; then
  echo "  (--only-test — done)" >&2
  exit 0
fi

# 4. Reset gateway (clear current-model and restart)
echo "=== Reseting and restarting gateway ===" >&2
"$SCRIPT_DIR/gateway.sh" reset 2>&1 | sed 's/^/  /' >&2
