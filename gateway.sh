#!/usr/bin/env bash
# Unified gateway management CLI
# Usage: ./gateway.sh <command> [args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="logs/gateway.pid"
# How long to wait (seconds) for the gateway PID to appear / become alive.
START_TIMEOUT=10
COMMAND="${1:-help}"

# ---------------------------------------------------------------------------
# Process management (bash)
# ---------------------------------------------------------------------------
# The PID file is owned by gateway.py itself (written on startup, removed on a
# clean shutdown). gateway.sh only *verifies* liveness here, so the file stays
# accurate no matter how the gateway was started (gateway.sh start, docker
# restart, or a direct `python3 src/gateway.py`).

read_pid() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid=$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null)
    [ -n "$pid" ] || return 1
    echo "$pid"
}

is_running() {
    local pid
    pid=$(read_pid) || return 1
    kill -0 "$pid" 2>/dev/null
}

cleanup_stale_pid() {
    # Remove the PID file only if the recorded process is no longer alive.
    if [ -f "$PID_FILE" ] && ! is_running; then
        local pid
        pid=$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null)
        echo "Removing stale PID file (PID ${pid:-unknown})."
        rm -f "$PID_FILE"
    fi
}

wait_alive() {
    # Wait up to START_TIMEOUT seconds for the PID file to appear & be alive.
    local i
    for i in $(seq 1 "$START_TIMEOUT"); do
        if is_running; then
            return 0
        fi
        sleep 1
    done
    return 1
}

cmd_start() {
    if is_running; then
        echo "Gateway already running (PID $(cat "$PID_FILE"))"
        exit 1
    fi
    cleanup_stale_pid

    mkdir -p logs
    # gateway.py writes logs/gateway.pid itself on startup.
    nohup python3 src/gateway.py > logs/gateway.log 2>&1 &

    if wait_alive; then
        echo "Gateway started (PID $(cat "$PID_FILE"))"
    else
        echo "ERROR: gateway failed to start (see logs/gateway.log)" >&2
        cleanup_stale_pid
        exit 1
    fi
}

cmd_stop() {
    if ! is_running; then
        echo "Gateway is not running."
        cleanup_stale_pid
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")
    kill "$pid" 2>/dev/null || true
    echo "Stopping gateway (PID $pid)..."

    # Wait for graceful shutdown; escalate to SIGKILL if it hangs.
    local i
    for i in $(seq 1 "$START_TIMEOUT"); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "Gateway did not stop gracefully, sending SIGKILL..."
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "Gateway stopped."
}

cmd_restart() {
    if is_running; then
        cmd_stop
    else
        echo "Gateway not running; starting fresh."
        cleanup_stale_pid
    fi
    sleep 1
    cmd_start
}

# ---------------------------------------------------------------------------
# State management (delegate to Python)
# ---------------------------------------------------------------------------

cmd_switch() {
    local category="${1:-}"
    if [ -z "$category" ]; then
    echo "Usage: gateway.sh switch <category>"
    exit 1
  fi
  python3 src/commands.py switch "$category"
  cmd_restart
}

cmd_reset() {
    echo "Removing saved state..."
    python3 src/commands.py reset
    cmd_restart
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

case "$COMMAND" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    switch)
        cmd_switch "$2"
        ;;
    reset)
        cmd_reset
        ;;
    rotate)
        PROVIDER="${2:-}"
        if [ -z "$PROVIDER" ]; then
            echo "Usage: gateway.sh rotate <provider>"
            exit 1
        fi
        python3 src/commands.py rotate "$PROVIDER"
        cmd_restart
        ;;
    state)
        STATE_LINES="${2:-}"
        if [ "$STATE_LINES" = "-n" ]; then
            STATE_LINES="${3:-20}"
        elif [ -z "$STATE_LINES" ]; then
            STATE_LINES=20
        fi
        python3 src/commands.py state
        if [ -f logs/gateway.log ]; then
            echo ""
            echo "--- Last $STATE_LINES log lines ---"
            tail -n "$STATE_LINES" logs/gateway.log
        fi
        ;;
    help|--help|-h)
        python3 src/commands.py help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        python3 src/commands.py help
        exit 1
        ;;
esac
