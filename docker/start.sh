#!/usr/bin/env bash
set -e

# Auto-copy example configs if they don't exist
if [ ! -f config.json ]; then
    echo "config.json not found. Copying from docker/config.example.json..."
    cp docker/config.example.json config.json
fi

if [ ! -f providers.json ]; then
    echo "providers.json not found. Copying from docker/providers.example.json..."
    cp docker/providers.example.json providers.json
fi

if [ ! -f deny.json ]; then
    echo "deny.json not found. Copying from docker/deny.example.json..."
    cp docker/deny.example.json deny.json
fi

# Auto-generate gateway API key if it's still the example value
python3 << 'EOF'
import json
import secrets

# Read both files
with open('docker/config.example.json', 'r') as f:
    example_config = json.load(f)
with open('config.json', 'r') as f:
    current_config = json.load(f)

# Check if api_key matches the example
example_key = example_config.get('gateway', {}).get('api_key', '')
current_key = current_config.get('gateway', {}).get('api_key', '')

if current_key == example_key:
    # Generate random API key
    random_key = secrets.token_hex(32)
    current_config['gateway']['api_key'] = random_key
    
    # Write back (4-space indent to match config.example.json)
    with open('config.json', 'w') as f:
        json.dump(current_config, f, indent=4)
        f.write('\n')
    
    print(f"Generated random API key for gateway: {random_key}")
    print("Use this key to authenticate requests to the gateway.")
EOF

KEEP_ALIVE_PID=""
TAIL_PID=""
SCHEDULER_PID=""

cleanup() {
    echo "Received shutdown signal. Stopping gateway..."

    # Kill the background processes first
    if [ -n "$TAIL_PID" ]; then
        kill "$TAIL_PID" 2>/dev/null || true
    fi
    if [ -n "$SCHEDULER_PID" ]; then
        kill "$SCHEDULER_PID" 2>/dev/null || true
    fi
    if [ -n "$KEEP_ALIVE_PID" ]; then
        kill "$KEEP_ALIVE_PID" 2>/dev/null || true
    fi

    # Stop gateway
    ./gateway.sh stop

    # Wait for gateway process to actually stop (max 5 seconds)
    for i in {1..5}; do
        if [ -f logs/gateway.pid ]; then
            PID=$(cat logs/gateway.pid)
            if kill -0 "$PID" 2>/dev/null; then
                echo "Waiting for gateway to stop..."
                sleep 1
            else
                echo "Gateway stopped."
                break
            fi
        else
            echo "Gateway stopped."
            break
        fi
    done

    exit 0
}

trap cleanup SIGTERM SIGINT

# Start gateway
./gateway.sh start

# Wait for log file to appear (max 5 seconds)
for i in {1..10}; do
    if [ -f logs/gateway.log ]; then
        break
    fi
    sleep 0.5
done

# Stream logs to stdout for Docker
if [ -f logs/gateway.log ]; then
    tail -F logs/gateway.log &
    TAIL_PID=$!
    echo "Gateway logs streaming to stdout..."
fi

# Start scheduler (if configured via environment)
if [ -n "${SCHEDULER_INTERVAL:-}" ] || [ -n "${SCHEDULER_TIMES:-}" ]; then
    ./docker/scheduler.sh &
    SCHEDULER_PID=$!
    echo "Scheduler started (PID $SCHEDULER_PID)."
fi

# Keep container alive with a loop that can be interrupted
echo "Container is running. Press Ctrl+C or send SIGTERM to stop."
while true; do
    sleep 1
done &
KEEP_ALIVE_PID=$!

# Wait for the background process (this allows trap to work)
wait $KEEP_ALIVE_PID