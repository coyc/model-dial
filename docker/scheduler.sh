#!/usr/bin/env bash
#
# scheduler.sh — periodic model testing
#
# Runs ./run.sh on a schedule. Two modes (mutually exclusive):
#   SCHEDULER_INTERVAL — run every N minutes
#   SCHEDULER_TIMES    — run at specific times (space-separated "HH:MM" list)
#
# SCHEDULER_ARGS — extra arguments passed to ./run.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_PREFIX="[scheduler]"

log() {
    echo "$LOG_PREFIX $*" >&2
}

# --- Validate configuration ---

if [[ -n "${SCHEDULER_INTERVAL:-}" && -n "${SCHEDULER_TIMES:-}" ]]; then
    log "ERROR: SCHEDULER_INTERVAL and SCHEDULER_TIMES are mutually exclusive. Set only one."
    exit 1
fi

if [[ -z "${SCHEDULER_INTERVAL:-}" && -z "${SCHEDULER_TIMES:-}" ]]; then
    log "No scheduler configuration found (SCHEDULER_INTERVAL or SCHEDULER_TIMES). Scheduler disabled."
    exit 0
fi

# --- Helpers ---

minutes_until() {
    local target_h="$1"
    local target_m="$2"
    local now_h now_m now_epoch target_epoch diff_min

    now_h=$(date +%H)
    now_m=$(date +%M)
    now_epoch=$(date -d "${now_h}:${now_m}" +%s 2>/dev/null || date -j -H%H -M%M +%s 2>/dev/null || echo "$(date +%s)")
    target_epoch=$(date -d "${target_h}:${target_m}" +%s 2>/dev/null || date -j -H"${target_h}" -M"${target_m}" +%s 2>/dev/null)

    if [[ -z "$target_epoch" ]]; then
        return 1
    fi

    diff_min=$(( (target_epoch - now_epoch) / 60 ))

    # If target is in the past, schedule for tomorrow
    if (( diff_min <= 0 )); then
        diff_min=$(( diff_min + 1440 ))
    fi

    echo "$diff_min"
}

parse_times() {
    local times_str="$1"
    local -a times=()
    local t

    for t in $times_str; do
        if [[ ! "$t" =~ ^[0-9]{1,2}:[0-9]{2}$ ]]; then
            log "ERROR: invalid time format '$t'. Expected HH:MM (e.g. 08:00)."
            exit 1
        fi
        times+=("$t")
    done

    if (( ${#times[@]} == 0 )); then
        log "ERROR: SCHEDULER_TIMES is empty."
        exit 1
    fi

    echo "${times[@]}"
}

find_next_timeslot() {
    local -a times=("$@")
    local min_wait=999999
    local t h m wait

    for t in "${times[@]}"; do
        h="${t%%:*}"
        m="${t##*:}"
        wait=$(minutes_until "$h" "$m")
        if (( wait < min_wait )); then
            min_wait=$wait
        fi
    done

    echo "$min_wait"
}

run_test() {
    log "Starting model test..."
    if ./run.sh ${SCHEDULER_ARGS:-}; then
        log "Test completed successfully."
    else
        log "WARNING: run.sh exited with error code $?."
    fi
}

# --- Interval mode ---

run_interval_mode() {
    local interval="$SCHEDULER_INTERVAL"

    if [[ ! "$interval" =~ ^[1-9][0-9]*$ ]]; then
        log "ERROR: SCHEDULER_INTERVAL must be a positive integer (minutes). Got: '$interval'."
        exit 1
    fi

    log "Interval mode: every ${interval} minute(s)."

    while true; do
        sleep $(( interval * 60 ))
        run_test
    done
}

# --- Times mode ---

run_times_mode() {
    local -a times
    read -ra times <<< "$(parse_times "$SCHEDULER_TIMES")"

    log "Times mode: run at ${times[*]}."

    while true; do
        local wait_minutes
        wait_minutes=$(find_next_timeslot "${times[@]}")
        local wait_seconds=$(( wait_minutes * 60 ))

        local target_h wait_h wait_m
        target_h=$(( wait_seconds / 3600 ))
        wait_m=$(( (wait_seconds % 3600) / 60 ))

        log "Next run in ${wait_minutes} minute(s) (~${target_h}h ${wait_m}m)."
        sleep "$wait_seconds"
        run_test
    done
}

# --- Entry point ---

log "Starting scheduler..."

trap 'log "Scheduler stopped."; exit 0' SIGTERM SIGINT

if [[ -n "${SCHEDULER_INTERVAL:-}" ]]; then
    run_interval_mode
else
    run_times_mode
fi
