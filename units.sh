#!/usr/bin/env bash
#
# units.sh — run models-tester unit tests
#
# All tests run without real HTTP requests (everything is mocked).
#
# Usage:
#   ./units.sh              — all tests
#   ./units.sh -k "TestModel" — only tests with "TestModel" in name
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found" >&2
  exit 1
fi

exec python3 -m pytest "$SCRIPT_DIR/tests" -v "$@"
