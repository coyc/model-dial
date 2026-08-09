#!/usr/bin/env python3
"""Simple interactive client for the Model-Dial gateway.

Usage: python3 tools/client.py
"""

import asyncio
import json
import sys
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip3 install aiohttp", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"
DEFAULT_PORT = 8765
DEFAULT_TIMEOUT_MS = 60000

# ---------------------------------------------------------------------------
# Helper functions — pure, testable without I/O
# ---------------------------------------------------------------------------


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load config.json, raise with a clear message on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def get_available_models(config: dict) -> list[str]:
    """Extract category names from requirements section, sorted alphabetically."""
    return sorted(config.get("requirements", {}).keys())


def parse_model_selection(raw_input: str, models: list[str]) -> str:
    """Parse user input into a model name.

    Accepts:
      - empty string → first model in list (default)
      - number (1-based index) → corresponding model
    Raises ValueError on invalid input.
    """
    if not models:
        raise ValueError("No models available")

    raw_input = raw_input.strip()

    if raw_input == "":
        return models[0]

    try:
        index = int(raw_input)
    except ValueError:
        raise ValueError(f"Invalid input: {raw_input!r}. Enter a number (1-{len(models)}).")

    if index < 1 or index > len(models):
        raise ValueError(f"Invalid selection: {index}. Choose between 1 and {len(models)}.")

    return models[index - 1]


def get_gateway_base_url(port: int) -> str:
    """Build base URL: http://localhost:{port}/v1."""
    return f"http://localhost:{port}/v1"


def build_headers(api_key: str | None) -> dict[str, str]:
    """Build auth headers. Include Authorization only if api_key is set."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_request_payload(model: str, messages: list[dict]) -> dict:
    """Build the OpenAI-compatible request payload."""
    return {
        "model": model,
        "messages": messages,
        "stream": False,
    }


def extract_response_content(response_json: dict) -> str:
    """Extract assistant content from an OpenAI chat completion response."""
    choices = response_json.get("choices", [])
    if not choices:
        raise ValueError("Response contains no choices")
    content = choices[0].get("message", {}).get("content")
    if content is None:
        raise ValueError("Response choice has no content")
    return content


def extract_error_message(response_json: dict) -> str:
    """Extract error message from an OpenAI error response."""
    error = response_json.get("error", {})
    return error.get("message", "Unknown error")


def print_welcome(model: str, base_url: str) -> None:
    """Print banner with selected model, gateway URL, usage instructions."""
    print(f"\nModel: {model}")
    print(f"Gateway: {base_url}")
    print("\nType your message and press Enter to send.")
    print("Commands: /exit or /quit to exit.\n")
    print("=" * 40 + "\n")


# ---------------------------------------------------------------------------
# Async networking
# ---------------------------------------------------------------------------


async def send_message(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    model: str,
    messages: list[dict],
    timeout: float,
) -> dict:
    """POST to /v1/chat/completions, return parsed JSON response."""
    payload = build_request_payload(model, messages)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with session.post(url, json=payload, headers=headers, timeout=client_timeout) as resp:
        body = await resp.json()
        if resp.status != 200:
            error_msg = extract_error_message(body)
            raise RuntimeError(f"HTTP {resp.status}: {error_msg}")
        return body


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def main() -> int:
    """Main flow: load config, select model, run message loop."""
    # 1. Load config
    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # 2. Extract gateway settings
    gateway = config.get("gateway", {})
    api_key: str | None = gateway.get("api_key") or None
    port: int = gateway.get("port", DEFAULT_PORT)
    timeout_ms: int = gateway.get("request_timeout_ms", DEFAULT_TIMEOUT_MS)
    timeout_sec = timeout_ms / 1000

    # 3. Extract available models and let user choose
    models = get_available_models(config)
    if not models:
        print("Error: No models found in config.json requirements section.", file=sys.stderr)
        return 1

    try:
        selected = parse_model_selection(
            _prompt_model_selection(models),
            models,
        )
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        return 0

    base_url = get_gateway_base_url(port)
    url = f"{base_url}/chat/completions"
    headers = build_headers(api_key)

    # 4. Welcome
    print_welcome(selected, base_url)

    # 5. Message loop
    messages: list[dict] = []
    try:
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    user_input = input("You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nBye!")
                    return 0

                if user_input.lower() in ("/exit", "/quit"):
                    print("\nBye!")
                    return 0

                if not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})

                try:
                    response_json = await send_message(session, url, headers, selected, messages, timeout_sec)
                    content = extract_response_content(response_json)
                    messages.append({"role": "assistant", "content": content})
                    print(f"\nAssistant: {content}\n")
                except RuntimeError as exc:
                    print(f"\nError: {exc}\n")
                except aiohttp.ClientConnectorError:
                    print(f"\nError: Cannot connect to gateway at {base_url}")
                    print("Hint: Make sure the gateway is running — make start\n")
                except asyncio.TimeoutError:
                    print(f"\nError: Request timed out after {timeout_sec:.0f}s\n")
                except Exception as exc:  # noqa: BLE001 — keep interactive loop alive on unexpected errors
                    print(f"\nError: {exc}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        return 0

    return 0


def _prompt_model_selection(models: list[str]) -> str:
    """Display numbered model list and prompt user for selection."""
    print("\nAvailable models:")
    for i, model in enumerate(models, start=1):
        print(f"  {i}. {model}")

    default_name = models[0]
    return input(f"\nSelect model (default: {default_name}, press Enter): ")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
