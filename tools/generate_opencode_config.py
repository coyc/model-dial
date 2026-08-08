#!/usr/bin/env python3
"""Generate OpenCode provider config from config.json requirements.

Reads config.json, extracts the requirements and gateway settings,
and outputs a ready-to-use OpenCode provider configuration.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.json")
GATEWAY_PORT = 8765
GATEWAY_BASE_URL = f"http://localhost:{GATEWAY_PORT}/v1"


def _build_model_entry(category: str, models_catalog: dict) -> dict:
    """Build a single OpenCode model entry from a requirement's models_catalog."""
    entry: dict = {
        "name": f"{category.capitalize()} (auto)",
    }

    if models_catalog.get("tool_call") is True:
        entry["tool_call"] = True
    if models_catalog.get("reasoning") is True:
        entry["reasoning"] = True
    if models_catalog.get("attachment") is True:
        entry["attachment"] = True

    if "modalities" in models_catalog:
        entry["modalities"] = {
            "input": models_catalog["modalities"]["input"],
            "output": ["text"],
        }

    if "limit" in models_catalog:
        entry["limit"] = models_catalog["limit"]

    return entry


def generate_opencode_config(config: dict) -> dict:
    """Generate OpenCode provider config dict from Model Dial config."""
    requirements = config.get("requirements", {})
    gateway = config.get("gateway", {})
    api_key = gateway.get("api_key", "")

    models = {}
    for category, req in sorted(requirements.items()):
        models_catalog = req.get("models_catalog", {})
        models[category] = _build_model_entry(category, models_catalog)

    return {
        "provider": {
            "gateway": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Model-Dial Gateway",
                "options": {
                    "baseURL": GATEWAY_BASE_URL,
                    "apiKey": api_key,
                },
                "models": models,
            }
        }
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenCode provider config from config.json requirements."
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config.json (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation level (default: 4)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {args.config}: {e}", file=sys.stderr)
        return 1

    result = generate_opencode_config(config)
    output = json.dumps(result, indent=args.indent, ensure_ascii=False)

    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Config written to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
