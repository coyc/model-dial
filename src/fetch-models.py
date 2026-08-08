#!/usr/bin/env python3
"""
Fetch available models from all providers configured in providers.json.

Reads provider credentials (api_key, base_url, type) from providers.json,
queries each provider's /v1/models endpoint, and outputs a combined JSON result.

Usage:
    python3 fetch-models.py [--providers-file PATH]

Output: { "providers": [{ provider, count, models: [{ id, owned_by }] }] }
"""

import json
import sys
import urllib.request
import urllib.error
import ssl
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.json"
_DEFAULT_PROVIDERS = _PROJECT_ROOT / "providers.json"

# Load user_agent from config
with open(_DEFAULT_CONFIG, "r") as _f:
    _CFG = json.load(_f)
USER_AGENT = _CFG.get("user_agent", "models-tester/1.0")
FETCH_TIMEOUT = 30


def _resolve_providers(providers_creds: dict) -> dict:
    """Convert providers.json format to flat {type, base_url, api_key, model_filter}.

    Uses the ``current: true`` credential for each provider (or the first
    credential if none is marked current).
    """
    resolved = {}
    for prov_name, prov_data in providers_creds.items():
        creds = prov_data.get("credentials", [])
        if not creds:
            continue
        # Find current credential
        current = creds[0]
        for cred in creds:
            if cred.get("current"):
                current = cred
                break
        entry = {
            "type": prov_data.get("type", "openai"),
            "base_url": current.get("base_url"),
            "api_key": current.get("api_key"),
        }
        if prov_data.get("model_filter"):
            entry["model_filter"] = prov_data["model_filter"]
        resolved[prov_name] = entry
    return resolved


# ---------------------------------------------------------------------------
# Fetch models from a provider
# ---------------------------------------------------------------------------
def fetch_openai_models(base_url: str, api_key: str) -> list[dict]:
    """Fetch models from an OpenAI-compatible /v1/models endpoint."""
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    })
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT, context=ctx) as resp:
        data = json.loads(resp.read())
    return data.get("data", [])


def fetch_google_models(base_url: str, api_key: str) -> list[dict]:
    """Fetch models from Google Generative AI API."""
    url = f"{base_url.rstrip('/')}/models?key={api_key}"
    req = urllib.request.Request(url, headers={
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    })
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT, context=ctx) as resp:
        data = json.loads(resp.read())
    return [
        {"id": m.get("name", "").replace("models/", ""), "owned_by": "google"}
        for m in data.get("models", [])
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch available models from providers in providers.json")
    parser.add_argument("--providers-file", type=Path, default=_DEFAULT_PROVIDERS, help="Path to providers.json")
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated list of provider IDs to fetch (default: all)")
    args = parser.parse_args()

    providers_path = args.providers_file
    if not providers_path.exists():
        print(f"Error: providers file not found: {providers_path}", file=sys.stderr)
        sys.exit(1)

    with open(providers_path) as f:
        providers_creds = json.load(f)

    providers = _resolve_providers(providers_creds)
    if not providers:
        print("Error: no providers in providers.json", file=sys.stderr)
        sys.exit(1)

    # Filter by --providers if specified
    if args.providers is not None:
        requested = [p.strip() for p in args.providers.split(",") if p.strip()]
        providers = {k: v for k, v in providers.items() if k in requested}
        if not providers:
            print(f"Error: none of the requested providers found: {args.providers}", file=sys.stderr)
            sys.exit(1)

    # Query each provider
    results = []
    errors = []

    for provider_id, prov_cfg in providers.items():
        api_key = prov_cfg.get("api_key")
        base_url = prov_cfg.get("base_url")
        api_type = prov_cfg.get("type", "openai")

        if not api_key:
            print(f"[skip] {provider_id}: no api_key", file=sys.stderr)
            continue

        print(f"[fetch] {provider_id}: {base_url}", file=sys.stderr)
        try:
            if api_type == "google":
                models = fetch_google_models(base_url, api_key)
            elif base_url:
                models = fetch_openai_models(base_url, api_key)
            else:
                errors.append({"provider": provider_id, "error": "No base_url"})
                print(f"[error] {provider_id}: no base_url", file=sys.stderr)
                continue

            # Filter models by model_filter substring if configured
            model_filter = prov_cfg.get("model_filter")
            if model_filter:
                before = len(models)
                models = [m for m in models if model_filter in m["id"]]
                print(f"[filter] {provider_id}: {before} -> {len(models)} models (filter: '{model_filter}')", file=sys.stderr)

            results.append({
                "provider": provider_id,
                "base_url": base_url,
                "count": len(models),
                "models": [{"id": m["id"], "owned_by": m.get("owned_by", provider_id)} for m in models],
            })
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            msg = f"HTTP {e.code}: {e.reason} — {body}"
            errors.append({"provider": provider_id, "error": msg})
            print(f"[error] {provider_id}: {msg}", file=sys.stderr)
        except Exception as e:
            errors.append({"provider": provider_id, "error": str(e)})
            print(f"[error] {provider_id}: {e}", file=sys.stderr)

    output = {"providers": results}
    if errors:
        output["errors"] = errors
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
