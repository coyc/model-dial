#!/usr/bin/env python3
"""
Test models by sending a chat completion request to each.

Reads result-requirements.json (output of check-requirements.py), filters to
models that passed requirements checking, and tests each with a simple
prompt via streaming, measuring TTFT + total response time.

Outputs result-test.json with test results merged with pre-computed
capabilities data.

Usage:
    python3 test-models.py --input result-requirements.json

    # With custom config
    python3 test-models.py --input result-requirements.json --concurrency 5
"""

import asyncio
import json
import re
import sys
import time
import argparse
import ssl
from pathlib import Path

from credential_manager import ProviderCredentialManager
from error_detection import is_quota_error

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip3 install aiohttp", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config — load defaults from config.json
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(_PROJECT_ROOT / "config.json", "r") as _f:
    _CFG = json.load(_f)

USER_AGENT = _CFG.get("user_agent", "models-tester/1.0")
DEFAULTS = _CFG.get("test_models", {})
DEFAULT_PROMPT = DEFAULTS.get("prompt", "What is 2+2? Reply with just the number.")
DEFAULT_EXPECTED = DEFAULTS.get("expected", "4")
DEFAULT_CONCURRENCY = DEFAULTS.get("concurrency", 10)
DEFAULT_TIMEOUT = DEFAULTS.get("timeout", 15)
DEFAULT_MAX_TOKENS = DEFAULTS.get("max_tokens", 128)

DEFAULT_CONFIG = _PROJECT_ROOT / "config.json"
DEFAULT_PROVIDERS = _PROJECT_ROOT / "providers.json"
REQUIREMENTS = _CFG.get("requirements", {})


def _resolve_providers(providers_creds: dict) -> dict:
    """Convert providers.json format to flat {type, model_filter}.

    Credential selection and rotation are handled by ProviderCredentialManager.
    """
    resolved = {}
    for prov_name, prov_data in providers_creds.items():
        entry = {
            "type": prov_data.get("type", "openai"),
        }
        if prov_data.get("model_filter"):
            entry["model_filter"] = prov_data["model_filter"]
        resolved[prov_name] = entry
    return resolved


PROVIDERS_CFG = _resolve_providers(
    json.loads(DEFAULT_PROVIDERS.read_text()) if DEFAULT_PROVIDERS.exists() else {}
)


# ---------------------------------------------------------------------------
# Answer validation
# ---------------------------------------------------------------------------
def validate_answer(answer: str | None, expected: str = DEFAULT_EXPECTED) -> bool:
    """Check if the model answer contains valid JSON with the expected value.

    Expects JSON format: {"answer": <number>, "reasoning": "<text>"}
    Handles models that output thinking before JSON — extracts the last JSON object.
    Returns True only if:
    1. A valid JSON object with "answer" field is found
    2. "answer" field matches expected value
    """
    if not answer:
        return False

    text = answer.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try to parse entire string as JSON first
    data = _try_parse_json(text)

    # If whole-string parse failed, find the LAST JSON object in the text
    if data is None:
        data = _extract_last_json(text)

    if data is None or not isinstance(data, dict):
        return False

    answer_val = data.get("answer")
    if answer_val is None:
        return False

    # Compare as string (handles both int 4 and string "4")
    return str(answer_val) == str(expected)


def _try_parse_json(text: str) -> dict | None:
    """Try to parse text as JSON. Returns dict or None."""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_last_json(text: str) -> dict | None:
    """Find a JSON object in text using raw_decode.

    Handles both patterns:
    - Chatty models: JSON first, then extra text (raw_decode stops at closing })
    - Thinking models: thinking first, then JSON (scan from end)
    """
    _decoder = json.JSONDecoder()
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '{':
            try:
                data, _ = _decoder.raw_decode(text, i)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def load_api_keys(providers_path: Path) -> dict[str, str]:
    """Load API keys from providers.json. Returns { provider_id: api_key }."""
    if not providers_path.exists():
        return {}
    with open(providers_path) as f:
        providers_creds = json.load(f)
    keys = {}
    for pid, prov_data in providers_creds.items():
        creds = prov_data.get("credentials", [])
        if creds:
            # Use current credential, or first if none marked current
            current = creds[0]
            for cred in creds:
                if cred.get("current"):
                    current = cred
                    break
            api_key = current.get("api_key")
            if api_key:
                keys[pid] = api_key
    return keys


# ---------------------------------------------------------------------------
# Build request for different provider types
# ---------------------------------------------------------------------------
def build_openai_request(base_url: str, model_id: str, api_key: str, prompt: str) -> dict:
    """Build an OpenAI-compatible streaming request."""
    return {
        "url": f"{base_url.rstrip('/')}/chat/completions",
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        "body": {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": True,
        },
    }


def build_google_request(base_url: str, model_id: str, api_key: str, prompt: str) -> dict:
    """Build a Google Generative AI streaming request."""
    return {
        "url": f"{base_url.rstrip('/')}/models/{model_id}:streamGenerateContent?alt=sse",
        "headers": {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "User-Agent": USER_AGENT,
        },
        "body": {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": DEFAULT_MAX_TOKENS},
        },
    }


# ---------------------------------------------------------------------------
# Parse streaming SSE response
# ---------------------------------------------------------------------------
def parse_openai_sse_line(line: str) -> tuple[str | None, str | None]:
    """Extract content and reasoning_content from an OpenAI SSE data line.
    Returns (content, reasoning_content) tuple.
    """
    if not line.startswith("data: "):
        return None, None
    payload = line[6:].strip()
    if payload == "[DONE]":
        return None, None
    try:
        chunk = json.loads(payload)
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        # Some models use "reasoning", others "reasoning_content"
        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        return content, reasoning
    except (json.JSONDecodeError, IndexError, KeyError):
        return None, None


def parse_google_sse_line(line: str) -> tuple[str | None, None]:
    """Extract content from a Google SSE data line.
    Returns (content, None) tuple for compatibility.
    """
    if not line.startswith("data: "):
        return None, None
    try:
        chunk = json.loads(line[6:].strip())
        candidates = chunk.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text"), None
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return None, None


# ---------------------------------------------------------------------------
# Test a single model
# ---------------------------------------------------------------------------
async def test_model(
    session: aiohttp.ClientSession,
    provider_id: str,
    model_id: str,
    base_url: str,
    api_key: str,
    api_type: str,
    prompt: str,
    timeout: int,
    expected: str = DEFAULT_EXPECTED,
) -> dict:
    """Test a single model and return timing metrics."""
    result = {
        "provider": provider_id,
        "model_id": model_id,
        "test": {
            "status": "error",
            "ttft_ms": None,
            "total_ms": None,
            "answer": None,
            "correct": None,
            "error": None,
            "info": None,
        },
    }

    start = time.monotonic()
    try:
        if api_type == "google":
            req = build_google_request(base_url, model_id, api_key, prompt)
            parse_line = parse_google_sse_line
        else:
            req = build_openai_request(base_url, model_id, api_key, prompt)
            parse_line = parse_openai_sse_line

        ttft = None
        answer = ""
        raw_lines_sample = []  # first 5 non-empty data lines for diagnostics

        async with session.post(
            req["url"],
            headers=req["headers"],
            json=req["body"],
            timeout=aiohttp.ClientTimeout(total=timeout),
            ssl=False,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result["test"]["error"] = f"HTTP {resp.status}: {body[:200]}"
                result["test"]["total_ms"] = round((time.monotonic() - start) * 1000)
                return result

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                # Collect first 5 data lines for diagnostics (only if answer is still empty)
                if not answer and line.startswith("data: ") and len(raw_lines_sample) < 5:
                    raw_lines_sample.append(line[:300])
                content, reasoning = parse_line(line)
                if content:
                    if ttft is None:
                        ttft = round((time.monotonic() - start) * 1000)
                    answer += content
                elif reasoning:
                    # Track reasoning content for thinking models
                    if ttft is None:
                        ttft = round((time.monotonic() - start) * 1000)
                    answer += reasoning

        total = round((time.monotonic() - start) * 1000)
        result["test"]["status"] = "ok"
        result["test"]["ttft_ms"] = ttft
        result["test"]["total_ms"] = total
        result["test"]["answer"] = answer.strip()
        result["test"]["correct"] = validate_answer(answer, expected)

        # Diagnostics: if answer is empty, include what we received
        if not answer:
            if raw_lines_sample:
                result["test"]["info"] = f"empty answer, received {len(raw_lines_sample)} data lines"
                result["test"]["info_lines"] = raw_lines_sample
            else:
                result["test"]["info"] = "empty answer, no data lines received"

    except asyncio.TimeoutError:
        result["test"]["error"] = f"Timeout after {timeout}s"
        result["test"]["total_ms"] = round((time.monotonic() - start) * 1000)
    except aiohttp.ClientError as e:
        result["test"]["error"] = str(e)
    except Exception as e:
        result["test"]["error"] = f"{type(e).__name__}: {e}"

    return result


def _make_error_result(provider_id: str, model_id: str, error_msg: str) -> dict:
    """Create a failed test result for a model."""
    return {
        "provider": provider_id,
        "model_id": model_id,
        "test": {
            "status": "error",
            "ttft_ms": None,
            "total_ms": None,
            "answer": None,
            "correct": None,
            "error": error_msg,
            "info": None,
        },
    }


# ---------------------------------------------------------------------------
# Semaphore limiter per provider (to avoid rate limits)
# ---------------------------------------------------------------------------
class ProviderSemaphore:
    """Track per-provider concurrency."""

    def __init__(self, max_per_provider: int = 3):
        self._max = max_per_provider
        self._locks: dict[str, asyncio.Semaphore] = {}

    def get(self, provider_id: str) -> asyncio.Semaphore:
        if provider_id not in self._locks:
            self._locks[provider_id] = asyncio.Semaphore(self._max)
        return self._locks[provider_id]


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
async def run_tests(
    checked_results: list[dict],
    prompt: str,
    concurrency: int,
    timeout: int,
    expected: str = DEFAULT_EXPECTED,
    providers_path: Path = DEFAULT_PROVIDERS,
) -> dict:
    """Run tests on models that passed requirements check.

    checked_results: flat list of {provider, model_id, capabilities,
                     requirements_breakdown, rejected}
                     from check-requirements.py output.
    """
    cred_manager = ProviderCredentialManager(providers_path)
    sem = asyncio.Semaphore(concurrency)
    provider_sem = ProviderSemaphore(max_per_provider=3)

    results = []

    # Build lookup for pre-computed data
    checked_lookup: dict[tuple[str, str], dict] = {}
    for entry in checked_results:
        key = (entry["provider"], entry["model_id"])
        checked_lookup[key] = {
            "capabilities": entry.get("capabilities"),
            "requirements_breakdown": entry.get("requirements_breakdown", {}),
            "rejected": entry.get("rejected", False),
        }

    # Group models by provider
    provider_models: dict[str, list[str]] = {}
    for entry in checked_results:
        pid = entry["provider"]
        if pid not in provider_models:
            provider_models[pid] = []
        provider_models[pid].append(entry["model_id"])

    async def limited_test(
        session: aiohttp.ClientSession,
        provider_id: str,
        model_id: str,
        prompt: str,
        timeout: int,
        expected: str,
    ) -> dict:
        psem = provider_sem.get(provider_id)
        max_attempts = cred_manager.credential_count(provider_id) or 1
        first_cred = cred_manager.get_credential(provider_id)
        if not first_cred:
            return _make_error_result(provider_id, model_id, "no API key in config.json")

        for attempt in range(max_attempts):
            base_url, api_key = first_cred.get("base_url", ""), first_cred.get("api_key", "")
            api_type = "google" if "googleapis.com" in base_url else "openai"

            async with sem, psem:
                result = await test_model(
                    session, provider_id, model_id, base_url, api_key, api_type, prompt, timeout, expected
                )

            if result["test"]["status"] == "ok":
                return result

            # Check for quota error — rotate credential and retry
            error = result["test"].get("error", "")
            body = error.split(": ", 1)[1] if ": " in error else error
            if is_quota_error(body) and attempt < max_attempts - 1:
                new_cred = cred_manager.advance_credential(provider_id)
                if new_cred:
                    first_cred = new_cred
                    continue

            return result

        return _make_error_result(provider_id, model_id, "unexpected: all attempts exhausted")

    connector = aiohttp.TCPConnector(limit=concurrency, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for pid, model_ids in provider_models.items():
            prov_cfg = PROVIDERS_CFG.get(pid)
            if not prov_cfg:
                print(f"  [skip] {pid}: no provider config in config.json", file=sys.stderr)
                continue

            if not cred_manager.get_credential(pid):
                print(f"  [skip] {pid}: no API key in config.json", file=sys.stderr)
                continue

            for model_id in model_ids:
                tasks.append(
                    limited_test(session, pid, model_id, prompt, timeout, expected)
                )

        total = len(tasks)
        print(f"[info] Testing {total} models with concurrency={concurrency}...", file=sys.stderr)

        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro

            # Merge with pre-computed capabilities data
            key = (result["provider"], result["model_id"])
            pre = checked_lookup.get(key, {})
            result["capabilities"] = pre.get("capabilities")
            result["requirements_breakdown"] = pre.get("requirements_breakdown", {})
            result["rejected"] = pre.get("rejected", False)

            results.append(result)
            done += 1
            t = result.get("test", {})
            status = "✓" if t.get("status") == "ok" else "✗"
            ttft = f"{t['ttft_ms']}ms" if t.get('ttft_ms') else "-"
            total_ms = f"{t['total_ms']}ms" if t.get('total_ms') else "-"
            info = f" [{t['info']}]" if t.get("info") else ""
            error = ""
            if t.get("error"):
                err = t["error"]
                if err.startswith("HTTP "):
                    error = f" error={err.split(':')[0]}"
                else:
                    error = f" error={err[:60]}"
            print(
                f"  [{done}/{total}] {status} {result['provider']}/{result['model_id'][:40]:<40} ttft={ttft} total={total_ms}{error}{info}",
                file=sys.stderr,
            )

    # Apply per-category test thresholds (ttft_ms, total_ms)
    stat_filtered = 0
    for r in results:
        if r.get("test", {}).get("status") != "ok":
            continue
        breakdown = r.get("requirements_breakdown", {})
        t = r.get("test", {})
        for cat in list(breakdown):
            if not breakdown[cat]:
                continue
            test_reqs = REQUIREMENTS.get(cat, {}).get("test", {})
            ttft_limit = test_reqs.get("ttft_ms")
            total_limit = test_reqs.get("total_ms")
            if ttft_limit is not None and t.get("ttft_ms") is not None and t["ttft_ms"] > ttft_limit:
                breakdown[cat] = False
                stat_filtered += 1
            elif total_limit is not None and t.get("total_ms") is not None and t["total_ms"] > total_limit:
                breakdown[cat] = False
                stat_filtered += 1

    if stat_filtered:
        print(f"[info] Filtered {stat_filtered} category assignments by test thresholds", file=sys.stderr)

    # Aggregate
    ok = [r for r in results if r.get("test", {}).get("status") == "ok"]
    failed = [r for r in results if r.get("test", {}).get("status") != "ok"]
    correct = [r for r in results if r.get("test", {}).get("correct") is True]

    # Per-provider summary
    summary = {}
    for r in results:
        t = r.get("test", {})
        pid = r["provider"]
        if pid not in summary:
            summary[pid] = {"total": 0, "ok": 0, "failed": 0, "correct": 0, "avg_ttft_ms": 0, "avg_total_ms": 0}
        summary[pid]["total"] += 1
        if t.get("status") == "ok":
            summary[pid]["ok"] += 1
            if t.get("ttft_ms") is not None:
                summary[pid]["avg_ttft_ms"] += t["ttft_ms"]
            if t.get("total_ms") is not None:
                summary[pid]["avg_total_ms"] += t["total_ms"]
        else:
            summary[pid]["failed"] += 1
        if t.get("correct") is True:
            summary[pid]["correct"] += 1

    for pid, s in summary.items():
        if s["ok"] > 0:
            s["avg_ttft_ms"] = round(s["avg_ttft_ms"] / s["ok"])
            s["avg_total_ms"] = round(s["avg_total_ms"] / s["ok"])

    return {
        "prompt": prompt,
        "concurrency": concurrency,
        "total": len(results),
        "ok": len(ok),
        "failed": len(failed),
        "correct": len(correct),
        "summary": summary,
        "requirements": REQUIREMENTS,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test models from check-requirements.py output")
    parser.add_argument("--input", type=Path, help="Input JSON file (default: read from stdin)")
    parser.add_argument("--providers-file", type=Path, default=DEFAULT_PROVIDERS, help="Path to providers.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Test prompt")
    parser.add_argument("--expected", default=DEFAULT_EXPECTED, help="Expected answer to validate against")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max parallel requests")
    parser.add_argument("--test-timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout per request (seconds)")
    parser.add_argument("--providers", help="Comma-separated provider IDs to test")
    args = parser.parse_args()

    # Read input
    if args.input:
        with open(args.input) as f:
            checked_data = json.load(f)
    else:
        checked_data = json.load(sys.stdin)

    # Filter to eligible models
    all_results = checked_data.get("results", [])
    eligible = [r for r in all_results if not r.get("rejected") and any(r.get("requirements_breakdown", {}).values())]

    if not eligible:
        print("[error] No eligible models found (all rejected or don't meet requirements)", file=sys.stderr)
        sys.exit(1)

    print(
        f"[info] {len(eligible)} of {len(all_results)} models eligible for testing",
        file=sys.stderr,
    )

    providers_filter = set(args.providers.split(",")) if args.providers else None

    # Apply provider filter if set
    if providers_filter:
        eligible = [r for r in eligible if r["provider"] in providers_filter]
        if not eligible:
            print("[error] No eligible models match the specified providers", file=sys.stderr)
            sys.exit(1)

    result = asyncio.run(
        run_tests(
            eligible, args.prompt, args.concurrency, args.test_timeout, args.expected,
            providers_path=args.providers_file,
        )
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
