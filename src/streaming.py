"""SSE streaming helpers — chunk parsing, error detection, and request forwarding."""

import asyncio
import json
import logging
from typing import AsyncGenerator

import aiohttp

from error_detection import is_quota_error, is_retryable_error

logger = logging.getLogger("gateway")


def parse_chunk_metadata(chunk_str: str) -> tuple[str | None, int | None]:
    """Parse finish_reason and completion_tokens from SSE chunk.

    Returns (finish_reason, completion_tokens) or (None, None) if not found.
    """
    if not chunk_str.startswith("data: "):
        return None, None

    payload = chunk_str[6:].strip()
    if payload == "[DONE]":
        return None, None

    try:
        data = json.loads(payload)
        finish_reason = None
        completion_tokens = None

        choices = data.get("choices", [])
        if choices:
            finish_reason = choices[0].get("finish_reason")

        usage = data.get("usage", {})
        if usage:
            completion_tokens = usage.get("completion_tokens")

        return finish_reason, completion_tokens
    except (json.JSONDecodeError, KeyError, IndexError):
        return None, None


def extract_chunk_content(chunk_str: str) -> str:
    """Extract content/text from an SSE chunk if present.

    Tries OpenAI-style `delta.content`, `delta.text`, and `choices[0].text`.
    Returns empty string if no content is found.
    """
    if not chunk_str.startswith("data: "):
        return ""

    payload = chunk_str[6:].strip()
    if payload == "[DONE]":
        return ""

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""

    choices = data.get("choices", [])
    if not choices:
        return ""

    choice = choices[0]
    delta = choice.get("delta", {})
    if "content" in delta and delta["content"]:
        return str(delta["content"])
    if "text" in delta and delta["text"]:
        return str(delta["text"])
    if "text" in choice and choice["text"]:
        return str(choice["text"])
    return ""


def has_tool_calls_in_chunk(chunk_str: str) -> bool:
    """Check if an SSE chunk contains tool_calls in the delta.

    Returns True if the chunk has tool_calls, indicating a function call response.
    """
    if not chunk_str.startswith("data: "):
        return False

    payload = chunk_str[6:].strip()
    if payload == "[DONE]":
        return False

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False

    choices = data.get("choices", [])
    if not choices:
        return False

    delta = choices[0].get("delta", {})
    return bool(delta.get("tool_calls"))


def is_useless_response(finish_reason: str | None, completion_tokens: int | None) -> bool:
    """Check if a streaming response is useless (empty/too short).

    Returns True if:
    - finish_reason is "length" (context overflow), OR
    - finish_reason is "stop" and completion_tokens <= 5 (empty/minimal response)
    """
    if finish_reason == "length":
        return True
    if finish_reason == "stop" and completion_tokens is not None and completion_tokens <= 5:
        return True
    return False


def parse_sse_error(line: str, switch_on_any_error: bool = False) -> dict | None:
    """If an SSE data line contains a switchable error JSON, return error_info dict.

    Checks if the line looks like ``data: {"error":...}`` and if so extracts
    the HTTP status code and full body text.

    Returns error_info if:
    - ``switch_on_any_error`` is True (any SSE error triggers a switch), or
    - the error matches retryable or quota patterns (provider-level errors
      like rate limits, resource exhaustion).

    Model-level errors (invalid_request_error, tool validation, etc.)
    are treated as normal chunks and passed through to the client.

    Returns ``None`` for normal chunks, ``[DONE]`` markers, non-JSON data,
    and non-switchable errors.
    """
    if not line.startswith("data: ") or "[DONE]" in line:
        return None
    raw = line.removeprefix("data: ").strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if "error" not in obj:
        return None
    err = obj["error"]
    if isinstance(err, dict):
        code = err.get("code", 500)
    else:
        code = 500
    try:
        status = int(code) if code is not None else 500
    except (ValueError, TypeError):
        status = 500

    if switch_on_any_error:
        return {"status": status, "body": raw}

    if not is_retryable_error(status, raw) and not is_quota_error(raw):
        return None

    return {"status": status, "body": raw}


async def forward_streaming(
    state,  # GatewayState — type hint omitted to avoid circular import
    model: dict,
    body: dict,
    user_agent: str,
    stream_idle_timeout: float = 30.0,
) -> AsyncGenerator[tuple[str, dict | None], None]:
    """Forward request and yield SSE chunks. Yields (chunk_str, error_info).

    If no data is received for *stream_idle_timeout* seconds, yields a stall
    error (status 408) so the caller can switch to the next model.
    """
    # Late imports to avoid circular dependency with gateway
    from convert_google import GoogleConverter

    provider_name = model["provider"]
    provider = state._resolve_provider(provider_name)
    if not provider:
        error = {"error": {"message": f"Unknown provider: {provider_name}", "type": "gateway_error"}}
        yield json.dumps(error), error
        return

    ptype = provider.get("type", "openai")
    model_id = model["model_id"]

    # Build payload — only replace model name, pass everything else as-is
    if ptype == "google":
        payload = GoogleConverter.openai_to_provider(body, model_id)
    else:
        payload = {body_key: body[body_key] for body_key in body}
        payload["model"] = model_id

    # Debug: log what we're actually sending
    msgs = payload.get("messages", [])
    for i, m in enumerate(msgs):
        has_tc = "tool_calls" in m
        tc_info = f", tool_calls={len(m['tool_calls'])}" if has_tc else ""
        logger.debug(f"[SEND {i}] role={m['role']}{tc_info}")
    extra_keys = [k for k in payload if k not in ("messages", "model")]
    logger.debug(f"[SEND_EXTRA] {extra_keys}")

    url = state._get_provider_url(provider, model_id)
    headers = state._get_provider_headers(provider, user_agent)

    sock_read = stream_idle_timeout if stream_idle_timeout > 0 else None
    http_timeout = aiohttp.ClientTimeout(sock_read=sock_read, sock_connect=10)
    try:
        async with aiohttp.ClientSession(timeout=http_timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    resp_body = await resp.text()
                    yield json.dumps({"error": {"message": f"Provider returned {resp.status}", "type": "provider_error", "code": resp.status, "detail": resp_body[:500]}}), {"status": resp.status, "body": resp_body}
                    return

                # Stream SSE chunks — detect errors embedded in SSE events
                switch_any = state.switch_on_any_error
                async for line in resp.content:
                    decoded = line.decode("utf-8", errors="replace")
                    if decoded.strip():
                        error_info = parse_sse_error(decoded, switch_on_any_error=switch_any)
                        if error_info:
                            logger.debug(f"[SSE_ERROR] {model_id} ({model['provider']}): {decoded[:200].strip()}")
                            yield decoded, error_info
                        elif ptype == "google" and decoded.startswith("data: "):
                            raw = decoded[6:].strip()
                            try:
                                google_data = json.loads(raw)
                                openai_chunk = GoogleConverter.provider_sse_to_openai(google_data)
                                if openai_chunk:
                                    yield f"data: {json.dumps(openai_chunk)}\n\n", None
                            except json.JSONDecodeError:
                                yield decoded, None
                        else:
                            yield decoded, None
                logger.debug(f"[STREAM_END] {model_id} ({model['provider']}) stream completed")
    except asyncio.TimeoutError:
        yield json.dumps({"error": {"message": "Stream stalled — no data received", "type": "stall_timeout"}}), {"status": 408, "body": "stall timeout"}
    except aiohttp.ClientError as e:
        # Connection / DNS failures (e.g. ClientConnectorDNSError from a transient
        # gaierror, ServerDisconnectedError, etc.). Treat as a switchable error so
        # the caller advances to the next model instead of crashing the ASGI
        # request — mirrors the non-streaming path in _handle_non_stream.
        err = str(e) or e.__class__.__name__
        logger.warning(
            f"[CONN_ERROR] {model_id} ({model['provider']}): {e.__class__.__name__}: {err}"
        )
        yield (
            json.dumps({"error": {"message": f"Connection error: {err}", "type": "connection_error"}}),
            {"status": 408, "body": err},
        )
