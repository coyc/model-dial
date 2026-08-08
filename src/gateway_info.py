"""Gateway message cleaning — strip internal fields before forwarding to providers.

Two categories of fields are stripped from inbound messages:

1. **Debug tags** — status information injected into streaming responses by
   the gateway itself (model name, credential position). Shown in the OpenCode
   chat UI for visibility, but stripped from conversation history before
   forwarding to providers (so the model never sees them).

2. **reasoning_content** — internal chain-of-thought from extended thinking
   models (DeepSeek R1, Claude with thinking, etc.). Must be stripped when
   the gateway switches to a different model because each model has its own
   incompatible reasoning format; passing one model's reasoning to another
   causes ``invalid_request_error`` (e.g. "The reasoning_content in the
   thinking mode must be passed back to the API").

Format (single line at the end of assistant message)::

    <debug>nvidia/nemotron-3-ultra-550b-a55b:free (openrouter), creds=13/15</debug>
"""

import re

DEBUG_TAG_OPEN = "<debug>"
DEBUG_TAG_CLOSE = "</debug>"

# Matches a single <debug>...</debug> tag including leading and trailing newlines.
# Non-greedy .*? to handle multiple tags in one message.
_DEBUG_TAG_RE = re.compile(
    rf"\n?{re.escape(DEBUG_TAG_OPEN)}.*?{re.escape(DEBUG_TAG_CLOSE)}\n?",
)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def build_debug_tag(
    model_id: str,
    provider_name: str,
    credential_position: str,
) -> str:
    """Build the gateway debug tag string (without trailing newline)."""
    return f"\n{DEBUG_TAG_OPEN}{model_id} ({provider_name}), creds={credential_position}{DEBUG_TAG_CLOSE}"


# ---------------------------------------------------------------------------
# Stripping (inbound)
# ---------------------------------------------------------------------------

def strip_debug_tags(text: str) -> str:
    """Remove all gateway debug tags from *text*."""
    return _DEBUG_TAG_RE.sub("", text)


def clean_messages(messages: list[dict]) -> list[dict]:
    """Strip gateway debug tags from assistant messages.

    Assistant messages whose content becomes empty after stripping are
    removed entirely.  Non-assistant messages are passed through unchanged.
    """
    cleaned: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            cleaned.append(msg)
            continue

        content = msg.get("content")
        if not isinstance(content, str):
            cleaned.append(msg)
            continue

        new_content = strip_debug_tags(content)
        if new_content.strip():
            cleaned.append({**msg, "content": new_content})
        # else: message contained only a debug tag → drop it

    return cleaned


def strip_reasoning_content(messages: list[dict]) -> list[dict]:
    """Replace reasoning_content with empty string in assistant messages.

    Extended thinking models (DeepSeek R1, Claude with thinking, etc.) include
    a ``reasoning_content`` field in their responses alongside ``content``.
    When switching between models, the reasoning text from model A should not
    be forwarded to model B (incompatible format), BUT some models (DeepSeek)
    require the field to be present (even if empty) once thinking mode is active.

    Returns a new list with ``reasoning_content`` replaced with empty string
    in assistant messages. Non-assistant messages and assistant messages without
    the field are passed through unchanged.
    """
    cleaned: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant" and "reasoning_content" in msg:
            msg = {**msg, "reasoning_content": ""}
        cleaned.append(msg)
    return cleaned
