#!/usr/bin/env python3
"""
Model capabilities lookup and requirements validation.

Loads the model catalog from URL or local cache, provides
functions to check whether a model meets requirements.

Supported requirement fields (models_catalog):
  - bool:      tool_call, attachment, reasoning, temperature,
               open_weights, structured_output
  - int/float: limit.context, limit.output, limit.input
  - list:      modalities.input, modalities.output
  - dict:      limit, modalities, interleaved
  - str:       family (prefix match, e.g. "claude" matches "claude-sonnet")

Supported name-based filters (model_id):
  - include:    list of substrings — reject if NONE found (case-insensitive)
  - exclude:    list of substrings — reject if any is found (case-insensitive)
  - min_params: minimum parameter count (parsed from ID)
  - providers:  list of provider names — reject if provider_id not in list (exact match)
"""

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Catalog cache fallback path
# ---------------------------------------------------------------------------
_CACHE_PATH = Path.home() / ".cache" / "opencode" / "models.json"


# ---------------------------------------------------------------------------
# Load catalog
# ---------------------------------------------------------------------------
def load_catalog(catalog_url: str | None = None) -> dict | None:
    """Load the model catalog from URL (preferred) or local cache.

    Returns {provider_id: {models: {model_id: {capabilities...}}}}
    or None if both sources fail.
    """
    # Try URL first
    if catalog_url:
        try:
            return _fetch_catalog(catalog_url)
        except Exception as e:
            print(f"[catalog] Failed to fetch from {catalog_url}: {e}", file=__import__("sys").stderr)

    # Fallback to local cache
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH) as f:
                data = json.load(f)
            print(f"[catalog] Loaded {len(data)} providers from cache: {_CACHE_PATH}", file=__import__("sys").stderr)
            return data
        except Exception as e:
            print(f"[catalog] Failed to read cache {_CACHE_PATH}: {e}", file=__import__("sys").stderr)
    else:
        print(f"[catalog] Cache not found: {_CACHE_PATH}", file=__import__("sys").stderr)

    return None


def _fetch_catalog(url: str) -> dict:
    """Fetch catalog JSON from URL."""
    print(f"[catalog] Fetching from {url}...", file=__import__("sys").stderr)
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "models-tester/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        data = json.loads(resp.read())
    if isinstance(data, dict):
        print(f"[catalog] Loaded {len(data)} providers from URL", file=__import__("sys").stderr)
    return data


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_id(mid: str) -> str:
    """Normalise a model ID for fuzzy matching.

    Strips suffixes, date patterns, quantization labels, and provider
    prefixes so that e.g.
    ``nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`` → ``nemotron_3_ultra_550b_a55b``
    and ``qwen3.7-max-2026-06-08`` → ``qwen3_7_max``.
    """
    n = mid.lower()
    # Strip :free and similar postfixes
    n = re.sub(r":\w+$", "", n)
    # Strip dates: -2026-06-08, -20260420
    n = re.sub(r"-\d{4}-\d{2}-\d{2}", "", n)
    n = re.sub(r"-\d{8}", "", n)
    # Strip repeated dates after first pass (e.g. qwen3.5-plus-02-15)
    n = re.sub(r"-\d{2}-\d{2}", "", n)
    # Strip -preview / -latest
    n = re.sub(r"-(preview|latest)$", "", n)
    # Strip quantization / format suffixes: -NVFP4, -BF16, -FP8, -AWQ, -GPTQ, -GGUF
    n = re.sub(r"-(nvfp4|fp16|fp32|bf16|int8|int4|fp8|awq|gptq|gguf)$", "", n)
    # Strip provider prefix (anything before /)
    n = re.sub(r"^[^/]+/", "", n)
    # Normalize dots → underscores (catalog uses underscores: llama-3_1 vs llama-3.1)
    n = n.replace(".", "_")
    return n


# ---------------------------------------------------------------------------
# Lazy-built cross‑provider index
# ---------------------------------------------------------------------------

_cross_index: dict[str, list[tuple[str, str, dict]]] | None = None


def _build_cross_index(catalog: dict) -> dict[str, list[tuple[str, str, dict]]]:
    """Build ``{norm_id: [(provider_id, original_model_id, data), …]}``."""
    idx: dict[str, list[tuple[str, str, dict]]] = {}
    for pid, pdata in catalog.items():
        if not isinstance(pdata, dict):
            continue
        for raw_mid, mdata in pdata.get("models", {}).items():
            norm = _norm_id(raw_mid)
            idx.setdefault(norm, []).append((pid, raw_mid, mdata))
    return idx


# ---------------------------------------------------------------------------
# Look up a model's capabilities in the catalog
# ---------------------------------------------------------------------------
def lookup_model(catalog: dict, provider_id: str, model_id: str) -> dict | None:
    """Look up model capabilities in the catalog.

    Strategy
    1. Exact match in *provider_id*   (fast path, same as before)
    2. Exact match in *any* provider  (also as before)
    3. Normalised match across all providers (strips dates, ``:free``, provider prefix)

    Returns the capability dict, or *None*.
    """
    if not catalog:
        return None

    # ------------------------------------------------------------------
    # 1. Direct provider lookup (exact)
    # ------------------------------------------------------------------
    prov = catalog.get(provider_id)
    if prov and isinstance(prov, dict):
        models = prov.get("models", {})
        if model_id in models:
            return models[model_id]

    # ------------------------------------------------------------------
    # 2. Exact match in any provider
    # ------------------------------------------------------------------
    for pid, p in catalog.items():
        if not isinstance(p, dict):
            continue
        models = p.get("models", {})
        if model_id in models:
            return models[model_id]

    # ------------------------------------------------------------------
    # 3. Normalised match across all providers
    # ------------------------------------------------------------------
    global _cross_index
    if _cross_index is None:
        _cross_index = _build_cross_index(catalog)

    norm = _norm_id(model_id)
    if not norm:
        return None

    candidates = _cross_index.get(norm)
    if not candidates:
        return None

    # Prefer match in the same provider family
    for pid, raw_mid, data in candidates:
        if pid == provider_id:
            return data

    # Otherwise return the first match
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Extract relevant capability fields (filter out extras)
# ---------------------------------------------------------------------------
_CAPABILITY_FIELDS = {
    "family",
    "attachment",
    "reasoning",
    "tool_call",
    "temperature",
    "modalities",
    "open_weights",
    "structured_output",
    "interleaved",
    "limit",
}


def extract_capabilities(raw: dict | None) -> dict | None:
    """Extract only the capability-relevant fields from a catalog entry."""
    if not raw:
        return None
    return {k: v for k, v in raw.items() if k in _CAPABILITY_FIELDS}


# ---------------------------------------------------------------------------
# Check requirements
# ---------------------------------------------------------------------------
def meets_requirements(capabilities: dict | None, requirements: dict) -> bool:
    """Check if model capabilities satisfy all requirements.

    Rules per type:
      - bool:      if required is True, model must be True.
                   if required is False, field is ignored (don't care).
      - int/float: model value must be >= required value.
      - list:      every item in required must be present in model.
      - dict:      recursively check nested keys.
      - str:       prefix match (required "claude" matches "claude-sonnet").

    If capabilities is None or a required key is missing → False.
    """
    if capabilities is None:
        return False

    for key, required_val in requirements.items():
        model_val = capabilities.get(key)

        if model_val is None:
            return False

        if isinstance(required_val, bool):
            # If required is False → don't care (skip)
            if required_val and not model_val:
                return False

        elif isinstance(required_val, (int, float)):
            if not isinstance(model_val, (int, float)):
                return False
            if model_val < required_val:
                return False

        elif isinstance(required_val, list):
            if not isinstance(model_val, list):
                return False
            for item in required_val:
                if item not in model_val:
                    return False

        elif isinstance(required_val, dict):
            if not isinstance(model_val, dict):
                return False
            if not meets_requirements(model_val, required_val):
                return False

        elif isinstance(required_val, str):
            if not isinstance(model_val, str):
                return False
            # Prefix match: required "claude" matches "claude-sonnet"
            if model_val != required_val and not model_val.startswith(required_val + "-"):
                if not model_val.startswith(required_val + "_"):
                    return False

        else:
            # Unknown type — fail closed
            return False

    return True


# ---------------------------------------------------------------------------
# Name-based model filtering: include, exclude, min_params
# ---------------------------------------------------------------------------

def parse_params_from_id(model_id: str) -> int | None:
    """Parse parameter count from model ID.

    Looks for patterns like ``70b``, ``8B``, ``550b-a55b`` (case-insensitive)
    and returns the *largest* integer value found (in billions).
    Returns ``None`` if no parameter pattern is found.

    Examples:
        >>> parse_params_from_id("llama-3.3-70b-versatile")
        70
        >>> parse_params_from_id("nemotron-550b-a55b")
        550
        >>> parse_params_from_id("deepseek-r1:free")
        None
    """
    matches = re.findall(r"(\d+)b", model_id.lower())
    if not matches:
        return None
    return max(int(v) for v in matches)


def _normalize_min_params(value) -> int:
    """Convert min_params config value to an integer (billions).

    Accepts:
      - ``80`` (int)
      - ``"80b"`` (str with suffix)
      - ``"80"`` (str without suffix)
    """
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return int(value.lower().rstrip("b"))
    raise TypeError(f"Unsupported min_params type: {type(value)}")


def model_id_meets_name_requirements(
    model_id: str, requirements: dict, *, provider_id: str | None = None,
) -> bool:
    """Check if a *model_id* passes name-based filters from *requirements*.

    Four checks are performed (all optional).  **exclude** always wins —
    if a model matches both *include* and *exclude*, it is rejected.

    1. **include** (list of str)
       If at least one substring from the list is found in *model_id*
       (case-insensitive), the model passes this check.  If **no** substring
       matches, the model is rejected.  Set to ``[]`` or omit to disable
       (model passes).

    2. **exclude** (list of str | null)
       If any substring from the list is found in *model_id* (case-insensitive),
       the model is rejected — even if it matched *include*.

    3. **min_params** (int | str | null)
       If the model ID contains a parameter pattern (e.g. ``70b``) AND its value
       is strictly less than ``min_params``, the model is rejected.
       Models that have **no** parameter pattern in their ID are **kept**
       (the check is skipped — we don't have enough information).

    4. **providers** (list of str | null)
       If the list is non-empty and *provider_id* is **not** in it,
       the model is rejected.  This is an **exact** match (no substring).
       Omit or set to ``[]`` to disable (model passes).

    Returns ``True`` if the model passes all name-based checks (or none are
    configured).
    """
    model_id_lower = model_id.lower()

    # --- 1. include substrings (require at least one match) ---
    include_list = requirements.get("include")
    if isinstance(include_list, list) and include_list:
        if not any(sub.lower() in model_id_lower for sub in include_list):
            return False

    # --- 2. exclude substrings (reject if any matches) ---
    exclude_list = requirements.get("exclude")
    if isinstance(exclude_list, list) and exclude_list:
        for substr in exclude_list:
            if substr.lower() in model_id_lower:
                return False

    # --- 3. min_params ---
    min_params = requirements.get("min_params")
    if min_params is not None:
        model_params = parse_params_from_id(model_id)
        # If we can't parse params from the name, skip this check
        if model_params is not None:
            if model_params < _normalize_min_params(min_params):
                return False

    # --- 4. providers (exact match on provider_id) ---
    providers_list = requirements.get("providers")
    if isinstance(providers_list, list) and providers_list:
        if provider_id is None or provider_id not in providers_list:
            return False

    return True
