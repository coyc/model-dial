#!/usr/bin/env python3
"""
Check model capabilities against requirements.

Reads result-fetch.json (flat model list), loads the model catalog,
looks up each model's capabilities in the catalog, and outputs
result-requirements.json with capabilities data and per-group requirements
check results.

Models without catalog data are marked as rejected (unreliable).
Only models that pass at least one category's requirements will be tested later.

Pipeline step: after fetch, before test.

Usage:
    python3 check-requirements.py
    python3 check-requirements.py --input result-fetch.json
"""

import argparse
import json
import sys
from pathlib import Path

# Import capabilities module
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capabilities

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT = _PROJECT_ROOT / "result-fetch.json"
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.json"

with open(_DEFAULT_CONFIG) as _f:
    _CFG = json.load(_f)
REQUIREMENTS = _CFG.get("requirements", {})
CATALOG_URL = _CFG.get("models_catalog_url")


# ---------------------------------------------------------------------------
# Deny list helpers
# ---------------------------------------------------------------------------
def _load_deny_list(project_root: Path) -> list[dict]:
    """Load deny list from *deny.json* in project root.

    Returns a list of ``{"model_id": …, "provider": … | None}`` entries,
    or an empty list if the file does not exist or is invalid.
    """
    deny_path = project_root / "deny.json"
    try:
        with open(deny_path) as f:
            return json.load(f).get("deny", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _is_denied(provider: str, model_id: str, deny_list: list[dict]) -> bool:
    """Check if a *(provider, model_id)* pair matches any deny entry.

    A deny entry matches when:
    - ``model_id`` matches **and** provider matches; OR
    - ``model_id`` matches **and** provider in the entry is ``None`` (wildcard).
    """
    for entry in deny_list:
        if entry.get("model_id") != model_id:
            continue
        entry_provider = entry.get("provider")
        if entry_provider is None or entry_provider == provider:
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(input_path: Path | None = None, output_file=None, deny_list: list[dict] | None = None):
    """Run check-requirements pipeline. Returns parsed JSON result."""
    src = input_path or _DEFAULT_INPUT

    if not src.exists() or src.stat().st_size == 0:
        print(f"[error] Input file not found or empty: {src}", file=sys.stderr)
        return None

    with open(src) as f:
        data = json.load(f)

    # Load deny list from project root if not provided
    if deny_list is None:
        deny_list = _load_deny_list(_PROJECT_ROOT)

    # Flatten the provider-based model list from fetch output
    providers_list = data.get("providers", [])
    raw_models = []
    for prov in providers_list:
        pid = prov.get("provider", "")
        for m in prov.get("models", []):
            raw_models.append({
                "provider": pid,
                "model_id": m["id"],
            })

    if not raw_models:
        print("[error] No models found in input", file=sys.stderr)
        return None

    # Load catalog
    catalog = capabilities.load_catalog(CATALOG_URL)
    if catalog is None:
        print("[error] No model catalog available", file=sys.stderr)
        return None

    output_results = []
    rejected_count = 0
    eligible_count = 0
    eligible_per_group: dict[str, int] = {g: 0 for g in REQUIREMENTS}

    for r in raw_models:
        provider = r["provider"]
        model_id = r["model_id"]

        # Look up in catalog
        raw = capabilities.lookup_model(catalog, provider, model_id)
        caps = capabilities.extract_capabilities(raw)

        if caps is None:
            # No catalog data — unreliable model, reject
            output_results.append({
                "provider": provider,
                "model_id": model_id,
                "capabilities": None,
                "requirements_breakdown": {},
                "rejected": True,
                "reject_reason": "no catalog data",
            })
            rejected_count += 1
            continue

        # Check deny list (early exit — no need to compute breakdown)
        if _is_denied(provider, model_id, deny_list):
            output_results.append({
                "provider": provider,
                "model_id": model_id,
                "capabilities": caps,
                "requirements_breakdown": {},
                "rejected": True,
                "reject_reason": "denied",
            })
            rejected_count += 1
            continue

        # Check requirements per group
        breakdown = {}
        for group, reqs in REQUIREMENTS.items():
            # Separate capability-based checks from name-based (exclude, min_params)
            cap_reqs = reqs.get("models_catalog", {})
            name_reqs = reqs.get("model_id", {})
            meets_cap = capabilities.meets_requirements(caps, cap_reqs)
            meets_name = capabilities.model_id_meets_name_requirements(model_id, name_reqs, provider_id=provider)
            eligible = meets_cap and meets_name
            breakdown[group] = eligible
            if eligible:
                eligible_per_group[group] += 1

        if any(breakdown.values()):
            eligible_count += 1

        output_results.append({
            "provider": provider,
            "model_id": model_id,
            "capabilities": caps,
            "requirements_breakdown": breakdown,
            "rejected": False,
            "reject_reason": None,
        })

    print(
        f"[requirements] Checked {len(raw_models)} models: "
        f"{rejected_count} rejected (no catalog data), "
        f"{eligible_count} eligible for testing",
        file=sys.stderr,
    )

    output = {
        "input_file": str(src),
        "total_models": len(raw_models),
        "stat_rejected": rejected_count,
        "stat_eligible": eligible_per_group,
        "results": output_results,
    }

    out = output_file or sys.stdout
    print(json.dumps(output, indent=2, ensure_ascii=False), file=out)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check model capabilities against requirements"
    )
    parser.add_argument("--input", type=Path, help="Path to result-fetch.json")
    args = parser.parse_args()
    main(input_path=args.input)
