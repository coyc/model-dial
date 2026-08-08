#!/usr/bin/env python3
"""
CLI commands for gateway state management (switch category, rotate credentials).

Called from gateway.sh for all non-process commands (switch_*, rotate, help).
Process management (start/stop/restart) stays in bash.
"""

import json
import sys
from pathlib import Path

from credential_manager import ProviderCredentialManager

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CURRENT_MODEL_PATH = _PROJECT_ROOT / "logs" / "current-model.json"
_PROVIDERS_PATH = _PROJECT_ROOT / "providers.json"
_CONFIG_PATH = _PROJECT_ROOT / "config.json"


def _load_categories() -> list[str]:
    """Load valid category names from config.json → requirements keys."""
    try:
        with open(_CONFIG_PATH) as f:
            config = json.load(f)
        return list(config.get("requirements", {}).keys())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def cmd_switch(category: str) -> None:
    """Advance to next model in category, persist to current-model.json."""
    data = _load_json(_CURRENT_MODEL_PATH)
    cat_data = data.get(category)
    if not cat_data or "models" not in cat_data:
        print(f"Category '{category}' not found in current-model.json", file=sys.stderr)
        sys.exit(1)

    models = cat_data["models"]
    if not models:
        print(f"No models in category '{category}'", file=sys.stderr)
        sys.exit(1)

    # Find current index
    current_idx = None
    for i, m in enumerate(models):
        if m.get("current"):
            current_idx = i
            break

    if current_idx is None:
        current_idx = 0  # first model is current by default

    # Clear current flag on all
    for m in models:
        m.pop("current", None)

    # Advance to next (wrap around)
    next_idx = (current_idx + 1) % len(models)
    models[next_idx]["current"] = True

    _save_json(_CURRENT_MODEL_PATH, data)

    new_model = models[next_idx]
    print(f"Switched {category}: {current_idx + 1} -> {next_idx + 1}")
    print(f"  {new_model['provider']} / {new_model['model_id']}")


def cmd_reset() -> None:
    """Remove current-model.json to reset state and start fresh."""
    if _CURRENT_MODEL_PATH.exists():
        _CURRENT_MODEL_PATH.unlink()
        print(f"Removed {_CURRENT_MODEL_PATH.name}")
    else:
        print("No saved state to remove")


def cmd_rotate(provider: str) -> None:
    """Rotate to next credential for provider, persist to providers.json."""
    if not _PROVIDERS_PATH.exists():
        print(f"Error: {_PROVIDERS_PATH} not found", file=sys.stderr)
        sys.exit(1)

    try:
        mgr = ProviderCredentialManager(_PROVIDERS_PATH)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: invalid JSON in {_PROVIDERS_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    if provider not in mgr.providers_creds:
        providers = ", ".join(mgr.providers_creds.keys())
        print(f"Provider '{provider}' not found in providers.json", file=sys.stderr)
        print(f"Available: {providers}", file=sys.stderr)
        sys.exit(1)

    creds_list = mgr._get_creds(provider)
    if not creds_list:
        print(f"No credentials for provider '{provider}'", file=sys.stderr)
        sys.exit(1)

    old_idx = mgr.credential_index.get(provider, 0)
    new_cred = mgr.advance_credential(provider)
    new_idx = mgr.credential_index.get(provider, 0)

    print(f"Rotated {provider}: credential {old_idx + 1} -> {new_idx + 1} (of {len(creds_list)})")


def cmd_state() -> None:
    """Show current gateway state: PID, current models, credential positions."""
    # PID
    pid_file = _PROJECT_ROOT / "logs" / "gateway.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        if Path(f"/proc/{pid}").exists():
            print(f"PID:   {pid} (running)")
        else:
            print(f"PID:   {pid} (stale)")
    else:
        print("PID:   not running")

    # Current models per category
    if _CURRENT_MODEL_PATH.exists():
        data = _load_json(_CURRENT_MODEL_PATH)
        categories = _load_categories() or list(data.keys())
        for cat in categories:
            cat_data = data.get(cat)
            if not cat_data or "models" not in cat_data:
                print(f"  {cat}:  —")
                continue
            models = cat_data["models"]
            total = len(models)
            current_idx = None
            for i, m in enumerate(models):
                if m.get("current"):
                    current_idx = i
                    break
            if current_idx is None and total:
                current_idx = 0
            if current_idx is not None:
                m = models[current_idx]
                print(f"  {cat}:  {m['provider']} / {m['model_id']}  ({current_idx + 1}/{total})")
            else:
                print(f"  {cat}:  —")
    else:
        print("Models: —")

    # Credential positions
    if _PROVIDERS_PATH.exists():
        mgr = ProviderCredentialManager(_PROVIDERS_PATH)
        print("Creds:")
        for prov_name, prov_data in mgr.providers_creds.items():
            creds = prov_data.get("credentials", [])
            total = len(creds)
            current_idx = mgr.credential_index.get(prov_name, 0)
            print(f"  {prov_name}: ({current_idx + 1}/{total})")
    else:
        print("Creds: —")


def cmd_help() -> None:
    cats = _load_categories()
    cat_list = "|".join(cats) if cats else "<category>"
    print(f"""Usage: gateway.sh <command> [args]

Commands:
  start                  Start gateway
  stop                   Stop gateway
  restart                Restart gateway
  state [-n N]           Show current state + last N log lines (default: 40)
  switch <category>      Switch to next model in category ({cat_list}) and restart
  reset                  Remove current-model.json and restart gateway
  rotate <provider>      Rotate to next credential for provider and restart
  help                   Show this help""")


def main() -> None:
    if len(sys.argv) < 2:
        cmd_help()
        sys.exit(1)

    command = sys.argv[1]

    if command == "switch":
        if len(sys.argv) < 3:
            cats = _load_categories()
            cat_list = "|".join(cats) if cats else "<category>"
            print(f"Usage: gateway.sh switch <{cat_list}>", file=sys.stderr)
            sys.exit(1)
        category = sys.argv[2]
        cats = _load_categories()
        if cats and category not in cats:
            print(f"Invalid category: {category}", file=sys.stderr)
            print(f"Valid categories: {', '.join(cats)}", file=sys.stderr)
            sys.exit(1)
        cmd_switch(category)
    elif command == "reset":
        cmd_reset()
    elif command == "rotate":
        if len(sys.argv) < 3:
            print("Usage: gateway.sh rotate <provider>", file=sys.stderr)
            sys.exit(1)
        cmd_rotate(sys.argv[2])
    elif command == "state":
        cmd_state()
    elif command == "help":
        cmd_help()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        cmd_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
