#!/usr/bin/env python3
"""
Shared credential rotation logic for providers.json.

Used by both gateway and test-runner to avoid duplicating the same
rotation algorithm, index tracking, and persistence logic.
"""

import json
from pathlib import Path


class ProviderCredentialManager:
    """Tracks and rotates provider credentials from providers.json.

    Loads all credentials, tracks the current index per provider,
    and persists rotation state.  The ``current`` flag in providers.json
    is always kept in sync with the active credential.

    If no credential is marked ``current`` for a provider, the first one
    is automatically marked and the change is persisted.
    """

    def __init__(self, providers_path: Path):
        self._providers_path = providers_path
        self._providers_creds: dict = {}
        self._credential_index: dict[str, int] = {}
        if providers_path.exists():
            self._providers_creds = json.loads(providers_path.read_text())
            dirty = False
            for prov_name, prov_data in self._providers_creds.items():
                creds = prov_data.get("credentials", [])
                if not creds:
                    continue
                found_current = False
                for i, cred in enumerate(creds):
                    if cred.get("current"):
                        self._credential_index[prov_name] = i
                        found_current = True
                        break
                if not found_current:
                    self._credential_index[prov_name] = 0
                    creds[0]["current"] = True
                    dirty = True
            if dirty:
                self._persist()

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def providers_creds(self) -> dict:
        """Full providers config dict (provider_name → {type, credentials, ...})."""
        return self._providers_creds

    @property
    def credential_index(self) -> dict[str, int]:
        """Current credential index per provider."""
        return self._credential_index

    def _get_creds(self, provider_id: str) -> list[dict]:
        return self._providers_creds.get(provider_id, {}).get("credentials", [])

    def get_credential(self, provider_id: str) -> dict | None:
        """Return the current credential dict (base_url, api_key, ...) or None."""
        creds = self._get_creds(provider_id)
        if not creds:
            return None
        idx = self._credential_index.get(provider_id, 0)
        if idx >= len(creds):
            idx = 0
            self._credential_index[provider_id] = 0
        return creds[idx]

    def credential_count(self, provider_id: str) -> int:
        """Return the number of credentials configured for this provider."""
        return len(self._get_creds(provider_id))

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def advance_credential(self, provider_id: str) -> dict | None:
        """Rotate to next credential, persist to file, return new credential dict.

        Returns None if the provider has no credentials.
        """
        creds = self._get_creds(provider_id)
        if not creds:
            return None
        idx = self._credential_index.get(provider_id, 0)
        idx = (idx + 1) % len(creds)
        self._credential_index[provider_id] = idx
        self._save_credential_state(provider_id, idx)
        return creds[idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_credential_state(self, provider_name: str, credential_idx: int) -> None:
        """Update the ``current`` flag for a provider and persist to disk."""
        prov_data = self._providers_creds.get(provider_name)
        if not prov_data:
            return

        creds = prov_data.get("credentials", [])
        for i, cred in enumerate(creds):
            if i == credential_idx:
                cred["current"] = True
            else:
                cred.pop("current", None)

        self._persist()

    def _persist(self) -> None:
        """Write the full providers data back to providers.json."""
        path = self._providers_path
        if not path.exists():
            return
        try:
            with open(path, "w") as f:
                # 4-space indent to match docker/providers.example.json
                json.dump(self._providers_creds, f, indent=4)
                f.write("\n")
        except OSError:
            return
