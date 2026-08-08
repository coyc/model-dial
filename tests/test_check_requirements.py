"""Unit tests for check-requirements.py — model capabilities lookup and requirements check.

Reads result-fetch.json format (flat provider-based model list), looks up
each model in the catalog, and marks rejected / requirements_breakdown.
"""

import importlib.util
import json
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

# ---------------------------------------------------------------------------
# Load the module under test
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "check_requirements", _SCRIPT_DIR / "../src/check-requirements.py"
)
if _SPEC is None:
    raise FileNotFoundError("Could not load check-requirements.py")

cr = importlib.util.module_from_spec(_SPEC)
if _SPEC.loader is None:
    raise ImportError("Module loader is None")

# Re-exec with mocked config to control REQUIREMENTS at import time
_FAKE_CONFIG = {
    "models_catalog_url": "https://models.dev/api.json",
    "requirements": {
        "simple": {
            "models_catalog": {
                "tool_call": True,
                "limit": {"context": 135000},
            },
            "model_id": {},
        },
        "coder": {
            "models_catalog": {
                "tool_call": True,
                "reasoning": True,
                "temperature": True,
                "limit": {"context": 262144, "output": 32768},
            },
            "model_id": {},
        },
        "visual": {
            "models_catalog": {
                "tool_call": True,
                "attachment": True,
                "reasoning": True,
                "temperature": True,
                "modalities": {"input": ["text", "image"]},
                "limit": {"context": 131072, "output": 32768},
            },
            "model_id": {},
        },
    },
}

with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(_FAKE_CONFIG))):
    _SPEC.loader.exec_module(cr)


# ---------------------------------------------------------------------------
# Sample data — fetch format
# ---------------------------------------------------------------------------
SAMPLE_FETCH_INPUT = {
    "providers": [
        {
            "provider": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "models": [
                {"id": "llama-3.3-70b-versatile"},
                {"id": "error-model"},
            ],
        },
        {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "models": [
                {"id": "deepseek-r1:free"},
            ],
        },
        {
            "provider": "nvidia",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "models": [
                {"id": "mistral-7b"},
            ],
        },
    ],
}

SAMPLE_CATALOG = {
    "groq": {
        "models": {
            "llama-3.3-70b-versatile": {
                "family": "llama",
                "tool_call": True,
                "temperature": True,
                "limit": {"context": 131072, "output": 16384},
                "modalities": {"input": ["text"]},
                "reasoning": False,
                "extra_field": "ignored",
            }
        }
    },
    "openrouter": {
        "models": {
            "deepseek-r1": {
                "family": "deepseek-r1",
                "tool_call": True,
                "temperature": True,
                "limit": {"context": 131072, "output": 8192},
                "modalities": {"input": ["text"]},
                "reasoning": True,
            },
            "deepseek-r1:free": {
                "family": "deepseek-r1",
                "tool_call": True,
                "temperature": True,
                "limit": {"context": 131072, "output": 8192},
                "modalities": {"input": ["text"]},
                "reasoning": True,
            },
        }
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMainFlow(unittest.TestCase):
    def _run_main(self, input_data=None, catalog=None, deny_list=None):
        """Run main() with mocked file I/O."""
        if input_data is None:
            input_data = SAMPLE_FETCH_INPUT
        if catalog is None:
            catalog = SAMPLE_CATALOG
        if deny_list is None:
            deny_list = []

        fake_input = mock.mock_open(read_data=json.dumps(input_data))
        fake_stat = mock.MagicMock()
        fake_stat.st_size = 1000

        with mock.patch.object(cr, "CATALOG_URL", "https://models.dev/api.json"), \
             mock.patch.object(cr.capabilities, "load_catalog", return_value=catalog), \
             mock.patch.object(cr.Path, "exists", return_value=True), \
             mock.patch.object(cr.Path, "stat", return_value=fake_stat), \
             mock.patch.object(cr.sys, "stderr"), \
             mock.patch.object(cr.sys, "stdout", new_callable=StringIO) as out, \
             mock.patch("builtins.open", fake_input):
            cr.main(input_path=Path("/fake/result-fetch.json"), deny_list=deny_list)
            return json.loads(out.getvalue())

    def test_returns_output_structure(self):
        """Output has expected top-level keys."""
        out = self._run_main()
        self.assertIn("input_file", out)
        self.assertIn("total_models", out)
        self.assertIn("stat_rejected", out)
        self.assertNotIn("stat_meets_requirements", out)
        self.assertIn("stat_eligible", out)
        self.assertEqual(out["stat_eligible"], {"simple": 0, "coder": 0, "visual": 0})
        self.assertIn("results", out)

    def test_total_models_counts_all_input(self):
        """total_models counts every model from fetch output."""
        out = self._run_main()
        self.assertEqual(out["total_models"], 4)

    def test_model_with_catalog_has_capabilities(self):
        """Model found in catalog gets capabilities dict."""
        out = self._run_main()
        groq_model = [r for r in out["results"] if r["provider"] == "groq" and r["model_id"] == "llama-3.3-70b-versatile"][0]
        self.assertIsNotNone(groq_model["capabilities"])
        self.assertEqual(groq_model["capabilities"]["family"], "llama")
        self.assertIn("tool_call", groq_model["capabilities"])
        self.assertIn("limit", groq_model["capabilities"])

    def test_model_with_catalog_has_breakdown(self):
        """Model found in catalog gets per-group requirements_breakdown."""
        out = self._run_main()
        groq_model = [r for r in out["results"] if r["provider"] == "groq" and r["model_id"] == "llama-3.3-70b-versatile"][0]
        self.assertIn("simple", groq_model["requirements_breakdown"])
        self.assertIn("coder", groq_model["requirements_breakdown"])
        self.assertIn("visual", groq_model["requirements_breakdown"])

    def test_model_with_catalog_not_rejected(self):
        """Model found in catalog is not rejected."""
        out = self._run_main()
        models = [r for r in out["results"] if r["rejected"]]
        # Only error-model and mistral-7b should be rejected (not in catalog)
        self.assertEqual(len(models), 2)

        # Non-rejected models have reject_reason: None
        for r in out["results"]:
            if not r["rejected"]:
                self.assertIsNone(r.get("reject_reason"))

    def test_model_without_catalog_is_rejected(self):
        """Model NOT found in catalog is rejected with no capabilities."""
        out = self._run_main()
        rejected = [r for r in out["results"] if r["rejected"]]
        for r in rejected:
            self.assertIsNone(r["capabilities"])
            self.assertFalse(any(r.get("requirements_breakdown", {}).values()))
            self.assertEqual(r["requirements_breakdown"], {})
            self.assertEqual(r.get("reject_reason"), "no catalog data")

    def test_requirements_breakdown_reflects_capabilities(self):
        """requirements_breakdown shows which categories a model qualifies for."""
        out = self._run_main()
        groq_model = [r for r in out["results"] if r["provider"] == "groq" and r["model_id"] == "llama-3.3-70b-versatile"][0]
        # llama-3.3-70b-versatile: tool_call=T, temperature=T, limit.context=131072 < 135000
        self.assertFalse(groq_model["requirements_breakdown"]["simple"])
        # coder: needs reasoning=True, but model has reasoning=False
        self.assertFalse(groq_model["requirements_breakdown"]["coder"])

    def test_all_breakdown_false_when_no_group_passes(self):
        """Model that fails all group requirements has all breakdown entries false."""
        catalog = {
            "minimal": {
                "models": {
                    "minimal-v1": {
                        "tool_call": True,
                    }
                }
            }
        }
        data = {
            "providers": [
                {
                    "provider": "minimal",
                    "base_url": "https://x/v1",
                    "models": [{"id": "minimal-v1"}],
                },
            ]
        }
        out = self._run_main(input_data=data, catalog=catalog)
        minimal = out["results"][0]
        # tool_call=True alone isn't enough for any group
        for group in ("simple", "coder", "visual"):
            self.assertFalse(minimal["requirements_breakdown"][group])

    def test_stat_rejected_remains_after_removing_meets_requirements(self):
        """stat_rejected is still present after removing meets_requirements."""
        out = self._run_main()
        self.assertIn("stat_rejected", out)
        self.assertNotIn("stat_meets_requirements", out)

    def test_stat_rejected_counts_correctly(self):
        """stat_rejected counts models without catalog data."""
        out = self._run_main()
        # error-model (groq), mistral-7b (nvidia) — not in catalog
        self.assertEqual(out["stat_rejected"], 2)

    def test_empty_input_returns_none(self):
        """Empty or missing input returns None."""
        with mock.patch.object(cr.Path, "exists", return_value=False), \
             mock.patch.object(cr.sys, "stderr"):
            result = cr.main(input_path=Path("/nonexistent.json"))
            self.assertIsNone(result)

    def test_no_catalog_returns_none(self):
        """When catalog is unavailable, main returns None."""
        fake_stat = mock.MagicMock()
        fake_stat.st_size = 1000
        fake_input = mock.mock_open(read_data=json.dumps({"providers": []}))
        with mock.patch.object(cr.capabilities, "load_catalog", return_value=None), \
             mock.patch.object(cr.Path, "exists", return_value=True), \
             mock.patch.object(cr.Path, "stat", return_value=fake_stat), \
             mock.patch("builtins.open", fake_input), \
             mock.patch.object(cr.sys, "stderr"):
            result = cr.main(input_path=Path("/fake/result-fetch.json"))
            self.assertIsNone(result)

    def test_no_models_returns_none(self):
        """When input has no models, main returns None."""
        data = {"providers": []}
        fake_input = mock.mock_open(read_data=json.dumps(data))
        fake_stat = mock.MagicMock()
        fake_stat.st_size = 1000
        with mock.patch.object(cr.capabilities, "load_catalog", return_value=SAMPLE_CATALOG), \
             mock.patch.object(cr.Path, "exists", return_value=True), \
             mock.patch.object(cr.Path, "stat", return_value=fake_stat), \
             mock.patch.object(cr.sys, "stderr"), \
             mock.patch("builtins.open", fake_input):
            result = cr.main(input_path=Path("/fake/result-fetch.json"))
            self.assertIsNone(result)

    def test_extra_catalog_fields_removed(self):
        """Extra fields from catalog (not in _CAPABILITY_FIELDS) are filtered out."""
        out = self._run_main()
        groq_model = [r for r in out["results"] if r["provider"] == "groq" and r["model_id"] == "llama-3.3-70b-versatile"][0]
        self.assertNotIn("extra_field", groq_model["capabilities"])

    # ------------------------------------------------------------------
    # Name-based filtering: exclude substrings
    # ------------------------------------------------------------------

    def _make_minimal_catalog(self, model_id: str, caps: dict | None = None) -> tuple[dict, dict]:
        """Helper: build a single-model catalog + input data pair."""
        if caps is None:
            caps = {"tool_call": True, "limit": {"context": 131072}, "modalities": {"input": ["text"]}}
        catalog = {"test_provider": {"models": {model_id: caps}}}
        data = {"providers": [{"provider": "test_provider", "base_url": "https://x/v1", "models": [{"id": model_id}]}]}
        return catalog, data

    def _run_with_reqs(self, input_data: dict, catalog: dict, reqs: dict) -> dict:
        """Run main() with custom REQUIREMENTS."""
        with mock.patch.object(cr, "REQUIREMENTS", reqs):
            return self._run_main(input_data=input_data, catalog=catalog)

    def test_exclude_substring_removes_from_group(self):
        """Model whose name contains an excluded substring is rejected from that group."""
        catalog, data = self._make_minimal_catalog("llama-nano-1b")
        reqs = {"simple": {"models_catalog": {"tool_call": True, "limit": {"context": 131072}}, "model_id": {"exclude": ["nano", "micro"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_exclude_does_not_affect_other_groups(self):
        """Exclude substrings only affect their own group's breakdown entry."""
        catalog, data = self._make_minimal_catalog("safe-model")
        reqs = {
            "simple": {"models_catalog": {"tool_call": True, "limit": {"context": 131072}}, "model_id": {"exclude": ["nano"]}},
            "coder": {"models_catalog": {"tool_call": True, "reasoning": True}, "model_id": {"exclude": ["nano"]}},
        }
        # safe-model doesn't match "nano", so coder fails only on reasoning=False
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])
        self.assertFalse(out["results"][0]["requirements_breakdown"]["coder"])

    def test_exclude_case_insensitive(self):
        """Exclude substring matching is case-insensitive."""
        catalog, data = self._make_minimal_catalog("FLASH-test")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"exclude": ["flash"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_exclude_empty_list_does_nothing(self):
        """Empty exclude list does not affect results."""
        catalog, data = self._make_minimal_catalog("model-nano")
        reqs = {"simple": {"models_catalog": {"tool_call": True, "limit": {"context": 131072}}, "model_id": {"exclude": []}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_exclude_no_match_keeps_model(self):
        """Model whose name does NOT match any exclude substring is kept."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"exclude": ["nano", "micro", "flash"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_exclude_multiple_matches(self):
        """Model with multiple matching substrings still fails once."""
        catalog, data = self._make_minimal_catalog("tiny-micro-nano")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"exclude": ["nano", "micro"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    # ------------------------------------------------------------------
    # Name-based filtering: min_params
    # ------------------------------------------------------------------

    def test_min_params_rejects_small_model(self):
        """Model with fewer params than min_params is rejected."""
        catalog, data = self._make_minimal_catalog("llama-3.2-3b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"min_params": "8b"}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_min_params_keeps_large_model(self):
        """Model with params >= min_params is kept."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"min_params": 8}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_min_params_skip_when_no_params_in_name(self):
        """Model without param count in name passes min_params check."""
        catalog, data = self._make_minimal_catalog("deepseek-r1:free")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"min_params": "8b"}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_min_params_picks_max_match(self):
        """When model_id has multiple param patterns, max is used."""
        catalog, data = self._make_minimal_catalog("nemotron-550b-a55b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"min_params": "80b"}}}
        out = self._run_with_reqs(data, catalog, reqs)
        # 550 >= 80 → keep
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_min_params_none_is_ignored(self):
        """min_params: null means no check."""
        catalog, data = self._make_minimal_catalog("llama-3.2-3b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"min_params": None}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_min_params_string_number_without_b(self):
        """min_params can be specified as plain number string like '8'."""
        catalog, data = self._make_minimal_catalog("llama-3.2-3b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"min_params": "8"}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_name_and_capability_both_needed(self):
        """A model passing name filters but failing capabilities is still rejected."""
        catalog, data = self._make_minimal_catalog("big-model-70b")
        # No tool_call in caps, but min_params passes
        catalog["test_provider"]["models"]["big-model-70b"] = {"limit": {"context": 131072}}
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"min_params": "8b"}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    # ------------------------------------------------------------------
    # Name-based filtering: include substring array
    # ------------------------------------------------------------------

    def test_include_keeps_matching_model(self):
        """Model whose ID contains a required substring passes."""
        catalog, data = self._make_minimal_catalog("qwen3-vision-max")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_rejects_non_matching_model(self):
        """Model whose ID does NOT contain any required substring is rejected."""
        catalog, data = self._make_minimal_catalog("qwen3-text-max")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_matches_any_of_array(self):
        """Model matches if ANY substring from include list is found."""
        catalog, data = self._make_minimal_catalog("qwen3-pro-max")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision", "pro"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_case_insensitive(self):
        """include substring matching is case-insensitive."""
        catalog, data = self._make_minimal_catalog("QWEN3-VISION-MAX")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_case_insensitive_reject(self):
        """include rejection is also case-insensitive."""
        catalog, data = self._make_minimal_catalog("QWEN3-TEXT-MAX")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_null_is_ignored(self):
        """include: null means no check."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": None}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_empty_list_is_ignored(self):
        """include: [] (empty list) means no check."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": []}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_absent_is_ignored(self):
        """Omitting include entirely means no check."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"min_params": "8b"}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_with_exclude_combined(self):
        """include passes but exclude rejects — exclude wins."""
        catalog, data = self._make_minimal_catalog("qwen3-vision-nano")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"], "exclude": ["nano"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        # include passes ("vision" found), but exclude rejects ("nano" found)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_with_min_params_combined(self):
        """include passes but min_params rejects."""
        catalog, data = self._make_minimal_catalog("qwen3-vision-3b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"], "min_params": "8b"}}}
        out = self._run_with_reqs(data, catalog, reqs)
        # include passes, but min_params rejects (3 < 8)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_only_affects_own_group(self):
        """include only affects the group it's configured in."""
        catalog, data = self._make_minimal_catalog("qwen3-text-max")
        reqs = {
            "simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"]}},
            "coder": {"models_catalog": {"tool_call": True}, "model_id": {}},
        }
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])
        self.assertTrue(out["results"][0]["requirements_breakdown"]["coder"])

    def test_include_substring_at_start(self):
        """include matches substring at the beginning of model_id."""
        catalog, data = self._make_minimal_catalog("vision-qwen3-max")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_include_substring_at_end(self):
        """include matches substring at the end of model_id."""
        catalog, data = self._make_minimal_catalog("qwen3-max-vision")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    # ------------------------------------------------------------------
    # Name-based filtering: providers (exact match)
    # ------------------------------------------------------------------

    def _make_multi_provider_catalog(self, providers: list[str], model_id: str) -> tuple[dict, dict]:
        """Helper: build a catalog with models on multiple providers."""
        catalog = {}
        models_list = []
        for pid in providers:
            catalog[pid] = {"models": {model_id: {"tool_call": True, "limit": {"context": 131072}, "modalities": {"input": ["text"]}}}}
            models_list.append({"id": model_id})
        # Use the first provider for the input data (all providers see the same model)
        data = {"providers": [{"provider": providers[0], "base_url": "https://x/v1", "models": [{"id": model_id}]}]}
        return catalog, data

    def test_providers_keeps_matching_provider(self):
        """Model from an allowed provider passes the providers check."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"providers": ["test_provider"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_rejects_non_matching_provider(self):
        """Model from a non-allowed provider is rejected."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"providers": ["opencode", "google"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_exact_match(self):
        """providers requires exact match — substring does not count."""
        catalog, data = self._make_minimal_catalog("model-v1")
        # "test" is a substring of "test_provider", but not an exact match
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"providers": ["test"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_multiple_providers(self):
        """Model matches if its provider is any of the listed providers."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"providers": ["other", "test_provider", "another"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_empty_list_is_ignored(self):
        """Empty providers list means no check."""
        catalog, data = self._make_minimal_catalog("model-v1")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"providers": []}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_null_is_ignored(self):
        """providers: null means no check."""
        catalog, data = self._make_minimal_catalog("model-v1")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"providers": None}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_absent_is_ignored(self):
        """Omitting providers entirely means no check."""
        catalog, data = self._make_minimal_catalog("model-v1")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"min_params": "8b"}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_combined_with_include(self):
        """Both providers and include must pass for the model to be eligible."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        # include passes ("llama" is in model_id), providers passes ("test_provider" matches)
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["llama"], "providers": ["test_provider"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_combined_with_include_rejects(self):
        """Model passes providers but fails include → rejected."""
        catalog, data = self._make_minimal_catalog("qwen3-text-max")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["vision"], "providers": ["test_provider"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        # include fails ("vision" not in "qwen3-text-max")
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_combined_with_exclude(self):
        """Model passes providers and include, but exclude wins → rejected."""
        catalog, data = self._make_minimal_catalog("llama-nano-1b")
        reqs = {"simple": {"models_catalog": {"tool_call": True}, "model_id": {"include": ["llama"], "exclude": ["nano"], "providers": ["test_provider"]}}}
        out = self._run_with_reqs(data, catalog, reqs)
        # providers passes, include passes ("llama"), but exclude rejects ("nano")
        self.assertFalse(out["results"][0]["requirements_breakdown"]["simple"])

    def test_providers_only_affects_own_group(self):
        """providers filter only affects the group it's configured in."""
        catalog, data = self._make_minimal_catalog("llama-3.3-70b")
        reqs = {
            "simple": {"models_catalog": {"tool_call": True}, "model_id": {"providers": ["test_provider"]}},
            "coder": {"models_catalog": {"tool_call": True}, "model_id": {"providers": ["opencode"]}},
        }
        out = self._run_with_reqs(data, catalog, reqs)
        self.assertTrue(out["results"][0]["requirements_breakdown"]["simple"])
        self.assertFalse(out["results"][0]["requirements_breakdown"]["coder"])

    # ------------------------------------------------------------------
    # Deny list
    # ------------------------------------------------------------------

    def test_deny_by_model_id_globally(self):
        """Deny entry without provider blocks model_id from any provider."""
        deny = [{"model_id": "llama-3.3-70b-versatile"}]
        out = self._run_main(deny_list=deny)
        model = next(r for r in out["results"] if r["model_id"] == "llama-3.3-70b-versatile")
        self.assertTrue(model["rejected"])
        self.assertEqual(model["reject_reason"], "denied")

    def test_deny_by_model_and_provider(self):
        """Deny entry with specific provider blocks only that provider."""
        deny = [{"model_id": "deepseek-r1:free", "provider": "openrouter"}]
        out = self._run_main(deny_list=deny)
        denied = next(r for r in out["results"] if r["model_id"] == "deepseek-r1:free")
        self.assertTrue(denied["rejected"])
        self.assertEqual(denied["reject_reason"], "denied")

    def test_deny_does_not_affect_other_providers(self):
        """Deny entry with a specific provider leaves the same model on other providers untouched."""
        deny = [{"model_id": "llama-3.3-70b-versatile", "provider": "nvidia"}]
        out = self._run_main(deny_list=deny)
        # groq/llama-3.3-70b-versatile should NOT be denied (nvidia doesn't have it)
        model = next(r for r in out["results"] if r["model_id"] == "llama-3.3-70b-versatile")
        self.assertFalse(model["rejected"])
        self.assertIsNone(model["reject_reason"])

    def test_deny_empty_list_does_nothing(self):
        """Empty deny list does not affect results."""
        out = self._run_main(deny_list=[])
        rejected = [r for r in out["results"] if r["rejected"]]
        # Only the 2 no-catalog models
        self.assertEqual(len(rejected), 2)
        for r in rejected:
            self.assertEqual(r["reject_reason"], "no catalog data")

    def test_deny_non_matching_model_untouched(self):
        """Model not in deny list is not affected."""
        deny = [{"model_id": "some-other-model"}]
        out = self._run_main(deny_list=deny)
        model = next(r for r in out["results"] if r["model_id"] == "llama-3.3-70b-versatile")
        self.assertFalse(model["rejected"])
        self.assertIsNone(model["reject_reason"])

    def test_denied_model_has_capabilities(self):
        """Denied model retains capabilities in the output (for debugging)."""
        deny = [{"model_id": "llama-3.3-70b-versatile"}]
        out = self._run_main(deny_list=deny)
        model = next(r for r in out["results"] if r["model_id"] == "llama-3.3-70b-versatile")
        self.assertIsNotNone(model["capabilities"])
        self.assertEqual(model["capabilities"]["family"], "llama")

    def test_denied_model_has_empty_breakdown(self):
        """Denied model has empty requirements_breakdown (no per-group check needed)."""
        deny = [{"model_id": "llama-3.3-70b-versatile"}]
        out = self._run_main(deny_list=deny)
        model = next(r for r in out["results"] if r["model_id"] == "llama-3.3-70b-versatile")
        self.assertEqual(model["requirements_breakdown"], {})

    def test_deny_no_catalog_takes_precedence(self):
        """Model without catalog gets 'no catalog data' even if also in deny list."""
        deny = [{"model_id": "error-model"}]
        out = self._run_main(deny_list=deny)
        model = next(r for r in out["results"] if r["model_id"] == "error-model")
        self.assertTrue(model["rejected"])
        self.assertEqual(model["reject_reason"], "no catalog data")

    def test_deny_does_not_count_as_eligible(self):
        """Denied model does NOT increment stat_eligible counts."""
        # Without deny: deepseek-r1:free passes simple if ctx requirement is lowered
        # Use a custom setup where model WOULD be eligible but for deny
        catalog = {
            "test_provider": {
                "models": {
                    "good-model": {
                        "tool_call": True,
                        "temperature": True,
                        "limit": {"context": 200000},
                        "modalities": {"input": ["text"]},
                    }
                }
            }
        }
        data = {
            "providers": [
                {
                    "provider": "test_provider",
                    "base_url": "https://x/v1",
                    "models": [{"id": "good-model"}],
                },
            ]
        }
        reqs = {"simple": {"models_catalog": {"tool_call": True, "limit": {"context": 131072}}, "model_id": {}}}

        deny = [{"model_id": "good-model"}]
        with mock.patch.object(cr, "REQUIREMENTS", reqs):
            out = self._run_main(input_data=data, catalog=catalog, deny_list=deny)
        self.assertEqual(out["stat_eligible"]["simple"], 0)
        # The model is in results but denied
        model = out["results"][0]
        self.assertTrue(model["rejected"])
        self.assertEqual(model["reject_reason"], "denied")


if __name__ == "__main__":
    unittest.main()
