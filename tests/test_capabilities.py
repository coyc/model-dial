#!/usr/bin/env python3
"""Unit tests for src/capabilities.py — _norm_id normalisation helper.

Tests cover all transformation steps applied by _norm_id:
lowercasing, colon suffixes, date stripping, quantisation suffixes,
provider-prefix stripping, and dot→underscore normalisation.
"""

import pytest

from src.capabilities import _norm_id


# =========================================================================
# Baseline: no transformation needed
# =========================================================================

class TestNormIdPassthrough:
    """Ids that should pass through _norm_id unchanged (except lowercasing)."""

    def test_simple_model_id(self):
        assert _norm_id("gpt-4") == "gpt-4"

    def test_already_lowercase(self):
        assert _norm_id("llama-3") == "llama-3"

    def test_underscores_preserved(self):
        assert _norm_id("some_model_v2") == "some_model_v2"

    def test_no_dash_suffix_to_strip(self):
        assert _norm_id("bert-base-uncased") == "bert-base-uncased"


# =========================================================================
# Lowercasing
# =========================================================================

class TestNormIdLowercases:
    """Uppercase letters are folded to lowercase."""

    def test_uppercase_provider(self):
        # Provider prefix is stripped, so result is just the model name in lowercase
        assert _norm_id("OpenAI/GPT-4") == "gpt-4"

    def test_uppercase_model_name(self):
        assert _norm_id("GPT-4") == "gpt-4"

    def test_all_caps_model(self):
        assert _norm_id("LLAMA-3-70B") == "llama-3-70b"


# =========================================================================
# Colon suffix stripping
# =========================================================================

class TestNormIdStripsColonSuffixes:
    """Trailing :<word> suffixes are removed."""

    def test_colon_free(self):
        assert _norm_id("model:free") == "model"

    def test_colon_variant(self):
        assert _norm_id("model:standard") == "model"

    def test_colon_not_at_end(self):
        # colon in the middle is not stripped
        assert _norm_id("model:free-pro") == "model:free-pro"


# =========================================================================
# Date stripping (−YYYY-MM-DD and −YYYYMMDD)
# =========================================================================

class TestNormIdStripsDates:
    """ISO-style and compact date suffixes are removed."""

    def test_iso_date(self):
        assert _norm_id("model-2026-06-08") == "model"

    def test_compact_date(self):
        assert _norm_id("model-20260420") == "model"

    def test_date_after_model_name(self):
        assert _norm_id("qwen3-max-2026-06-08") == "qwen3-max"

    def test_date_in_middle_kept_if_not_iso(self):
        # -02-15 is also stripped by the repeated-date rule
        assert _norm_id("model-02-15") == "model"


# =========================================================================
# Repeated / short date stripping (−DD-MM)
# =========================================================================

class TestNormIdStripsShortDates:
    """Short date patterns (DD-MM) are stripped after ISO-date pass."""

    def test_short_date(self):
        assert _norm_id("model-02-15") == "model"


# =========================================================================
# −preview / −latest stripping
# =========================================================================

class TestNormIdStripsPreviewLatest:
    """Trailing -preview and -latest are removed."""

    def test_strips_preview(self):
        assert _norm_id("gemini-2.5-flash-preview") == "gemini-2_5-flash"

    def test_strips_latest(self):
        assert _norm_id("model-latest") == "model"

    def test_does_not_strip_preview_in_middle(self):
        # only trailing -preview / -latest are stripped
        assert _norm_id("model-preview-pro") == "model-preview-pro"


# =========================================================================
# Quantisation / format suffix stripping
# =========================================================================

class TestNormIdStripsQuantisationSuffixes:
    """Known quantisation/format suffixes are removed."""

    @pytest.mark.parametrize(
        "model_id, expected",
        [
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", "nvidia-nemotron-3-ultra-550b-a55b"),
            ("meta/Llama-3-70B-BF16", "llama-3-70b"),
            ("model-FP8", "model"),
            ("model-AWQ", "model"),
            ("model-GPTQ", "model"),
            ("model-GGUF", "model"),
            ("model-FP16", "model"),
            ("model-FP32", "model"),
            ("model-INT8", "model"),
            ("model-INT4", "model"),
        ],
    )
    def test_strips_suffix(self, model_id, expected):
        assert _norm_id(model_id) == expected


# =========================================================================
# Provider prefix stripping (everything before /)
# =========================================================================

class TestNormIdStripsProviderPrefix:
    """The part before / is removed, keeping only the model name."""

    def test_strips_openai_prefix(self):
        assert _norm_id("openai/gpt-4") == "gpt-4"

    def test_strips_google_prefix(self):
        assert _norm_id("google/gemini-1.5-pro") == "gemini-1_5-pro"

    def test_strips_nvidia_prefix(self):
        assert _norm_id("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4") == "nvidia-nemotron-3-ultra-550b-a55b"

    def test_no_provider_prefix_unchanged(self):
        assert _norm_id("gpt-4") == "gpt-4"


# =========================================================================
# Dot → underscore normalisation
# =========================================================================

class TestNormIdNormalisesDots:
    """Dots in model names are converted to underscores."""

    def test_qwen_dotted(self):
        assert _norm_id("qwen3.7-max") == "qwen3_7-max"

    def test_llama_dotted(self):
        assert _norm_id("llama-3.1") == "llama-3_1"

    def test_multiple_dots(self):
        assert _norm_id("model.1.2.3") == "model_1_2_3"


# =========================================================================
# Combined transformations
# =========================================================================

class TestNormIdCombined:
    """Complex ids that trigger multiple transformation steps."""

    def test_nvidia_nemotron_nvfp4(self):
        # provider prefix stripped + NVFP4 suffix stripped
        assert _norm_id("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4") == "nvidia-nemotron-3-ultra-550b-a55b"

    def test_qwen_with_date(self):
        # dot→underscore + ISO date stripped
        assert _norm_id("qwen3.7-max-2026-06-08") == "qwen3_7-max"

    def test_google_with_preview(self):
        # provider prefix + dot→underscore + trailing -preview stripped
        assert _norm_id("google/gemini-2.5-flash-preview") == "gemini-2_5-flash"

    def test_full_pipeline(self):
        # all transformations combined
        assert _norm_id("openai/gpt-4-turbo-2026-06-08-preview") == "gpt-4-turbo"

    def test_quantisation_with_provider(self):
        assert _norm_id("mistralai/Mistral-7B-Instruct-v0.2-AWQ") == "mistral-7b-instruct-v0_2"


# =========================================================================
# Edge cases
# =========================================================================

class TestNormIdEdgeCases:
    """Surprising or boundary inputs."""

    def test_empty_string(self):
        assert _norm_id("") == ""

    def test_only_provider_slash(self):
        assert _norm_id("provider/") == ""

    def test_single_character(self):
        assert _norm_id("a") == "a"

    def test_hyphen_only(self):
        assert _norm_id("---") == "---"
