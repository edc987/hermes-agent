"""Tests for lookup_custom_provider_context_length helper.

Verifies that context_length is correctly resolved from custom_providers
config entries, and that edge cases (missing model, invalid values, empty
lists) return None gracefully.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


CUSTOM_PROVIDERS = [
    {
        "name": "my-provider",
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "models": {
            "model-a": {"context_length": 1000000},
            "model-b": {"context_length": 262144},
            "model-bad": {"context_length": "256K"},  # invalid — not a plain int
        },
    },
    {
        "name": "other-provider",
        "base_url": "https://other.example.com/v1",
        "model": "model-c",
        "models": {
            "model-c": {"context_length": 500000},
        },
    },
]


class TestLookupCustomProviderContextLength:
    """lookup_custom_provider_context_length — happy path."""

    def test_exact_match(self):
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "model-a", "https://api.example.com/v1", CUSTOM_PROVIDERS,
        )
        assert ctx == 1_000_000

    def test_trailing_slash_normalised(self):
        """base_url with trailing slash should still match."""
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "model-a", "https://api.example.com/v1/", CUSTOM_PROVIDERS,
        )
        assert ctx == 1_000_000

    def test_second_provider_match(self):
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "model-c", "https://other.example.com/v1", CUSTOM_PROVIDERS,
        )
        assert ctx == 500_000

    def test_second_model_in_same_provider(self):
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "model-b", "https://api.example.com/v1", CUSTOM_PROVIDERS,
        )
        assert ctx == 262_144


class TestLookupCustomProviderContextLengthMisses:
    """lookup_custom_provider_context_length — miss cases return None."""

    def test_model_not_in_provider(self):
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "unknown-model", "https://api.example.com/v1", CUSTOM_PROVIDERS,
        )
        assert ctx is None

    def test_base_url_no_match(self):
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "model-a", "https://unrelated.example.com/v1", CUSTOM_PROVIDERS,
        )
        assert ctx is None

    def test_empty_custom_providers(self):
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "model-a", "https://api.example.com/v1", [],
        )
        assert ctx is None

    def test_none_custom_providers(self):
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "model-a", "https://api.example.com/v1", None,
        )
        assert ctx is None

    def test_empty_base_url(self):
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "model-a", "", CUSTOM_PROVIDERS,
        )
        assert ctx is None

    def test_invalid_context_length_returns_none(self):
        """Non-integer context_length (e.g. '256K') should return None."""
        from agent.model_metadata import lookup_custom_provider_context_length

        ctx = lookup_custom_provider_context_length(
            "model-bad", "https://api.example.com/v1", CUSTOM_PROVIDERS,
        )
        assert ctx is None

    def test_model_without_models_dict(self):
        """Provider entry with no 'models' key should not crash."""
        from agent.model_metadata import lookup_custom_provider_context_length

        providers = [{"name": "x", "base_url": "https://x.com/v1"}]
        ctx = lookup_custom_provider_context_length(
            "any-model", "https://x.com/v1", providers,
        )
        assert ctx is None

    def test_non_dict_entry_skipped(self):
        """Non-dict entries in custom_providers list are silently skipped."""
        from agent.model_metadata import lookup_custom_provider_context_length

        providers = ["not-a-dict", {"base_url": "https://x.com/v1", "models": {}}]
        ctx = lookup_custom_provider_context_length(
            "m", "https://x.com/v1", providers,
        )
        assert ctx is None
