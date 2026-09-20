"""Invariant: ``provider: azure-foundry`` resolves a registered adapter.

The quoted runtime error ``no adapter registered for provider "azure-foundry"``
is not raised in this tree — transports are keyed by api_mode, and the
in-tree adapter is the bundled ``AzureFoundryProfile`` plus the dedicated
runtime resolver. These tests pin that chain so a missing plugin, dropped
alias, or skipped Foundry shortcut cannot silently pass.

Sibling providers (openrouter, deepseek) stay registered — this file must not
regress the rest of the model-provider set.
"""

from __future__ import annotations

import sys

import pytest


_AZURE_FOUNDRY_ALIASES = ("azure", "azure-ai-foundry", "azure-ai")


def _clear_provider_caches():
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    for mod in list(sys.modules):
        if mod.startswith(("plugins.model_providers", "_hermes_user_provider")):
            del sys.modules[mod]


@pytest.fixture
def rediscovered_providers():
    _clear_provider_caches()
    yield
    _clear_provider_caches()


def test_azure_foundry_profile_registers_name_and_aliases(rediscovered_providers):
    from providers import get_provider_profile, list_providers

    profile = get_provider_profile("azure-foundry")
    assert profile is not None
    assert profile.name == "azure-foundry"
    loaded = sys.modules.get("plugins.model_providers.azure_foundry")
    assert loaded is not None, "bundled azure-foundry plugin was not imported"
    assert isinstance(profile, loaded.AzureFoundryProfile)
    for alias in _AZURE_FOUNDRY_ALIASES:
        assert get_provider_profile(alias) is profile

    names = {p.name for p in list_providers()}
    assert "azure-foundry" in names
    assert "openrouter" in names
    assert "deepseek" in names


def test_azure_foundry_auth_registry_includes_canonical_and_aliases():
    """resolve_provider / auxiliary_client look up PROVIDER_REGISTRY by id."""
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider

    cfg = PROVIDER_REGISTRY.get("azure-foundry")
    assert cfg is not None
    assert cfg.id == "azure-foundry"
    assert "AZURE_FOUNDRY_API_KEY" in cfg.api_key_env_vars
    for alias in _AZURE_FOUNDRY_ALIASES:
        assert PROVIDER_REGISTRY.get(alias) is cfg
        assert resolve_provider(alias) == "azure-foundry"
    assert resolve_provider("azure-foundry") == "azure-foundry"


def test_azure_foundry_get_provider_resolves_overlay_and_aliases():
    """hermes_cli.providers.get_provider backs picker / aux identity preservation."""
    from hermes_cli.providers import get_provider, normalize_provider

    resolved = get_provider("azure-foundry")
    assert resolved is not None
    assert resolved.id == "azure-foundry"
    for alias in _AZURE_FOUNDRY_ALIASES:
        assert normalize_provider(alias) == "azure-foundry"
        assert get_provider(alias) is not None
        assert get_provider(alias).id == "azure-foundry"


def test_azure_foundry_runtime_resolves_adapter_and_transport(monkeypatch):
    """A config using provider: azure-foundry must produce a runtime + transport.

    DeepSeek (and other non-GPT-5 Foundry models) stay on chat_completions.
    """
    from agent.transports import get_transport
    from hermes_cli import runtime_provider as rp
    from hermes_cli.models import azure_foundry_model_api_mode

    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-test-key")
    monkeypatch.setenv(
        "AZURE_FOUNDRY_BASE_URL",
        "https://example.services.ai.azure.com/openai/v1",
    )
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "azure-foundry",
            "base_url": "https://example.services.ai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "default": "DeepSeek-V3",
        },
    )
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)

    resolved = rp.resolve_runtime_provider(requested="azure-foundry")

    assert resolved["provider"] == "azure-foundry"
    assert resolved["api_key"] == "az-test-key"
    assert resolved["base_url"].rstrip("/") == (
        "https://example.services.ai.azure.com/openai/v1"
    )
    assert azure_foundry_model_api_mode("DeepSeek-V3") is None
    assert resolved["api_mode"] == "chat_completions"

    transport = get_transport(resolved["api_mode"])
    assert transport is not None


def test_azure_alias_uses_foundry_runtime_shortcut(monkeypatch):
    """provider: azure (plugin alias) must hit the Foundry resolver, not generic api_key."""
    from hermes_cli import runtime_provider as rp

    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-test-key")
    monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "azure",
            "base_url": "https://example.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "default": "DeepSeek-R1",
        },
    )
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)

    resolved = rp.resolve_runtime_provider(requested="azure")

    assert resolved["provider"] == "azure-foundry"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == "az-test-key"
