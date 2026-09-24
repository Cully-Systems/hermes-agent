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
    assert profile.auth_type == "api_key"
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


def test_azure_foundry_auxiliary_aliases_use_foundry_route(monkeypatch):
    """Auxiliary provider aliases, including provider: main, canonicalize to Foundry."""
    from agent import auxiliary_client as aux

    for alias in _AZURE_FOUNDRY_ALIASES:
        assert aux._normalize_aux_provider(alias) == "azure-foundry"

    monkeypatch.setattr(aux, "_read_main_provider", lambda: "azure")
    assert aux._normalize_aux_provider("main") == "azure-foundry"


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


def test_azure_foundry_auth_command_aliases_use_canonical_pool():
    from hermes_cli import auth_commands

    for alias in _AZURE_FOUNDRY_ALIASES:
        assert auth_commands._normalize_provider(alias) == "azure-foundry"


def test_azure_foundry_alias_honors_canonical_disablement(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        rp._config_mod,
        "load_config",
        lambda: {"providers": {"azure-foundry": {"enabled": False}}},
    )

    with pytest.raises(ValueError, match="azure-foundry.*disabled"):
        rp.resolve_runtime_provider(requested="azure")


def test_azure_alias_does_not_shadow_named_custom_provider(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(rp, "has_named_custom_provider", lambda provider: provider == "azure")
    monkeypatch.setattr(
        rp,
        "_resolve_azure_foundry_runtime",
        lambda **kwargs: pytest.fail("Azure Foundry shortcut must not shadow named custom provider"),
    )

    assert rp._resolve_requested_shortcuts("azure", None, None, "custom-model") is None


def test_custom_azure_alias_skips_canonical_disablement(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(rp, "has_named_custom_provider", lambda provider: provider == "azure")
    monkeypatch.setattr(
        rp._config_mod,
        "load_config",
        lambda: {"providers": {"azure-foundry": {"enabled": False}}},
    )
    expected = {
        "provider": "custom",
        "api_mode": "chat_completions",
        "base_url": "https://custom.example/v1",
        "api_key": "custom-key",
    }
    monkeypatch.setattr(
        rp,
        "_ladder_rungs",
        lambda *_args, **_kwargs: iter([expected]),
    )

    assert rp.resolve_runtime_provider(requested="azure") is expected


def test_foundry_does_not_reuse_custom_azure_base_url(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(rp, "has_named_custom_provider", lambda provider: provider == "azure")
    model_cfg = {
        "provider": "azure",
        "base_url": "https://custom.example/v1",
    }

    assert rp._config_base_url_for_provider(model_cfg, "azure-foundry") == ""
    assert rp._config_base_url_for_provider(model_cfg, "azure") == "https://custom.example/v1"


def test_aux_main_preserves_raw_custom_azure_name(monkeypatch):
    from agent import auxiliary_client as aux
    from hermes_cli import runtime_provider as rp

    custom = {
        "name": "azure",
        "base_url": "https://custom.example/v1",
        "api_mode": "chat_completions",
        "model": "custom-model",
    }
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "azure")
    monkeypatch.setattr(rp, "_get_named_custom_provider", lambda name: custom if name == "azure" else None)
    monkeypatch.setattr(aux, "_named_custom_api_key", lambda *_args, **_kwargs: "custom-key")
    fake_client = object()
    monkeypatch.setattr(aux, "_named_custom_openai_wire_client", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(aux, "_wrap_transport", lambda _req, client, *_args, **_kwargs: client)
    monkeypatch.setattr(aux, "_route_client", lambda _req, client, model: (client, model))

    client, model = aux.resolve_provider_client("main", model="custom-model")

    assert client is fake_client
    assert model == "custom-model"


def test_registry_aliases_do_not_bypass_auto_detect_exclusions(monkeypatch):
    from hermes_cli import auth

    copilot = auth.PROVIDER_REGISTRY["copilot"]
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, "github", copilot)
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, "github-copilot", copilot)

    detected = auth._env_key_auto_detected(
        lambda name: "ambient-github-token" if name == "GITHUB_TOKEN" else "",
        None,
    )

    assert detected != "copilot"
    assert detected not in {"github", "github-copilot"}
