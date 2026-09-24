"""Azure Foundry registration/runtime invariants.

Keep this fix to two behavior-level tests per repository policy.
"""

from __future__ import annotations

import pytest

_AZURE_FOUNDRY_ALIASES = ("azure", "azure-ai-foundry", "azure-ai")


def test_azure_foundry_registration_aliases_and_runtime(monkeypatch):
    from agent.transports import get_transport
    from hermes_cli import auth
    from hermes_cli import auth_commands
    from hermes_cli import runtime_provider as rp
    from providers import get_provider_profile

    profile = get_provider_profile("azure-foundry")
    assert profile is not None
    assert profile.name == "azure-foundry"
    assert profile.auth_type == "api_key"

    cfg = auth.PROVIDER_REGISTRY["azure-foundry"]
    for alias in _AZURE_FOUNDRY_ALIASES:
        assert get_provider_profile(alias) is profile
        assert auth.PROVIDER_REGISTRY[alias] is cfg
        assert auth.resolve_provider(alias) == "azure-foundry"
        assert auth_commands._normalize_provider(alias) == "azure-foundry"

    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-test-key")
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
    assert get_transport(resolved["api_mode"]) is not None


def test_azure_alias_policy_preserves_custom_precedence_and_auto_detect_exclusions(monkeypatch):
    from agent import auxiliary_client as aux
    from hermes_cli import auth
    from hermes_cli import runtime_provider as rp

    custom = {
        "name": "azure",
        "base_url": "https://custom.example/v1",
        "api_mode": "chat_completions",
        "model": "custom-model",
    }
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
    monkeypatch.setattr(rp, "_ladder_rungs", lambda *_args, **_kwargs: iter([expected]))
    assert rp.resolve_runtime_provider(requested="azure") is expected
    assert rp._config_base_url_for_provider(
        {"provider": "azure", "base_url": custom["base_url"]},
        "azure-foundry",
    ) == ""

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

    copilot = auth.PROVIDER_REGISTRY["copilot"]
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, "github", copilot)
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, "github-copilot", copilot)
    detected = auth._env_key_auto_detected(
        lambda name: "ambient-github-token" if name == "GITHUB_TOKEN" else "",
        None,
    )
    assert detected not in {"copilot", "github", "github-copilot"}
