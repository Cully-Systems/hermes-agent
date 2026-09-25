from __future__ import annotations

import pytest


@pytest.mark.parametrize("alias", ["azure", "azure-ai-foundry", "azure-ai"])
def test_azure_wizard_defaults_load_from_alias_config(monkeypatch, alias):
    from hermes_cli import auth, config
    from hermes_cli.model_setup_flows_azure import _azure_current

    monkeypatch.setattr(config, "get_env_value", lambda _name: "")
    monkeypatch.setattr(config, "load_config", lambda: {})
    monkeypatch.setattr(auth, "_plugin_aliases", lambda: {
        "azure": "azure-foundry",
        "azure-ai-foundry": "azure-foundry",
        "azure-ai": "azure-foundry",
    })

    current = _azure_current({"model": {
        "provider": alias,
        "base_url": "https://resource.openai.azure.com/openai/v1",
        "api_mode": "anthropic_messages",
        "auth_mode": "entra_id",
        "entra": {"scope": "https://custom.scope/.default"},
    }})

    assert current.base_url == "https://resource.openai.azure.com/openai/v1"
    assert current.api_mode == "anthropic_messages"
    assert current.auth_mode == "entra_id"
    assert current.entra == {"scope": "https://custom.scope/.default"}


def test_azure_wizard_does_not_load_named_custom_provider_defaults(monkeypatch):
    from hermes_cli import config
    from hermes_cli.model_setup_flows_azure import _azure_current

    custom_config = {
        "model": {
            "provider": "azure",
            "base_url": "https://private.example/v1",
            "api_mode": "chat_completions",
            "auth_mode": "entra_id",
        },
        "providers": {
            "azure": {
                "name": "Private Azure Proxy",
                "base_url": "https://private.example/v1",
                "key_env": "PRIVATE_AZURE_KEY",
            },
        },
    }
    monkeypatch.setattr(config, "get_env_value", lambda _name: "")
    monkeypatch.setattr(config, "load_config", lambda: custom_config)

    current = _azure_current(custom_config)

    assert current.base_url == ""
    assert current.api_mode == ""
    assert current.auth_mode == "api_key"
