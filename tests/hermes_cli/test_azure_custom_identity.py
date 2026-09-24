from __future__ import annotations


def test_doctor_checks_credentials_for_raw_named_custom_provider(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli import config, doctor_config

    custom_config = {
        "model": {"provider": "azure", "default": "deployment"},
        "providers": {
            "azure": {
                "name": "Private Azure Proxy",
                "base_url": "https://private.example/v1",
                "key_env": "PRIVATE_AZURE_KEY",
            },
        },
    }
    monkeypatch.setattr(config, "read_user_config_raw", lambda _path: custom_config)
    monkeypatch.setattr(config, "get_env_value", lambda name: "private-key" if name == "PRIVATE_AZURE_KEY" else None)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", SimpleNamespace(_DHH="~/.hermes"))
    from hermes_cli import auth
    monkeypatch.setattr(auth, "get_auth_status", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("custom Azure credentials must not be looked up through the Foundry registry")))
    issues = []

    doctor_config._validate_model_config("unused", issues)

    assert not any("no API key is configured" in issue for issue in issues)


def test_doctor_reports_missing_custom_alias_credential(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli import config, doctor_config

    custom_config = {
        "model": {"provider": "azure", "default": "deployment"},
        "providers": {"azure": {"name": "Private Azure Proxy", "base_url": "https://private.example/v1",
                                  "key_env": "PRIVATE_AZURE_KEY"}},
    }
    monkeypatch.setattr(config, "read_user_config_raw", lambda _path: custom_config)
    monkeypatch.setattr(config, "get_env_value", lambda _name: None)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", SimpleNamespace(_DHH="~/.hermes"))
    issues = []

    doctor_config._validate_model_config("unused", issues)

    assert any("No credentials found for provider 'azure'" in issue for issue in issues)


def test_doctor_accepts_inline_key_for_custom_alias(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli import config, doctor_config

    custom_config = {
        "model": {"provider": "azure", "default": "deployment"},
        "providers": {"azure": {"name": "Private Azure Proxy", "base_url": "https://private.example/v1",
                                  "api_key": "inline-private-key"}},
    }
    monkeypatch.setattr(config, "read_user_config_raw", lambda _path: custom_config)
    monkeypatch.setattr(config, "get_env_value", lambda _name: None)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", SimpleNamespace(_DHH="~/.hermes"))
    issues = []

    doctor_config._validate_model_config("unused", issues)

    assert not any("no API key is configured" in issue for issue in issues)


def test_status_labels_named_custom_alias_by_its_configured_name(monkeypatch):
    from hermes_cli import status

    custom_config = {
        "model": {"provider": "azure"},
        "providers": {
            "azure": {
                "name": "Private Azure Proxy",
                "base_url": "https://private.example/v1",
                "key_env": "PRIVATE_AZURE_KEY",
            },
        },
    }
    monkeypatch.setattr(status, "resolve_requested_provider", lambda: "azure")
    monkeypatch.setattr(status, "resolve_provider", lambda *_args, **_kwargs: "azure-foundry")
    monkeypatch.setattr(status, "load_config", lambda: custom_config)
    monkeypatch.setattr(status, "has_named_custom_provider", lambda provider: provider == "azure")

    assert status._effective_provider_label() == "Private Azure Proxy"
