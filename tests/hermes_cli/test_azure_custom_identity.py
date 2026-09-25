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


def test_doctor_accepts_inline_key_for_legacy_custom_provider(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli import config, doctor_config

    legacy_entry = {
        "name": "claude",
        "base_url": "https://private.example/v1",
        "api_key": "legacy-inline-key",
    }
    custom_config = {
        "model": {"provider": "custom:claude", "default": "deployment"},
        "custom_providers": [legacy_entry],
    }
    monkeypatch.setattr(config, "read_user_config_raw", lambda _path: custom_config)
    monkeypatch.setattr(config, "get_env_value", lambda _name: None)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", SimpleNamespace(_DHH="~/.hermes"))
    issues = []

    doctor_config._validate_model_config("unused", issues)

    assert not any("no API key is configured" in issue for issue in issues)


def test_doctor_keeps_canonical_builtin_credentials_when_custom_name_collides(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli import auth, config, doctor_config

    custom_config = {
        "model": {"provider": "anthropic", "default": "claude-sonnet-4-5"},
        "providers": {"anthropic": {"base_url": "https://private.example/v1", "api_key": "custom-key"}},
    }
    monkeypatch.setattr(config, "read_user_config_raw", lambda _path: custom_config)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", SimpleNamespace(_DHH="~/.hermes"))
    monkeypatch.setattr(auth, "get_auth_status", lambda _provider: {"configured": False})
    issues = []

    doctor_config._validate_model_config("unused", issues)

    assert any("No credentials found for provider 'anthropic'" in issue for issue in issues)


def test_doctor_accepts_key_command_for_custom_provider(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli import config, doctor_config

    custom_config = {
        "model": {"provider": "azure", "default": "deployment"},
        "providers": {"azure": {"base_url": "https://private.example/v1", "key_cmd": "token-helper"}},
    }
    monkeypatch.setattr(config, "read_user_config_raw", lambda _path: custom_config)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", SimpleNamespace(_DHH="~/.hermes"))
    issues = []

    doctor_config._validate_model_config("unused", issues)

    assert not any("no API key is configured" in issue for issue in issues)


def test_doctor_accepts_custom_provider_credential_pool(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli import config, doctor_config, runtime_provider

    custom_config = {
        "model": {"provider": "azure", "default": "deployment"},
        "providers": {"azure": {"base_url": "https://private.example/v1"}},
    }
    monkeypatch.setattr(config, "read_user_config_raw", lambda _path: custom_config)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", SimpleNamespace(_DHH="~/.hermes"))
    pool_lookups = []
    monkeypatch.setattr(runtime_provider, "_try_resolve_from_custom_pool",
                        lambda base_url, provider, provider_name=None: pool_lookups.append((base_url, provider, provider_name)) or {"api_key": "pool-key"})
    issues = []

    doctor_config._validate_model_config("unused", issues)

    assert not any("no API key is configured" in issue for issue in issues)
    assert pool_lookups == [("https://private.example/v1", "custom", "azure")]


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


def test_endpointless_plugin_azure_alias_owns_catalog_cache_and_doctor_identity(monkeypatch):
    from types import SimpleNamespace
    import time

    from hermes_cli import auth, config, doctor_config, models
    from hermes_cli import providers as provider_defs
    from providers import ProviderProfile

    profile = ProviderProfile(
        name="custom-azure-plugin",
        display_name="Dynamic Azure Plugin",
        aliases=("azure",),
        env_vars=("CUSTOM_AZURE_PLUGIN_KEY",),
        fallback_models=("dynamic-azure-model",),
        base_url="",
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile if name in {"azure", profile.name} else None)
    monkeypatch.setattr("providers.list_providers", lambda: [profile])
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, profile.name, auth.ProviderConfig(
        id=profile.name, name=profile.display_name, auth_type="api_key",
        api_key_env_vars=profile.env_vars))
    monkeypatch.setattr(provider_defs, "_models_dev_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(models, "_api_key_credentials", lambda _provider: ("", ""))

    assert models.normalize_provider("azure") == profile.name
    assert provider_defs.normalize_provider("azure") == profile.name
    assert models.provider_model_ids("azure") == ["dynamic-azure-model"]
    assert models.curated_models_for_provider("azure") == [("dynamic-azure-model", "")]
    assert models._normalized_cache_slug("azure") == profile.name
    now = time.time()
    monkeypatch.setattr(models, "_load_provider_models_cache", lambda: {
        profile.name: {"fp": "plugin-fp", "at": now, "models": ["plugin-cache-model"]},
        "azure-foundry": {"fp": "plugin-fp", "at": now, "models": ["wrong-foundry-model"]},
    })
    monkeypatch.setattr(models, "_credential_fingerprint", lambda _provider: "plugin-fp")
    assert models.cached_provider_model_ids("azure") == ["plugin-cache-model"]

    provider_def = provider_defs.get_provider("azure", allow_network=False)
    assert provider_def is not None
    assert provider_def.id == profile.name
    assert provider_def.base_url == ""
    alias_def = provider_defs.resolve_provider_full("azure")
    assert alias_def is not None
    assert alias_def.id == profile.name
    assert alias_def.source == "plugin-profile"
    assert alias_def.base_url == ""

    raw_config = {"model": {"provider": "azure", "default": "dynamic-azure-model"}}
    monkeypatch.setattr(config, "read_user_config_raw", lambda _path: raw_config)
    monkeypatch.setattr(auth, "get_auth_status", lambda *_args, **_kwargs: {"configured": True})
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", SimpleNamespace(_DHH="~/.hermes"))
    issues = []
    doctor_config._validate_model_config("unused", issues)
    assert not any("not a recognised provider" in issue for issue in issues)
    assert not any("no API key is configured" in issue for issue in issues)
