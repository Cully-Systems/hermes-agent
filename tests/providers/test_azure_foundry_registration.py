"""Azure Foundry registration/runtime invariants.

Keep this fix to two behavior-level tests per repository policy.
"""

from __future__ import annotations

import uuid

from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool
from agent.transports import get_transport
from hermes_cli import auth, auth_commands, config
from hermes_cli import runtime_provider as rp
from providers import get_provider_profile


_AZURE_FOUNDRY_ALIASES = ("azure", "azure-ai-foundry", "azure-ai")


def test_azure_foundry_registration_aliases_and_pool_runtime(monkeypatch, tmp_path):
    profile = get_provider_profile("azure-foundry")
    assert profile is not None
    assert profile.name == "azure-foundry"
    assert profile.auth_type == "api_key"

    cfg = auth.PROVIDER_REGISTRY["azure-foundry"]
    assert cfg.id == "azure-foundry"
    for alias in _AZURE_FOUNDRY_ALIASES:
        assert get_provider_profile(alias) is profile
        assert auth.PROVIDER_REGISTRY[alias] is cfg
        assert auth.resolve_provider(alias) == "azure-foundry"
        assert auth_commands._normalize_provider(alias) == "azure-foundry"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "azure", "base_url": "https://example.openai.azure.com/openai/v1",
        "api_mode": "chat_completions", "default": "DeepSeek-R1",
    })
    load_pool("azure-foundry").add_entry(PooledCredential(
        provider="azure-foundry", id=uuid.uuid4().hex[:6], label="api-key-1", auth_type=AUTH_TYPE_API_KEY,
        priority=0, source=SOURCE_MANUAL, access_token="az-pool-key",
    ))
    status = auth.get_auth_status("azure-foundry")
    assert status["logged_in"] is True
    assert status["key_source"] == "credential_pool:azure-foundry"

    resolved = rp.resolve_runtime_provider(requested="azure")
    assert resolved["provider"] == "azure-foundry"
    assert resolved["api_key"] == "az-pool-key"
    assert resolved["source"] == "manual"
    assert resolved["base_url"] == "https://example.openai.azure.com/openai/v1"
    assert get_transport(resolved["api_mode"]) is not None


def test_azure_alias_policy_preserves_custom_precedence_and_auto_detect_exclusions(monkeypatch, tmp_path):
    from agent import auxiliary_client as aux
    import providers as provider_registry

    real_list_providers = provider_registry.list_providers
    real_get_provider_aliases = provider_registry.get_provider_aliases
    real_get_provider_profile = provider_registry.get_provider_profile
    original_profiles = provider_registry._REGISTRY.copy()
    original_aliases = provider_registry._ALIASES.copy()
    real_resolve_runtime_provider = rp.resolve_runtime_provider
    real_ladder_rungs = rp._ladder_rungs

    monkeypatch.setattr(config, "load_config", lambda: {"model": {"provider": "azure"}})
    assert auth._config_model_provider()[1] == "azure-foundry"
    monkeypatch.setattr(config, "load_config", lambda: {
        "model": {"provider": "azure"},
        "providers": {"azure": {"base_url": "https://custom.example/v1", "api_key": "custom-key"}},
    })
    assert auth._config_model_provider()[1] == "custom"

    custom = {
        "name": "azure",
        "base_url": "https://custom.example/v1",
        "api_mode": "chat_completions",
        "model": "custom-model",
    }
    monkeypatch.setattr(rp, "has_named_custom_provider", lambda provider: provider == "azure")
    monkeypatch.setattr(rp._config_mod, "load_config", lambda: {"providers": {"azure-foundry": {"enabled": False}}})
    expected = {
        "provider": "custom",
        "api_mode": "chat_completions",
        "base_url": "https://custom.example/v1",
        "api_key": "custom-key",
    }
    monkeypatch.setattr(rp, "_ladder_rungs", lambda *_args, **_kwargs: iter([expected]))
    assert rp.resolve_runtime_provider(requested="azure") is expected
    assert rp._config_base_url_for_provider(
        {"provider": "azure", "base_url": custom["base_url"]}, "azure-foundry"
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

    # A stale Entra mode must not suppress the canonical pool after the user switches
    # away from Foundry. When Foundry remains selected, Entra keeps its token path.
    pooled = {"provider": "azure-foundry", "api_key": "pool-key", "source": "manual",
              "base_url": "https://example.openai.azure.com/v1"}
    pool_calls = []
    monkeypatch.setattr(rp, "has_named_custom_provider", lambda _provider: False)
    monkeypatch.setattr(rp, "_resolve_from_pool", lambda *args: pool_calls.append(args) or pooled)
    foundry_runtime = {"provider": "azure-foundry", "source": "entra"}
    monkeypatch.setattr(rp, "_resolve_azure_foundry_runtime", lambda **_kwargs: foundry_runtime)

    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "openai", "auth_mode": "entra_id", "base_url": "https://example.openai.azure.com/v1",
    })
    assert rp._resolve_requested_shortcuts("azure-foundry", None, None, None) is pooled
    assert pool_calls[-1][0] == "azure-foundry"

    calls_before = len(pool_calls)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "azure-foundry", "auth_mode": "entra_id",
        "base_url": "https://example.openai.azure.com/v1",
    })
    assert rp._resolve_requested_shortcuts("azure-foundry", None, None, None) is foundry_runtime
    assert len(pool_calls) == calls_before

    # Auxiliary calls share the main resolver so canonical pool credentials are
    # available for title/compression/vision requests as well as main chat.
    monkeypatch.setattr(config, "load_config_readonly", lambda: {
        "model": {"provider": "openai", "auth_mode": "api_key", "default": "deployment"},
    })
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "openai", "auth_mode": "api_key", "default": "deployment",
    })
    monkeypatch.setattr(rp, "resolve_runtime_provider", lambda **kwargs: rp._resolve_requested_shortcuts(
        kwargs["requested"], kwargs["explicit_api_key"], kwargs["explicit_base_url"], kwargs["target_model"],
    ))
    monkeypatch.setattr(aux, "_create_openai_client", lambda **kwargs: kwargs)
    client, model = aux._try_azure_foundry(model="deployment")
    assert client["api_key"] == "pool-key"
    assert client["base_url"] == pooled["base_url"]
    assert model == "deployment"

    # Discovered provider aliases obey the same last-writer-wins rule as profile
    # registration and must take precedence over the static Azure Foundry shortcut.
    from providers.base import ProviderProfile

    custom_profile = ProviderProfile(
        name="custom-azure", aliases=("claude",), env_vars=("CUSTOM_AZURE_API_KEY",),
        base_url="https://custom-azure.example/v1")
    monkeypatch.setattr("providers.list_providers", lambda: [custom_profile])
    monkeypatch.setattr("providers.get_provider_aliases", lambda: {"claude": "custom-azure"})
    monkeypatch.setattr(
        "providers.get_provider_profile",
        lambda name: custom_profile if name in {"claude", "custom-azure"} else None,
    )
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", dict(auth.PROVIDER_REGISTRY))
    auth._register_plugin_provider(custom_profile)
    assert auth.PROVIDER_REGISTRY["claude"] is auth.PROVIDER_REGISTRY["custom-azure"]
    assert auth._plugin_aliases()["claude"] == "custom-azure"
    assert auth.resolve_provider("claude") == "custom-azure"
    from agent.agent_init import _provider_default_routes

    plugin_routes = _provider_default_routes("custom-azure")
    assert "https://custom-azure.example/v1" in plugin_routes
    assert "https://api.anthropic.com" not in plugin_routes
    monkeypatch.setattr(rp, "has_named_custom_provider", lambda _provider: False)
    assert rp._resolve_requested_shortcuts("claude", None, None, None) is None

    # Load a real user profile through the provider discovery loader. A later
    # same-name override owns its alias and must rebuild auth's registry config
    # from the winning profile rather than reusing the bundled Foundry settings.
    monkeypatch.setattr(provider_registry, "_REGISTRY", original_profiles.copy())
    monkeypatch.setattr(provider_registry, "_ALIASES", original_aliases.copy())
    monkeypatch.setattr(provider_registry, "_discovered", True)
    monkeypatch.setattr(provider_registry, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(provider_registry, "list_providers", real_list_providers)
    monkeypatch.setattr(provider_registry, "get_provider_aliases", real_get_provider_aliases)
    monkeypatch.setattr(provider_registry, "get_provider_profile", real_get_provider_profile)
    plugin_dir = tmp_path / "hermes-home" / "plugins" / "model-providers" / f"azure-foundry-override-{uuid.uuid4().hex}"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(\n"
        "    name='azure-foundry', aliases=('azure-custom',),\n"
        "    env_vars=('CUSTOM_AZURE_API_KEY', 'CUSTOM_AZURE_BASE_URL'),\n"
        "    base_url='https://custom-azure.example/v1',\n"
        "))\n",
        encoding="utf-8",
    )
    provider_registry._import_plugin_dir(plugin_dir, "user")
    azure_override = provider_registry.get_provider_profile("azure-foundry")
    assert azure_override is not None
    assert provider_registry.get_provider_profile("azure-custom") is azure_override
    assert auth._plugin_aliases()["azure-custom"] == "azure-foundry"
    assert auth.resolve_provider("azure-custom") == "azure-foundry"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("CUSTOM_AZURE_API_KEY", "custom-profile-key")
    monkeypatch.setenv("CUSTOM_AZURE_BASE_URL", "https://custom-azure.example/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "stale-bundled-key")
    old_config = auth.PROVIDER_REGISTRY["azure-foundry"]
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, "azure-foundry", old_config)
    monkeypatch.delitem(auth.PROVIDER_REGISTRY, "azure-custom", raising=False)
    for alias in ("azure", "azure-ai-foundry", "azure-ai"):
        monkeypatch.setitem(auth.PROVIDER_REGISTRY, alias, auth.PROVIDER_REGISTRY[alias])
    auth._register_plugin_provider(azure_override)
    monkeypatch.setattr(rp, "PROVIDER_REGISTRY", auth.PROVIDER_REGISTRY)
    assert auth.PROVIDER_REGISTRY["azure-foundry"].api_key_env_vars == ("CUSTOM_AZURE_API_KEY",)
    assert auth.PROVIDER_REGISTRY["azure-foundry"].base_url_env_var == "CUSTOM_AZURE_BASE_URL"
    assert rp._resolve_requested_shortcuts("azure-custom", None, None, None) is None

    # The same-name override must flow through the ordinary profile-backed API-key
    # resolver, preserving its URL and credentials instead of taking the bundled
    # Foundry shortcut (which requires Azure-specific config and keys).
    monkeypatch.setattr(rp._config_mod, "load_config", lambda: {"model": {"provider": "azure-custom"}})
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "azure-custom"})
    monkeypatch.setattr(rp._config_mod, "load_config", lambda: {"model": {"provider": "azure-custom"}})
    monkeypatch.setattr(rp, "_resolve_from_pool", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rp, "resolve_runtime_provider", real_resolve_runtime_provider)
    monkeypatch.setattr(rp, "_ladder_rungs", real_ladder_rungs)
    custom_runtime = rp.resolve_runtime_provider(requested="azure-custom")
    assert custom_runtime["provider"] == "azure-foundry"
    assert custom_runtime["api_key"] == "custom-profile-key"
    assert custom_runtime["base_url"] == "https://custom-azure.example/v1"
