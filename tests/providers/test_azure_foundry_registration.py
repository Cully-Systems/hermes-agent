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

    resolved = rp.resolve_runtime_provider(requested="azure")
    assert resolved["provider"] == "azure-foundry"
    assert resolved["api_key"] == "az-pool-key"
    assert resolved["source"] == "manual"
    assert resolved["base_url"] == "https://example.openai.azure.com/openai/v1"
    assert get_transport(resolved["api_mode"]) is not None


def test_azure_alias_policy_preserves_custom_precedence_and_auto_detect_exclusions(monkeypatch):
    from agent import auxiliary_client as aux
    import providers as provider_registry

    real_list_providers = provider_registry.list_providers
    real_get_provider_aliases = provider_registry.get_provider_aliases

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

    custom_profile = ProviderProfile(name="custom-azure", aliases=("azure",))
    monkeypatch.setattr("providers.list_providers", lambda: [custom_profile])
    monkeypatch.setattr("providers.get_provider_aliases", lambda: {"azure": "custom-azure"})
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, "custom-azure", auth.ProviderConfig(
        id="custom-azure", name="Custom Azure", auth_type="api_key"))
    assert auth._plugin_aliases()["azure"] == "custom-azure"
    assert auth.resolve_provider("azure") == "custom-azure"
    monkeypatch.setattr(rp, "has_named_custom_provider", lambda _provider: False)
    assert rp._resolve_requested_shortcuts("azure", None, None, None) is None

    # Same-name replacement preserves the registry's original list slot, but
    # alias ownership follows actual registration order. The Anthropic profile
    # must not reclaim claude after a later Azure Foundry override registers it.
    monkeypatch.setattr(provider_registry, "_REGISTRY", {})
    monkeypatch.setattr(provider_registry, "_ALIASES", {})
    monkeypatch.setattr(provider_registry, "_discovered", True)
    monkeypatch.setattr(provider_registry, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(provider_registry, "list_providers", real_list_providers)
    monkeypatch.setattr(provider_registry, "get_provider_aliases", real_get_provider_aliases)
    provider_registry.register_provider(ProviderProfile(name="azure-foundry"))
    provider_registry.register_provider(ProviderProfile(name="anthropic", aliases=("claude",)))
    provider_registry.register_provider(ProviderProfile(name="azure-foundry", aliases=("claude",)))
    listed = provider_registry.list_providers()
    assert [profile.name for profile in listed] == ["azure-foundry", "anthropic"]
    assert auth._plugin_aliases()["claude"] == "azure-foundry"
    assert auth.resolve_provider("claude") == "azure-foundry"
