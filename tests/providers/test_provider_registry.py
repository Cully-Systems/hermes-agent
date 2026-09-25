import pytest

from providers.base import ProviderProfile
import providers


@pytest.fixture(autouse=True)
def isolate_provider_registry():
    registry = providers._REGISTRY.copy()
    aliases = providers._ALIASES.copy()
    provider_list_cache = (
        None
        if providers._PROVIDER_LIST_CACHE is None
        else list(providers._PROVIDER_LIST_CACHE)
    )
    discovered = providers._discovered

    yield

    providers._REGISTRY.clear()
    providers._REGISTRY.update(registry)
    providers._ALIASES.clear()
    providers._ALIASES.update(aliases)
    providers._PROVIDER_LIST_CACHE = provider_list_cache
    providers._discovered = discovered


def _profile(name: str, *aliases: str) -> ProviderProfile:
    return ProviderProfile(name=name, aliases=aliases)


def _reset_registry() -> None:
    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = True


def test_list_providers_reuses_cached_snapshot_until_registration_changes():
    _reset_registry()
    first = _profile("alpha")
    providers.register_provider(first)

    listed = providers.list_providers()
    listed.clear()

    assert providers.list_providers() == [first]

    # Hit-path copy guard: mutating a CACHED return must not corrupt the
    # module-level snapshot for later callers (aliasing bug class).
    providers.list_providers().clear()
    assert providers.list_providers() == [first]

    second = _profile("beta")
    providers.register_provider(second)

    assert providers.list_providers() == [first, second]


def test_list_providers_dedupes_aliases_in_cached_snapshot():
    _reset_registry()
    profile = _profile("kimi", "moonshot", "kimi-k2")
    providers.register_provider(profile)

    assert providers.get_provider_profile("moonshot") is profile
    assert providers.list_providers() == [profile]


@pytest.mark.parametrize(
    ("provider_id", "legacy_id", "base_url_env_var"),
    [
        ("ai-gateway", "vercel", ""),
        ("kilocode", "kilo", "KILOCODE_BASE_URL"),
    ],
)
def test_profile_first_normalization_keeps_legacy_hermes_overlay(provider_id, legacy_id, base_url_env_var):
    from hermes_cli.providers import get_provider, is_aggregator

    profile = providers.get_provider_profile(provider_id)
    assert profile is not None
    assert providers.get_provider_profile(legacy_id) is profile

    provider = get_provider(provider_id, allow_network=False)
    assert provider is not None
    assert provider.id == profile.name
    assert provider.is_aggregator
    assert provider.base_url_env_var == base_url_env_var
    assert provider.base_url == profile.base_url
    assert is_aggregator(provider_id)
