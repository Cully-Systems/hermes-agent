"""Microsoft Foundry provider profile: OpenAI-compatible, per-resource base URL
supplied by the user at setup."""

from providers import register_provider
from providers.base import ProviderProfile


class AzureFoundryProfile(ProviderProfile):
    """Microsoft Foundry — OpenAI- and Anthropic-style endpoints on a user URL.

    Transports are selected by ``model.api_mode`` (chat_completions, anthropic_messages,
    codex_responses), not by this profile's default ``api_mode``. DeepSeek and most
    open-weight Foundry deployments stay on chat completions; GPT-5.x / Codex are
    upgraded at runtime via ``azure_foundry_model_api_mode``.
    """


azure_foundry = AzureFoundryProfile(
    name="azure-foundry",
    aliases=("azure", "azure-ai-foundry", "azure-ai"),
    display_name="Azure Foundry",
    description="Microsoft Foundry - OpenAI-compatible endpoint (user-supplied base URL)",
    signup_url="https://ai.azure.com/",
    env_vars=("AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_BASE_URL"),
    base_url="",  # per-resource; user provides at setup
    auth_type="api_key",
    supports_vision=True,
)

register_provider(azure_foundry)
