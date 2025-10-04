from typing import Any

from core.base import LLMProvider, ProviderType
from core.claude_service import ClaudeProvider
from core.gemini_service import GeminiProvider
from core.openai_service import OpenAIProvider
from core.ollama_service import OllamaProvider

class LLMFactory:
    """Factory class to create LLM provider instances"""

    _providers = {
        ProviderType.CLAUDE: ClaudeProvider,
        ProviderType.GEMINI: GeminiProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.OLLAMA: OllamaProvider,
    }

    @classmethod
    def create_provider(
        cls,
        provider_type: ProviderType,
        model: str,
        api_key: str | None = None,
    ) -> LLMProvider:
        """Create a provider instance"""
        if provider_type not in cls._providers:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        llm_provider = provider_class(model=model, api_key=api_key)
        llm_provider._provider_type = provider_type
        return llm_provider
    