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
        return provider_class(model=model, api_key=api_key)
    

class LLMClient:
    """Unified LLM client that can work with any provider"""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
    
    def add_user_message(self, messages: list[dict], message: str | Any) -> None:
        return self.provider.add_user_message(messages, message)

    def add_assistant_message(self, messages: list[dict], message: str | Any) -> None:
        return self.provider.add_assistant_message(messages, message)
    
    def text_from_message(self, message: Any) -> str:
        return self.provider.text_from_message(message)
    
    def chat(self, **kwargs) -> Any:
        return self.provider.chat(**kwargs)

    