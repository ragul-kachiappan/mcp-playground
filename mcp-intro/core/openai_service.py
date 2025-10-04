from typing import Any
from core.base import ProviderType

class OpenAIProvider:
    
    _provider_type: ProviderType = None

    def __init__(self, model: str, *args, **kwargs) -> None:
        self.client = None
        self.model = model
    
    def add_user_message(self, messages: list[dict], message: dict):
        raise NotImplementedError()
    
    def add_assistant_message(self, messages: list[dict], message: dict):
        raise NotImplementedError()
    
    def text_from_message(self, message: dict) -> str:
        raise NotImplementedError()
    
    def has_tool_calls(self, response: Any) -> bool:
        """Check if the response contains tool calls"""
        raise NotImplementedError()
    
    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        temperature: float = 1.0,
        stop_sequences: list[str] = None,
        tools: list[dict] | None = None,
        thinking: bool = False,
        thinking_budget: int = 1024,
        **kwargs,
    ) -> Any:
        raise NotImplementedError()

        