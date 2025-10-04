from typing import Protocol, Any
from enum import Enum

class ProviderType(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"

class LLMProvider(Protocol):
    """Protocol defining the LLM interface for providers"""
    _provider_type: ProviderType = None

    def add_user_message(self, messages: list[dict], message: dict | str) -> None:
        """Add a user message to the messages list"""
        ...
    
    def add_assistant_message(self, messages: list[dict], message: dict) -> None:
        """Add an assistant message to the messages list"""
        ...
    
    def text_from_message(self, messages: Any) -> str:
        """Extract text content from a message object"""
        ...
    
    def has_tool_calls(self, response: Any) -> bool:
        """Check if the response contains tool calls"""
        ...
    
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
        """Send chat request and return response"""
        ...
    

