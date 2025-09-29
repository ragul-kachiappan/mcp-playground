from ollama import Client
from ollama import Message, ChatResponse

class OllamaProvider:
    def __init__(self, model: str):
        self.client = Client
        self.model = model
    
    def add_user_message(self, messages: list[dict], message: str) -> None:
        user_message = {
            "role": "user",
            "content": message,
        }
        messages.append(user_message)

    
    def add_assistant_message(self, messages: list[dict], message: str) -> None:
        assistant_message = {
            "role": "assistant",
            "content": message,
        }
        messages.append(assistant_message)
    
    def text_from_message(self, message: Message) -> str:
        return message.content
    
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
    ) -> Message:
        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences
        }

        if thinking:
            params["think"] = thinking
            # NOTE ollama doesn't have thinking budget, rather low, high, medium settings.
            
        
        if tools:
            params["tools"] = tools
        
        if system:
            # NOTE not very optimised
            messages.insert(0, {"role": "system", "content": system})
        
        response: ChatResponse = self.client.chat(**params)
        return response.message

        
    