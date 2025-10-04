from core.base import LLMProvider
from mcp_client import MCPClient
from core.tools import ToolManager
from anthropic.types import MessageParam


class Chat:
    def __init__(self, llm_service: LLMProvider, clients: dict[str, MCPClient]):
        self.llm_service: LLMProvider = llm_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[MessageParam] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def _handle_tool_calls(self, response) -> bool:
        """Handle tool calls if present. Returns True if tool calls were handled, False otherwise."""
        if not self.llm_service.has_tool_calls(response):
            return False
            
        print(self.llm_service.text_from_message(response))
        tool_result_parts = await ToolManager.execute_tool_requests(
            self.clients, response
        )
        
        self.llm_service.add_user_message(
            self.messages, tool_result_parts
        )
        return True

    async def run(
        self,
        query: str,
    ) -> str:
        await self._process_query(query)

        while True:
            response = self.llm_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients, self.llm_service),
            )

            self.llm_service.add_assistant_message(self.messages, response.content)

            # Handle tool calls if present, otherwise return the final response
            # TODO execute_tool_requests needs to be updated to handle ollama
            if not await self._handle_tool_calls(response):
                return self.llm_service.text_from_message(response)
