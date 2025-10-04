import json
from typing import Optional, Literal, List
from mcp.types import CallToolResult, Tool, TextContent
from mcp_client import MCPClient
from anthropic.types import Message, ToolResultBlockParam
from core.base import LLMProvider, ProviderType


class ToolManager:
    @staticmethod
    def ollama_tool_schema_dict(mcp_tool: Tool) -> dict:
        """Convert MCP-Claude tool schema object to Ollama tool schema dict"""
        mcp_properties = mcp_tool.inputSchema.get('properties')
        ollama_properties = {}
        
        for prop_name, prop_def in mcp_properties.items():
            ollama_properties[prop_name] = {
                'type': prop_def.get('type'),
                'description': prop_def.get('description', prop_def.get('title', ''))
            }
            
            # Handle additional properties like enum, items, etc.
            if 'enum' in prop_def:
                ollama_properties[prop_name]['enum'] = prop_def['enum']
            if 'items' in prop_def:
                ollama_properties[prop_name]['items'] = prop_def['items']
        return {
        'type': 'function',
        'function': {
            'name': mcp_tool.name,
            'description': mcp_tool.description,
            'parameters': {
                'type': 'object',
                'properties': ollama_properties,
                'required': mcp_tool.inputSchema.get('required', [])
            }
        }
    }

    @staticmethod
    def mcp_tool_schema_dict(mcp_tool: Tool) -> dict:
        """Convert MCP-Claude tool schema object to dict"""
        return {
                    "name": mcp_tool.name,
                    "description": mcp_tool.description,
                    "input_schema": mcp_tool.inputSchema,
                }


    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient], llm_service: LLMProvider ) -> list[dict]:
        """Gets all tools from the provided clients."""
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            for t in tool_models:
                match llm_service._provider_type: 
                    case ProviderType.OLLAMA:
                        tool_dict = ToolManager.ollama_tool_schema_dict(t)
                    case ProviderType.CLAUDE:
                        tool_dict = ToolManager.mcp_tool_schema_dict(t)
                    case _:
                        raise NotImplementedError()
                tools.append(tool_dict)
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """Finds the first client that has the specified tool."""
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    def _build_tool_result_part(
        cls,
        tool_use_id: str,
        text: str,
        status: Literal["success"] | Literal["error"],
    ) -> ToolResultBlockParam:
        """Builds a tool result part dictionary."""
        return {
            "tool_use_id": tool_use_id,
            "type": "tool_result",
            "content": text,
            "is_error": status == "error",
        }

    # TODO need to make it generic to handle all llm providers
    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], message: Message
    ) -> List[ToolResultBlockParam]:
        """Executes a list of tool requests against the provided clients."""
        tool_requests = [
            block for block in message.content if block.type == "tool_use"
        ]
        tool_result_blocks: list[ToolResultBlockParam] = []
        for tool_request in tool_requests:
            tool_use_id = tool_request.id
            tool_name = tool_request.name
            tool_input = tool_request.input

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id, "Could not find that tool", "error"
                )
                tool_result_blocks.append(tool_result_part)
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )
                items = []
                if tool_output:
                    items = tool_output.content
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                content_json = json.dumps(content_list)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    content_json,
                    "error"
                    if tool_output and tool_output.isError
                    else "success",
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_part = cls._build_tool_result_part(
                    tool_use_id,
                    json.dumps({"error": error_message}),
                    "error"
                    if tool_output and tool_output.isError
                    else "success",
                )

            tool_result_blocks.append(tool_result_part)
        return tool_result_blocks
