import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.llm_provider import LLMFactory
from core.base import ProviderType

from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()

# LLM Config
claude_model = os.getenv("CLAUDE_MODEL", "")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
PROVIDER= os.getenv("PROVIDER", "ollama")
MODEL= os.getenv("MODEL", "gemma3:12b")
API_KEY=os.getenv ("API_KEY", "")


assert PROVIDER, "Error: PROVIDER cannot be empty. Update .env"
assert MODEL, "Error: MODEL cannot be empty. Update .env"
if PROVIDER != ProviderType.OLLAMA.value:
    assert API_KEY, (
        "Error: API_KEY cannot be empty. Update .env"
    )


async def main():
    llm_service = LLMFactory.create_provider(provider_type=ProviderType(PROVIDER), model=MODEL)

    server_scripts = sys.argv[1:]
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            llm_service=llm_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
