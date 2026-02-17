import asyncio
from typing import AsyncGenerator, Optional
from urllib.parse import quote_plus
from asgiref.sync import sync_to_async

from langchain_core.messages import ToolMessage, convert_to_openai_messages
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import Command, CompiledStateGraph
from langgraph.types import StateSnapshot
from langchain_core.runnables.config import RunnableConfig

from mem0 import AsyncMemory

from psycopg_pool import AsyncConnectionPool
from app.core.config import settings, Environment
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.prompts import load_system_prompt
from app.schemas import GraphState, Message
from app.services.llm_service import llm_service
from app.utils import dump_messages, prepare_messages, process_llm_response


class LangGraphAgents:
    """
    Manages the LangGraph Workflow, LLM interactions, and Memory persistence.
    """
    def __init__(self):
        # Bind tools to the LLM service so the model knows what fucntions it can call
        self.llm_service = llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}

        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        self.memory: Optional[AsyncMemory] = None
        logger.info("Langgraph_agent_initialized", model=settings.DEFAULT_LLM_MODEL)

    async def _long_term_memory(self) -> AsyncMemory:
        """Lazy-load the mem0ai memory client with pgvector configuration."""
        if self.memory is None:
            self.memory = await AsyncMemory.from_config(
                config_dict={
                    "vector-store": {
                        "provider": "pgvector",
                        "config": {
                            "collection_name": "agent_memory",
                            "dbname": settings.POSTGRES_DB,
                            "user": settings.POSTGRES_USER,
                            "password": settings.POSTGRES_PASSWORD,
                            "host": settings.POSTGRES_HOST,
                            "port": settings.POSTGRES_PORT,
                        },
                    },
                    "llm": {
                        "provider": "groq",
                        "config": {"model": settings.DEFAULT_LLM_MODEL},
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {"model": "text-embedding-3-small"}
                    },
                }
            )
        return self.memory
    
    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """
        Establish a connection pool specifically for LangGraph checkpointers.
        """
        if self._connection_pool is None:
            connection_url = (
                "postgresql://"
                f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )
            self._connection_pool = AsyncConnectionPool(
                connection_url,
                open=False,
                max_size=settings.POSTGRES_POOL_SIZE,
                kwargs={"automatic": True}
            )
            await self._connection_pool.open()
        return self._connection_pool
    
    # # Node Logic
    # async def _chat(self, state: GraphState, config: RunnableConfig) -> Command:
    #     """
    #     The main Chat Node.
    #     1. Loads system prompt with memory context.
    #     2. Prepares messages (trimming if needed).
    #     3. Calls LLM Service.
    #     """

    #     # Load system prompt with the Long-Term Memory retrieved from previous steps
    #     SYSTEM_PROMPT = load_system_prompt(long_term_memory=state.long_term_memory)
        
    #     # Prepare context window (trimming)
    #     current_llm = self.llm_service.get_llm()
    #     if current_llm is None:
    #         raise ValueError("LLM service returned None. Unable to prepare messages.")
        
    #     messages = prepare_messages(state.messages, current_llm, SYSTEM_PROMPT)
    #     try:
    #         # Invoke LLM (with retries handled by service)
    #         response_message = await self.llm_service.call(dump_messages(messages))
    #         response_message = process_llm_response(response_message)
    #         # Determine routing: If LLM wants to use a tool, go to 'tool_call', else END.
    #         if response_message.tool_calls:
    #             goto = "tool_call"
    #         else:
    #             goto = END
    #         # Return command to update state and route
    #         return Command(update={"messages": [response_message]}, goto=goto)
            
    #     except Exception as e:
    #         logger.error("llm_call_node_failed", error=str(e))
    #         raise