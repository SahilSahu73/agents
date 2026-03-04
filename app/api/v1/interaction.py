"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse

from app.api.v1.auth import get_current_session
from app.core.system.config import settings
from app.core.langgraph.graph import LangGraphAgents
from app.core.system.limiter import limiter
from app.core.system.logging import logger
from app.core.system.telemetry import llm_stream_duration_seconds
from app.models.session import Session
from app.services.llm_registry import LLMRegistry
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ModelsResponse,
    Message,
    StreamResponse,
)

router = APIRouter()
agent = LangGraphAgents()


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    models: list[ModelInfo] = []
    for provider, entries in LLMRegistry.LLMS.items():
        for entry in entries:
            models.append(
                ModelInfo(
                    provider=provider,
                    name=entry["name"],
                    extra_details=str(entry.get("extra_details", "")),
                )
            )
    return ModelsResponse(models=models)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph.
    Standard Request/Response chat Endpoint.
    Executes the full LangGraph workflow and returns the final state.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        ChatResponse: The processed chat response.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "chat_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        # Delegating execution to our LangGraph Agent
        # session.id becomes the "thread_id" for graph persistence
        result = await agent.get_response(
            chat_request.messages,
            session.id,
            user_id=str(session.user_id),
            model_provider=chat_request.model_provider,
            model_name=chat_request.model_name,
        )
        if result is None:
            logger.error("No_response_generated_from_agent", session_id=session.id)
            raise Exception

        logger.info("chat_request_processed", session_id=session.id)
        return ChatResponse(messages=result)
    
    except Exception as e:
        logger.error("chat_request_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph with streaming response.
    Streaming Chat Endpoint using Server-Sent Events (SSE).
    Allows the UI to display text character-by-character as it generates.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        StreamingResponse: A streaming response of the chat completion.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "stream_chat_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        async def event_generator():
            """Generate streaming events.
            Internal generator that yields SSE formatted chunks.

            Yields:
                str: Server-sent events in JSON format.

            Raises:
                Exception: If there's an error during streaming.
            """
            try:
                full_response = ""

                current_llm = agent.llm_service.get_llm()
                active_model_name = (
                    current_llm.name
                    if current_llm and hasattr(current_llm, "name")
                    else settings.DEFAULT_LLM_MODEL
                )
                requested_model = chat_request.model_name or active_model_name
                requested_provider = chat_request.model_provider or settings.DEFAULT_LLM_PROVIDER

                start_response = StreamResponse(
                    content="",
                    done=False,
                    event="start",
                    model_provider=requested_provider,
                    model_name=requested_model,
                )
                yield f"data: {json.dumps(start_response.model_dump())}\n\n"

                with llm_stream_duration_seconds.labels(model=active_model_name).time():
                    async for chunk in agent.get_stream_response(
                        chat_request.messages,
                        session.id,
                        user_id=str(session.user_id),
                        model_provider=chat_request.model_provider,
                        model_name=chat_request.model_name,
                    ):
                        full_response += chunk
                        response = StreamResponse(
                            content=chunk,
                            done=False,
                            event="chunk",
                            model_provider=requested_provider,
                            model_name=requested_model,
                        )
                        yield f"data: {json.dumps(response.model_dump())}\n\n"

                # Send final message indicating completion
                current_llm = agent.llm_service.get_llm()
                final_model_name = (
                    current_llm.name
                    if current_llm and hasattr(current_llm, "name")
                    else requested_model
                )
                final_response = StreamResponse(
                    content="",
                    done=True,
                    event="done",
                    model_provider=requested_provider,
                    model_name=final_model_name,
                )
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"

            except Exception as e:
                logger.error(
                    "stream_chat_request_failed",
                    session_id=session.id,
                    error=str(e),
                    exc_info=True,
                )
                error_response = StreamResponse(content=str(e), done=True, event="error")
                yield f"data: {json.dumps(error_response.model_dump())}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(
            "stream_chat_request_failed",
            session_id=session.id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def get_session_messages(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Get all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        ChatResponse: All messages in the session.

    Raises:
        HTTPException: If there's an error retrieving the messages.
    """
    try:
        messages = await agent.get_chat_history(session.id)
        if messages is None:
            raise Exception
        
        return ChatResponse(messages=messages)
    except Exception as e:
        logger.error("get_messages_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_history(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Clear all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        dict: A message indicating the chat history was cleared.
    """
    try:
        await agent.clear_chat_history(session.id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error("clear_chat_history_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
