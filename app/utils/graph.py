from typing import cast
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages import trim_messages as _trim_messages
from app.core.config import settings
from app.schemas.chat import Message


# LangGraph / LLM utilities
def dump_messages(messages: list[Message]) -> list[dict]:
    """
    Converts Pydantic message models into the dictionary format
    expected by OpenAI/LangChain.
    """
    return [message.model_dump() for message in messages]


def prepare_messages(messages: list[Message], llm: BaseChatModel, system_prompt: str) -> list[Message]:
    """
    Prepares the message history for the LLM context window.

    CRITICAL: This function prevents token overflow errors.
    It keeps the System Prompt + the most recent messages that fit
    within 'settings.MAX_TOKENS'.
    """
    try:
        # Intelligent trimming based on token count
        trimmed_messages = cast(list[Message], _trim_messages(
            dump_messages(messages),
            strategy="last",
            token_counter=llm,
            max_tokens=settings.MAX_TOKENS,
            start_on="human",
            include_system=False,
            allow_partial=False,
        ))
    except Exception as e:
        # Fallback if token counting fails (rare, but safety first)
        trimmed_messages = messages
    # always prepend the system prompt to enforce agent behavior
    return [Message(role="system", content=system_prompt)] + trimmed_messages


def process_llm_response(response: BaseMessage) -> BaseMessage:
    """
    Normalizes reponses from advanced models (like GPT-5 preview or claude).
    Some models return structured 'reasoning' blocks separate from content.
    This function flattens them into a single string.
    """
    if isinstance(response.content, list):
        text_parts = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        response.content = "".join(text_parts)
    return response