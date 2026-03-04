from functools import lru_cache
from typing import Callable, Optional, cast
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages import trim_messages as _trim_messages
from app.core.system.config import settings
from app.schemas.chat import Message
from app.core.system.logging import logger


# LangGraph / LLM utilities
def dump_messages(messages: list[Message]) -> list[dict]:
    """
    Converts Pydantic message models into the dictionary format
    expected by OpenAI/LangChain.
    """
    return [message.model_dump() for message in messages]


@lru_cache(maxsize=8)
def _load_hf_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _resolve_hf_model_id(model_provider: Optional[str], model_name: Optional[str]) -> Optional[str]:
    if not model_provider or not model_name:
        return None

    model_key = (model_provider.lower(), model_name.lower())
    model_map = {
        ("groq", "qwen/qwen3-32b"): "Qwen/Qwen3-32B",
    }
    return model_map.get(model_key)


def _build_token_counter(
    llm: BaseChatModel, model_provider: Optional[str], model_name: Optional[str]
) -> BaseChatModel | Callable[[list | str], int]:
    hf_model_id = _resolve_hf_model_id(model_provider, model_name)
    if not hf_model_id:
        return llm

    try:
        tokenizer = _load_hf_tokenizer(hf_model_id)
    except Exception as e:
        logger.warning(
            "model_specific_tokenizer_unavailable",
            model_provider=model_provider,
            model_name=model_name,
            tokenizer_model=hf_model_id,
            error=str(e),
        )
        return llm

    def token_counter(items: list | str) -> int:
        if isinstance(items, str):
            return len(tokenizer.encode(items, add_special_tokens=False))

        total_tokens = 0
        for item in items:
            if isinstance(item, dict):
                role = str(item.get("role", ""))
                content = str(item.get("content", ""))
                text = f"{role}: {content}"
            else:
                role = str(getattr(item, "type", getattr(item, "role", "")))
                content = getattr(item, "content", "")
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            parts.append(str(block["text"]))
                        elif isinstance(block, str):
                            parts.append(block)
                    content = " ".join(parts)
                text = f"{role}: {str(content)}"

            total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
        return total_tokens

    logger.debug(
        "using_model_specific_tokenizer",
        model_provider=model_provider,
        model_name=model_name,
        tokenizer_model=hf_model_id,
    )
    return token_counter


def prepare_messages(
    messages: list[Message],
    llm: BaseChatModel,
    system_prompt: str,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> list[Message]:
    """
    Prepares the message history for the LLM context window.

    CRITICAL: This function prevents token overflow errors.
    It keeps the System Prompt + the most recent messages that fit
    within 'settings.MAX_TOKENS'.
    """
    token_counter = _build_token_counter(llm, model_provider, model_name)
    try:
        # Intelligent trimming based on token count
        trimmed_messages = cast(list[Message], _trim_messages(
            dump_messages(messages),
            strategy="last",
            token_counter=token_counter,
            max_tokens=settings.MAX_TOKENS,
            start_on="human",
            include_system=False,
            allow_partial=False,
        ))
    except ImportError as e:
        logger.warning(
            "tokenizer_import_failed_skipping_trim",
            error=str(e),
            message_count=len(messages),
            model_provider=model_provider,
            model_name=model_name,
        )
        trimmed_messages = messages
    except ValueError as e:
        # Handle unrecognized content blocks (e.g., reasoning blocks from GPT-5)
        if "Unrecognized content block type" in str(e):
            logger.warning(
                "token_counting_failed_skipping_trim",
                error=str(e),
                message_count=len(messages),
            )
            # Skip trimming and return all messages
            trimmed_messages = messages
        else:
            raise
    # always prepend the system prompt to enforce agent behavior
    return [Message(role="system", content=system_prompt)] + trimmed_messages


def process_llm_response(response: BaseMessage) -> BaseMessage:
    """Process LLM response to handle structured content blocks (e.g., from GPT-5 models).

    GPT-5 models return content as a list of blocks like:
    [
        {'id': '...', 'summary': [], 'type': 'reasoning'},
        {'type': 'text', 'text': 'actual response'}
    ]

    This function extracts the actual text content from such structures.

    Args:
        response: The raw response from the LLM

    Returns:
        BaseMessage with processed content
    """
    if isinstance(response.content, list):
        # Extract text from content blocks
        text_parts = []
        for block in response.content:
            if isinstance(block, dict):
                # Handle text blocks
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(block["text"])
                # Log reasoning blocks for debugging
                elif block.get("type") == "reasoning":
                    logger.debug(
                        "reasoning_block_received",
                        reasoning_id=block.get("id"),
                        has_summary=bool(block.get("summary")),
                    )
            elif isinstance(block, str):
                text_parts.append(block)

        # Join all text parts
        response.content = "".join(text_parts)
        logger.debug(
            "processed_structured_content",
            block_count=len(response.content) if isinstance(response.content, list) else 1,
            extracted_length=len(response.content) if isinstance(response.content, str) else 0,
        )

    return response
