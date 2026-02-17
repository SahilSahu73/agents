from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from openai import APIError, APITimeoutError, OpenAIError, RateLimitError
from tenacity import(
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import logger
from app.services.llm_registry import LLMRegistry


class LLMService:
    """
    Manages LLM calls with automatic retires and fallback logic.
    """

    def __init__(self) -> None:
        self._llm: Optional[BaseChatModel] = None
        self._current_model_index: int = 0

        # Initialize with the default model from settings
        try:
            self._llm = LLMRegistry.get(settings.DEFAULT_LLM_PROVIDER, settings.DEFAULT_LLM_MODEL)
            all_names = LLMRegistry.get_all_names(settings.DEFAULT_LLM_PROVIDER)
            self._current_model_index = all_names.index(settings.DEFAULT_LLM_MODEL)
        except ValueError:
            # Fallback Safety
            self._llm = LLMRegistry.LLMS[settings.DEFAULT_LLM_PROVIDER][0]["llm"]

    def _switch_to_next_model(self) -> bool:
        """
        Circular Fallback: Switches to the next available model in the registry.
        Returns True if Successful
        """
        try:
            provider = settings.DEFAULT_LLM_PROVIDER
            next_index = (self._current_model_index + 1) % len(LLMRegistry.LLMS[provider])
            next_model_entry = LLMRegistry.LLMS[provider][next_index]

            logger.warning(
                "Switching model fallback",
                old_index=self._current_model_index,
                new_model=next_model_entry["name"]
            )
            self._current_model_index = next_index
            self._llm = next_model_entry["llm"]
            return True
        except Exception as e:
            logger.error("model switch failed", error=str(e))
            return False
        
    # The Retry Decorator
    # If the function raises specific exceptions,
    # Tenacity will wait (exponentially) and try again
    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, 30),   # level = Warning (represented by number 30)
        reraise=True,
    )
    async def _call_with_retry(self, messages: List[BaseMessage]) -> BaseMessage:
        """Internal method that executes the actual API call"""
        if not self._llm:
            raise RuntimeError("LLM not initialized")
        return await self._llm.ainvoke(messages)
    
    async def call(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Public Interface. Wraps the retry logic with a Fallback loop.
        If 'gpt-4o' fails 3 times, we switch to 'gpt-4o-mini' and try again.
        """
        provider = settings.DEFAULT_LLM_PROVIDER
        total_models = len(LLMRegistry.LLMS[provider])
        models_tried = 0

        while models_tried < total_models:
            try:
                # Attempt to generate response
                return await self._call_with_retry(messages)
            
            except Exception as e:
                # If we exhausted retries for this model, log and switch
                models_tried += 1
                logger.error(
                    "model_failed_exhausted_retries",
                    model=LLMRegistry.LLMS[provider][self._current_model_index]["name"],
                    error=str(e)
                )

                if models_tried >= total_models:
                    # We tried everything. The world is probably ending.
                    break

                self._switch_to_next_model()
        raise RuntimeError("Failed to get response from any LLM after exhausting all options.")
    
    def get_llm(self) -> Optional[BaseChatModel]:
       return self._llm
    
    def bind_tools(self, tools: List) -> "LLMService":
        """Bind tools to the current LLM instance"""
        if self._llm:
            self._llm.bind_tools(tools)
        return self
    
llm_service = LLMService()