from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
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

    def __init__(self):
        self._llm: Optional[BaseChatModel | Runnable] = None
        self._current_model_index: int = 0

        # Find index of default model in registry
        all_names = LLMRegistry.get_all_names(settings.DEFAULT_LLM_PROVIDER)

        # Initialize with the default model from settings
        try:
            self._llm = LLMRegistry.get(settings.DEFAULT_LLM_PROVIDER, settings.DEFAULT_LLM_MODEL)
            self._current_model_index = all_names.index(settings.DEFAULT_LLM_MODEL)

            logger.info("llm_service_initialised",
                        default_model=settings.DEFAULT_LLM_MODEL,
                        model_index=self._current_model_index,
                        total_models=len(all_names),
                        environment=settings.ENVIRONMENT.value)
            
        except (ValueError, Exception) as e:
            # Fallback Safety
            self._current_model_index = 0
            self._llm = LLMRegistry.LLMS[settings.DEFAULT_LLM_PROVIDER][0]["llm"]
            logger.warning("default_model_not_found_using_first",
                           requested=settings.DEFAULT_LLM_MODEL,
                           using=all_names[0] if all_names else "none",
                           error=str(e))

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
                "Switching_to_next_model",
                from_index=self._current_model_index,
                to_index=next_index,
                to_model=next_model_entry["name"],
            )
            self._current_model_index = next_index
            self._llm = next_model_entry["llm"]

            logger.info("model_switched", new_model=next_model_entry["name"], new_index=next_index)
            return True
        except Exception as e:
            logger.error("model_switch_failed", error=str(e))
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
        """
        Call the LLM with automatic retry logic.
        Internal method that executes the actual API call
        
        Args:
            messages: List of messages to send to the LLM
        Returns:
            BaseMessage response from the LLM
        Raises:
            OpenAIError: If all retries fail
        """
        if not self._llm:
            raise RuntimeError("LLM not initialized")
        
        try:
            response = await self._llm.ainvoke(messages)
            logger.debug("llm_call_successful", message_count=len(messages))
            return response
        except (RateLimitError, APITimeoutError, APIError) as e:
            logger.warning("llm_call_failed_retrying",
                           error_type=type(e).__name__,
                           error=str(e),
                           exc_info=True)
            raise
        except OpenAIError as e:
            logger.error("llm_call_failed", error_type=type(e).__name__,
                         error=str(e))
            raise
    
    async def call(self, 
                   messages: List[BaseMessage],
                   model_provider: Optional[str] = None,
                   model_name: Optional[str] = None,
                   **model_kwargs) -> BaseMessage:
        """
        Call the LLM with the specified messages and circular fallback.
        Public Interface. Wraps the retry logic with a Fallback loop.
        If 'gpt-4o' fails 3 times, we switch to 'gpt-4o-mini' and try again.

        Args:
            messages: List of messages to send to the LLM
            model_name: Optional specific model to use. If None, uses the current model.
            **model_kwargs: Optional kwargs to override default model configuration

        Returns:
            BaseMessage: response from the LLM

        Raises:
            RuntimeError: If all models fail after retries
        """
        if model_provider and model_name:
            try:
                self._llm = LLMRegistry.get(model_provider, model_name, **model_kwargs)
                # update index to match the requested model
                all_names = LLMRegistry.get_all_names(model_provider)
                try:
                    self._current_model_index = all_names.index(model_name)
                except ValueError:
                    pass
                logger.info("using_requested_model", model=model_name, has_custom_kwargs=bool(model_kwargs))
            except ValueError as e:
                logger.error("requested_model_not_found", model_name=model_name, error=str(e))
                raise


        provider = settings.DEFAULT_LLM_PROVIDER
        total_models = len(LLMRegistry.LLMS[provider])
        models_tried = 0
        starting_index = self._current_model_index
        last_error = None

        while models_tried < total_models:
            try:
                # Attempt to generate response
                response = await self._call_with_retry(messages)
                return response
            
            except Exception as e:
                last_error = e
                # If we exhausted retries for this model, log and switch
                models_tried += 1
                current_model_name = LLMRegistry.LLMS[provider][self._current_model_index]["name"]
                logger.error(
                    "llm_call_failed_after_retries",
                    model=current_model_name,
                    models_tried=models_tried,
                    total_models=total_models,
                    error=str(e),
                )

                if models_tried >= total_models:
                    logger.error("all_models_failed",
                                 models_tried=models_tried,
                                 starting_model=LLMRegistry.LLMS[provider][starting_index]["name"])
                    # We tried everything. The world is probably ending.
                    break

                # Switch to next model in circular fashion
                if not self._switch_to_next_model():
                    logger.error("failed_to_switch_to_next_model")
                    break

        raise RuntimeError(f"Failed to get response from any LLM after exhausting all options. Tried {models_tried} models. Lst error: {str(last_error)}")
    
    def get_llm(self) -> Optional[BaseChatModel | Runnable]:
       return self._llm
    
    def bind_tools(self, tools: List) -> "LLMService":
        """Bind tools to the current LLM instance"""
        if self._llm and isinstance(self._llm, BaseChatModel):
            self._llm = self._llm.bind_tools(tools)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self
    
llm_service = LLMService()