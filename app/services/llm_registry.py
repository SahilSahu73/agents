from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.logging import logger


# LLM Registry
class LLMRegistry:
    """
    Registry of available LLM models.
    This allows us to switch 'Brains' on the fly without changing code.
    """

    # We pre-configure models with different capabilities/costs
    LLMS: Dict[str, List[Dict[str, Any]]] = {
        "groq": [
            {
                "name": "qwen/qwen3-32b",
                "llm": ChatGroq(
                    model="qwen/qwen3-32b",
                    api_key=settings.GROQ_API_KEY,
                    max_tokens=12000,
                    reasoning_format="parsed",
                    reasoning_effort="default",
                    temperature=0.55,
                    max_retries=settings.MAX_LLM_CALL_RETRIES,
                ),
                "extra_details": """max_completion_token = 40960 \n context_window = 131072 \n
                
                Best Practices:

                    - Mode Selection: Use thinking mode (reasoning_effort="default") for complex reasoning 
                    with temperature=0.6, top_p=0.95, top_k=20, and min_p=0
                    - Non-thinking Mode: For general dialogue, use temperature=0.7, top_p=0.8, top_k=20, 
                    and min_p=0
                    - Math Problems: Include 'Please reason step by step, and put your final answer within
                    '\boxed{}' in the prompt
                    - Multiple-Choice: Add the following JSON structure to the prompt to standardize 
                    responses: "Please show your choice in the answer field with only the choice letter, 
                    e.g., "answer": "C"."
                    - History Management: In multi-turn conversations, only include final outputs without 
                    thinking content
                    - Reasoning format: Set reasoning_format to hidden to only return the final answer, 
                    or parsed to include the reasoning in a separate field
                """
            },
            {
                "name": "moonshotai/kimi-k2-instruct-0905",
                "llm": ChatGroq(
                    model="moonshotai/kimi-k2-instruct-0905",
                    api_key=settings.GROQ_API_KEY,
                    max_tokens=8000,
                    max_retries=settings.MAX_LLM_CALL_RETRIES,
                    temperature=settings.DEFAULT_LLM_TEMPERATURE,
                ),
                "extra_details": """

                context_window = 262144
                Max_completion_token = 16384
                 
                Best Practices:
                 - For frontend development, specify the framework (React, Vue, Angular) and provide context 
                 about existing codebase structure for consistent code generation.
                 - When building agents, leverage the improved scaffold integration by clearly defining agent 
                 roles, tools, and interaction patterns upfront.
                 - Utilize enhanced tool calling capabilities by providing comprehensive tool schemas with 
                 examples and error handling patterns.
                 - Structure complex coding tasks into modular components to take advantage of the model's 
                 improved full-stack development proficiency.
                 - Use the full 256K context window for maintaining codebase context across multiple files and 
                 maintaining development workflow continuity.
                
                """
            },
            {
                "name": "openai/gpt-oss-120b",
                "llm": ChatGroq(
                    model="openai/gpt-oss-120b",
                    api_key=settings.GROQ_API_KEY,
                    max_tokens=18000,
                    max_retries=settings.MAX_LLM_CALL_RETRIES,
                    temperature=0.6,
                    reasoning_effort="low",
                    reasoning_format="parsed"
                ),
                "extra_details": """
                
                context_window = 131072
                max_completion_token = 65536
                """
            },
            {
                "name": "openai/gpt-oss-20b",
                "llm": ChatGroq(
                    model="openai/gpt-oss-20b",
                    api_key=settings.GROQ_API_KEY,
                    max_tokens=22000,
                    max_retries=settings.MAX_LLM_CALL_RETRIES,
                    temperature=0.55,
                    reasoning_effort="medium",
                    reasoning_format="parsed"
                ),
                "extra_details": """
                
                context_window = 131072
                max_completion_token = 65536
                """
            },
        ],
        "openai": [
            {
                "name": "gpt-5-mini",
                "llm": ChatOpenAI(
                    model="gpt-5-mini",
                    api_key=settings.OPENAI_API_KEY,
                    max_completion_tokens=settings.MAX_TOKENS,
                    reasoning={"effort": "low"}
                ),
                "extra_details": "",
            },
            {
                "name": "gpt-4o",
                "llm": ChatOpenAI(
                    model="gpt-4o",
                    temperature=settings.DEFAULT_LLM_TEMPERATURE,
                    api_key=settings.OPENAI_API_KEY,
                    max_completion_tokens=settings.MAX_TOKENS,
                ),
                "extra_details": "",
            },
            {
                "name": "gpt-4o-mini",
                "llm": ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=settings.DEFAULT_LLM_TEMPERATURE,
                    api_key=settings.OPENAI_API_KEY,
                ),
                "extra_details": "",
            },
        ],
        "gemini": [],
    }

    @classmethod
    def get(cls, llm_provider: str, model_name: str, **kwargs) -> Optional[BaseChatModel | Runnable]:
        """Retrieve a specific model instance by name."""

        if llm_provider.lower() not in cls.LLMS.keys():
            logger.error("Invalid provider", provider=llm_provider)
            raise ValueError(f"Invalid Provider: {llm_provider}")

        model_entry = None
        for entry in cls.LLMS[llm_provider]:
            if entry["name"] == model_name:
                model_entry = entry
                break
        
        if not model_entry:
            available_models = cls.get_all_names(llm_provider)
            raise ValueError(
                f"model '{model_name}' not found in registry. available models: {', '.join(available_models)}"
            )
        
        logger.debug("using_the_provided_llm_instance",
                     model_name=model_name)
        # return the default instance
        return model_entry["llm"]
    
    @classmethod
    def get_all_names(cls, llm_provider: Optional[str] = None) -> List[str]:
        """
        If LLM_provider is passed: returns names for that provider.
        If None: return fully quantified names like "openai/gpt-4o".
        """
        if llm_provider:
            llm_provider = llm_provider.lower()
            return [e["name"] for e in cls.LLMS.get(llm_provider, [])]
        
        out: List[str] = []
        for provider, entries in cls.LLMS.items():
            for e in entries:
                out.append(f"{provider}/{e["names"]}")
        return out
    
    @classmethod
    def get_model_at_index(cls, index: int) -> Dict[str, Any]:
        """
        Get model entry at specific index.

        Args;
            index: Index of the model in the LLMs List

        Returns:
            Model entry dict
        """
        if 0 <= index < len(cls.get_all_names(settings.DEFAULT_LLM_PROVIDER)):
            return cls.LLMS[settings.DEFAULT_LLM_PROVIDER][index]
        return cls.LLMS["groq"][0]