"""
LLM client supporting both OpenAI (cloud) and Ollama (local).
"""
import logging
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config.settings import Settings as settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Manages LLM connection with automatic selection between OpenAI and Ollama.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model name (defaults to settings)
            temperature: Sampling temperature (defaults to settings)
            max_retries: Number of retry attempts
        """
        self.use_openai = settings.USE_OPENAI
        self.temperature = temperature or 0.7
        self.max_retries = max_retries
        
        if self.use_openai:
            self.model = model or settings.OPENAI_MODEL
            self.llm = self._create_openai_llm()
            logger.info(f"Initialized OpenAI LLM: {self.model}")
        else:
            self.model = model or settings.OLLAMA_MODEL
            self.llm = self._create_ollama_llm()
            logger.info(f"Initialized Ollama LLM: {self.model}")
    
    def _create_openai_llm(self) -> ChatOpenAI:
        """Create OpenAI LLM instance."""
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=SecretStr(settings.OPENAI_API_KEY),
            max_retries=self.max_retries
        )
    
    def _create_ollama_llm(self) -> ChatOllama:
        """Create Ollama LLM instance."""
        return ChatOllama(
            model=self.model,
            temperature=self.temperature,
            num_predict=1024,
            num_ctx=settings.OLLAMA_NUM_CTX,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
            base_url=settings.OLLAMA_BASE_URL,
            validate_model_on_init=not self.use_openai  # Skip validation for OpenAI
        )
    
    def invoke(self, prompt: str) -> Optional[str]:
        """Invoke the LLM with error handling."""
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                content = response.content
                if isinstance(content, list):
                    return str(content)
                return content
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            return None
    
    def stream(self, prompt: str):
        """Stream responses from the LLM."""
        try:
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception as e:
            logger.error(f"Error during LLM streaming: {e}")
            yield f"[Error: {str(e)}]"
    
    def with_structured_output(self, schema):
        """Create LLM instance that returns structured output."""
        return self.llm.with_structured_output(schema)
    
    def get_llm(self):
        """Get the underlying LLM instance."""
        return self.llm


def create_llm_client(**kwargs) -> LLMClient:
    """Factory function to create an LLM client."""
    return LLMClient(**kwargs)