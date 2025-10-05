"""
Ollama LLM client with error handling and retry logic.

This module provides a robust interface to locally-deployed LLMs via Ollama,
with proper initialization, validation, and error recovery mechanisms.
"""
import logging
from typing import Optional
from langchain_ollama import ChatOllama

from config.settings import Settings as settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Manages connection to local Ollama LLM with validation and error handling.
    
    The client handles common issues like connection failures, model availability,
    and provides retry logic for transient errors.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        max_retries: int = 3
    ):
        """
        Initialize Ollama client with configuration from settings.
        
        Args:
            model: Model name (defaults to settings)
            temperature: Sampling temperature (defaults to settings)
            num_ctx: Context window size (defaults to settings)
            max_retries: Number of retry attempts for initialization
        """
        self.model = model or settings.OLLAMA_MODEL
        self.temperature = temperature or settings.OLLAMA_TEMPERATURE
        self.num_ctx = num_ctx or settings.OLLAMA_NUM_CTX
        self.max_retries = max_retries
        
        # Initialize the LLM with retries
        self.llm = self._create_llm_with_retry()
    
    def _create_llm_with_retry(self) -> ChatOllama:
        """
        Create LLM client with retry logic for initialization failures.
        
        This handles cases where Ollama might not be fully started or
        the model needs to be pulled from the registry.
        
        Returns:
            Initialized ChatOllama instance
            
        Raises:
            ConnectionError: If unable to connect after all retries
            Exception: For other initialization failures
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Initializing Ollama (attempt {attempt + 1}/{self.max_retries}): "
                    f"model={self.model}, temp={self.temperature}"
                )
                
                llm = ChatOllama(
                    model=self.model,
                    temperature=self.temperature,
                    num_predict=1024,  # Max tokens per response
                    num_ctx=self.num_ctx,
                    top_k=40,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    keep_alive=settings.OLLAMA_KEEP_ALIVE,
                    base_url=settings.OLLAMA_BASE_URL,
                    validate_model_on_init=True  # Fail early if model unavailable
                )
                
                logger.info(f"Successfully initialized Ollama with {self.model}")
                return llm
                
            except ConnectionError as e:
                last_error = e
                logger.warning(
                    f"Connection failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                
                if attempt == self.max_retries - 1:
                    logger.error(
                        "Failed to connect to Ollama. "
                        "Ensure Ollama is running: ollama serve"
                    )
                    raise ConnectionError(
                        f"Could not connect to Ollama at {settings.OLLAMA_BASE_URL}. "
                        f"Make sure Ollama is running and the URL is correct."
                    ) from e
                    
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error initializing LLM: {e}")
                
                if "model not found" in str(e).lower():
                    logger.info(
                        f"Model {self.model} not found. "
                        f"Pull it with: ollama pull {self.model}"
                    )
                
                if attempt == self.max_retries - 1:
                    raise
        
        # Should not reach here, but just in case
        raise Exception(f"Failed to initialize LLM after {self.max_retries} attempts")
    
    def invoke(self, prompt: str) -> Optional[str]:
        """
        Invoke the LLM with error handling.
        
        This wraps the standard invoke method with error handling and logging
        to gracefully handle failures during inference.
        
        Args:
            prompt: The input prompt for the LLM
            
        Returns:
            The LLM response content, or None if an error occurred
        """
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
        """
        Stream responses from the LLM token by token.
        
        This enables progressive display of responses in the UI for better
        user experience with long-form content.
        
        Args:
            prompt: The input prompt for the LLM
            
        Yields:
            Response chunks as they are generated
        """
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
        """
        Create a version of the LLM that returns structured output.
        
        This is essential for extracting specific fields from LLM responses
        in a reliable, type-safe manner using Pydantic models.
        
        Args:
            schema: A Pydantic model class defining the output structure
            
        Returns:
            LLM instance configured to return structured output
        """
        return self.llm.with_structured_output(schema)
    
    def get_llm(self) -> ChatOllama:
        """
        Get the underlying ChatOllama instance.
        
        Use this when you need direct access to the LLM for advanced operations
        like binding tools or custom configuration.
        
        Returns:
            The ChatOllama instance
        """
        return self.llm


def create_ollama_client(**kwargs) -> OllamaClient:
    """
    Factory function to create an Ollama client with settings.
    
    This is a convenience function that can be used throughout the application
    to get a properly configured LLM client.
    
    Args:
        **kwargs: Optional overrides for client configuration
        
    Returns:
        Initialized OllamaClient instance
    """
    return OllamaClient(**kwargs)