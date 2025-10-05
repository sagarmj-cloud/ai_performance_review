# ============= src/llm/__init__.py =============
'''LLM package for Ollama client and utilities.'''

from .ollama_client import OllamaClient, create_ollama_client

__all__ = ['OllamaClient', 'create_ollama_client']