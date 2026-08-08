"""Base interface for provider format converters.

All converters transform between OpenAI format (universal) and provider-specific formats.
"""

from abc import ABC, abstractmethod


class ProviderConverter(ABC):
    """Abstract base class for provider-specific format converters.
    
    Each provider converter must implement three methods:
    1. openai_to_provider: Convert OpenAI request to provider format
    2. provider_to_openai: Convert provider response to OpenAI format
    3. provider_sse_to_openai: Convert provider SSE chunk to OpenAI streaming format
    
    All implementations should use @staticmethod since converters are stateless.
    """
    
    @staticmethod
    @abstractmethod
    def openai_to_provider(body: dict, model_id: str) -> dict:
        """Convert OpenAI format request to provider-specific format.
        
        Args:
            body: OpenAI format request body (messages, tools, etc.)
            model_id: Target model identifier
            
        Returns:
            Provider-specific request body
        """
        pass
    
    @staticmethod
    @abstractmethod
    def provider_to_openai(response: dict) -> dict:
        """Convert provider response to OpenAI format.
        
        Args:
            response: Provider-specific response body
            
        Returns:
            OpenAI format response
        """
        pass
    
    @staticmethod
    @abstractmethod
    def provider_sse_to_openai(chunk: dict) -> dict | None:
        """Convert provider SSE chunk to OpenAI streaming format.
        
        Args:
            chunk: Provider-specific SSE data chunk
            
        Returns:
            OpenAI format streaming chunk, or None if chunk should be skipped
        """
        pass

