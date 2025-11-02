"""LiteLLM client for unified access to 100+ LLM providers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import asyncio
import logging

from ..llm import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm import completion, acompletion, Router

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("LiteLLM not installed. Install with: pip install litellm")


@dataclass
class LiteLLMConfig:
    """Configuration for LiteLLM client."""

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 0.9
    timeout: int = 60
    max_retries: int = 3
    fallbacks: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Provider-specific settings
    azure_deployment: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_base: Optional[str] = None

    # Cost tracking
    track_cost: bool = True
    max_budget: Optional[float] = None

    # Debugging
    verbose: bool = False


class LiteLLMClient(LLMClient):
    """
    Production LLM client using LiteLLM for unified access to multiple providers.

    Supports:
    - OpenAI (GPT-3.5, GPT-4, etc.)
    - Anthropic (Claude-2, Claude-3, etc.)
    - Google (Gemini, PaLM)
    - Cohere
    - Azure OpenAI
    - AWS Bedrock
    - 100+ other providers

    Example:
        >>> client = LiteLLMClient(model="gpt-4")
        >>> response = client.complete("What is the capital of France?")

        >>> # Using Claude
        >>> client = LiteLLMClient(model="claude-3-sonnet-20240229")

        >>> # Using with fallbacks
        >>> client = LiteLLMClient(
        ...     model="gpt-4",
        ...     fallbacks=["claude-3-sonnet-20240229", "gpt-3.5-turbo"]
        ... )
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        fallbacks: Optional[List[str]] = None,
        config: Optional[LiteLLMConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LiteLLM client.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet-20240229")
            api_key: API key for the provider (can also use environment variables)
            api_base: Base URL for API (for custom endpoints)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            fallbacks: List of fallback models if primary fails
            config: Complete configuration object (overrides other params)
            **kwargs: Additional provider-specific parameters
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Install with: pip install litellm"
            )

        super().__init__(model=model)

        # Use provided config or create from parameters
        if config:
            self.config = config
        else:
            self.config = LiteLLMConfig(
                model=model,
                api_key=api_key,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                fallbacks=fallbacks,
                **kwargs,
            )

        # Set up API keys from environment if not provided
        self._setup_api_keys()

        # Configure LiteLLM settings
        if self.config.verbose:
            litellm.set_verbose = True

        # Set up router if fallbacks are configured
        self.router = None
        if self.config.fallbacks:
            self._setup_router()

    def _setup_api_keys(self) -> None:
        """Set up API keys from config or environment variables."""
        if not self.config.api_key:
            # Try to get API key from environment based on model provider
            if (
                "gpt" in self.config.model.lower()
                or "openai" in self.config.model.lower()
            ):
                self.config.api_key = os.getenv("OPENAI_API_KEY")
            elif (
                "claude" in self.config.model.lower()
                or "anthropic" in self.config.model.lower()
            ):
                self.config.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif "cohere" in self.config.model.lower():
                self.config.api_key = os.getenv("COHERE_API_KEY")
            elif "gemini" in self.config.model.lower():
                self.config.api_key = os.getenv("GOOGLE_API_KEY")

    def _setup_router(self) -> None:
        """Set up router for load balancing and fallbacks."""
        model_list = []

        # Add primary model
        model_list.append(
            {
                "model_name": self.config.model,
                "litellm_params": {
                    "model": self.config.model,
                    "api_key": self.config.api_key,
                    "api_base": self.config.api_base,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            }
        )

        # Add fallback models
        for fallback_model in self.config.fallbacks or []:
            model_list.append(
                {
                    "model_name": fallback_model,
                    "litellm_params": {
                        "model": fallback_model,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    },
                }
            )

        self.router = Router(
            model_list=model_list,
            fallbacks=(
                [{self.config.model: self.config.fallbacks}]
                if self.config.fallbacks
                else None
            ),
            num_retries=self.config.max_retries,
            timeout=self.config.timeout,
        )

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Generate completion for the given prompt.

        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters to pass to the model

        Returns:
            LLMResponse containing the generated text and metadata
        """
        # Prepare messages in chat format (most models now use this)
        messages = [{"role": "user", "content": prompt}]

        # Merge config with runtime kwargs
        call_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "timeout": kwargs.get("timeout", self.config.timeout),
            "num_retries": kwargs.get("num_retries", self.config.max_retries),
        }

        # Add API key if available
        if self.config.api_key:
            call_params["api_key"] = self.config.api_key
        if self.config.api_base:
            call_params["api_base"] = self.config.api_base

        # Filter out ACE-specific parameters and add remaining kwargs
        ace_specific_params = {"refinement_round", "max_refinement_rounds"}
        call_params.update(
            {
                k: v
                for k, v in kwargs.items()
                if k not in call_params and k not in ace_specific_params
            }
        )

        try:
            # Use router if available, otherwise direct completion
            if self.router:
                response = self.router.completion(**call_params)
            else:
                response = completion(**call_params)

            # Extract text from response
            text = response.choices[0].message.content

            # Build metadata
            metadata = {
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
                "cost": (
                    response._hidden_params.get("response_cost", None)
                    if hasattr(response, "_hidden_params")
                    else None
                ),
                "provider": self._get_provider_from_model(response.model),
            }

            return LLMResponse(text=text, raw=metadata)

        except Exception as e:
            logger.error(f"Error in LiteLLM completion: {e}")
            raise

    async def acomplete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Async version of complete.

        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters to pass to the model

        Returns:
            LLMResponse containing the generated text and metadata
        """
        # Prepare messages in chat format
        messages = [{"role": "user", "content": prompt}]

        # Merge config with runtime kwargs
        call_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "timeout": kwargs.get("timeout", self.config.timeout),
            "num_retries": kwargs.get("num_retries", self.config.max_retries),
        }

        # Add API key if available
        if self.config.api_key:
            call_params["api_key"] = self.config.api_key
        if self.config.api_base:
            call_params["api_base"] = self.config.api_base

        # Filter out ACE-specific parameters and add remaining kwargs
        ace_specific_params = {"refinement_round", "max_refinement_rounds"}
        call_params.update(
            {
                k: v
                for k, v in kwargs.items()
                if k not in call_params and k not in ace_specific_params
            }
        )

        try:
            # Use router if available, otherwise direct completion
            if self.router:
                response = await self.router.acompletion(**call_params)
            else:
                response = await acompletion(**call_params)

            # Extract text from response
            text = response.choices[0].message.content

            # Build metadata
            metadata = {
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
                "cost": (
                    response._hidden_params.get("response_cost", None)
                    if hasattr(response, "_hidden_params")
                    else None
                ),
                "provider": self._get_provider_from_model(response.model),
            }

            return LLMResponse(text=text, raw=metadata)

        except Exception as e:
            logger.error(f"Error in LiteLLM async completion: {e}")
            raise

    def complete_with_stream(self, prompt: str, **kwargs: Any):
        """
        Generate completion with streaming support.

        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters to pass to the model

        Yields:
            Chunks of generated text
        """
        messages = [{"role": "user", "content": prompt}]

        call_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        if self.config.api_key:
            call_params["api_key"] = self.config.api_key

        try:
            response = completion(**call_params)

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in LiteLLM streaming: {e}")
            raise

    def _get_provider_from_model(self, model: str) -> str:
        """Infer provider from model name."""
        model_lower = model.lower()

        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "palm" in model_lower:
            return "google"
        elif "command" in model_lower or "cohere" in model_lower:
            return "cohere"
        elif "llama" in model_lower or "mistral" in model_lower:
            return "meta"
        else:
            return "unknown"

    @classmethod
    def list_models(cls) -> List[str]:
        """List all supported models."""
        if not LITELLM_AVAILABLE:
            return []

        # This returns a list of common models
        # In practice, LiteLLM supports 100+ models
        return [
            # OpenAI
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            # Anthropic
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2",
            # Google
            "gemini-pro",
            "gemini-pro-vision",
            "palm-2",
            # Cohere
            "command",
            "command-light",
            "command-nightly",
            # Meta
            "llama-2-70b",
            "llama-2-13b",
            "llama-2-7b",
            # Mistral
            "mistral-7b",
            "mistral-medium",
            "mixtral-8x7b",
            # Note: Many more models are supported
            # See: https://docs.litellm.ai/docs/providers
        ]
