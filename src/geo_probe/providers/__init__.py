from __future__ import annotations

from ..schemas import ProviderConfig
from .anthropic import AnthropicProvider
from .base import (
    Provider,
    ProviderError,
    ProviderResponse,
    RetryableProviderError,
    with_retries,
)
from .perplexity import PerplexityProvider

__all__ = [
    "Provider",
    "ProviderError",
    "ProviderResponse",
    "RetryableProviderError",
    "with_retries",
    "AnthropicProvider",
    "PerplexityProvider",
    "build_provider",
]


def build_provider(cfg: ProviderConfig) -> Provider:
    if cfg.id == "anthropic":
        return AnthropicProvider(cfg.model)
    if cfg.id == "perplexity":
        return PerplexityProvider(cfg.model)
    raise ValueError(f"unknown provider id: {cfg.id}")
