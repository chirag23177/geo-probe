"""Anthropic provider. Measures parametric memory -- no retrieval, no citations."""

from __future__ import annotations

import time

import anthropic

from .base import (
    Provider,
    ProviderError,
    ProviderResponse,
    RetryableProviderError,
    require_env,
)

MAX_TOKENS = 1024


class AnthropicProvider(Provider):
    id = "anthropic"

    def __init__(self, model: str) -> None:
        self.model = model
        # max_retries=0: we own the retry loop so that backoff and the final
        # failure record are the same code path for both providers.
        self._client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"), max_retries=0)

    def complete(self, prompt: str, temperature: float | None) -> ProviderResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            # Thinking off: we are measuring the answer a buyer sees, and leaving
            # adaptive thinking on would multiply cost per run without changing
            # what is being measured.
            "thinking": {"type": "disabled"},
        }
        if temperature is not None:
            # 1.0 is the API default, so this is accepted on models that reject
            # non-default sampling parameters. Deliberately NOT lowered: we are
            # sampling the real distribution, not collapsing it.
            kwargs["temperature"] = temperature

        # No `seed` parameter is set anywhere, on any provider, on purpose.
        # Seeding would make repeated reps identical and destroy the very
        # variance this tool exists to measure.

        started = time.perf_counter()
        try:
            resp = self._client.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            raise RetryableProviderError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise RetryableProviderError(str(exc)) from exc
            raise ProviderError(f"{exc.status_code}: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        text = "".join(block.text for block in resp.content if block.type == "text")
        return ProviderResponse(
            text=text,
            # The version string the API actually returned, not the alias we asked for.
            model_returned=resp.model,
            latency_ms=latency_ms,
            citation_urls=[],
        )
