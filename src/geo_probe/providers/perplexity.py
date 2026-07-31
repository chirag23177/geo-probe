"""Perplexity provider. Measures retrieval-grounded output -- a different surface.

Raw httpx POST to /chat/completions; Perplexity has no first-party Python SDK.
"""

from __future__ import annotations

import time

import httpx

from .base import (
    Provider,
    ProviderError,
    ProviderResponse,
    RetryableProviderError,
    require_env,
)

ENDPOINT = "https://api.perplexity.ai/chat/completions"
MAX_TOKENS = 1024
TIMEOUT_S = 120.0


def _extract_citation_urls(payload: dict) -> list[str]:
    """Citations come from the response's own fields, never from parsing prose.

    Perplexity has shipped both a flat `citations` list of URLs and a
    `search_results` list of objects; accept either, prefer `search_results`.
    """
    urls: list[str] = []
    for item in payload.get("search_results") or []:
        if isinstance(item, dict) and isinstance(item.get("url"), str):
            urls.append(item["url"])
    if not urls:
        for item in payload.get("citations") or []:
            if isinstance(item, str):
                urls.append(item)
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


class PerplexityProvider(Provider):
    id = "perplexity"

    def __init__(self, model: str) -> None:
        self.model = model
        self._api_key = require_env("PERPLEXITY_API_KEY")
        self._client = httpx.Client(timeout=TIMEOUT_S)

    def complete(self, prompt: str, temperature: float | None) -> ProviderResponse:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
        }
        if temperature is not None:
            body["temperature"] = temperature
        # Provider default temperature when unset; no seed (see anthropic.py).

        started = time.perf_counter()
        try:
            resp = self._client.post(
                ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        except httpx.HTTPError as exc:
            raise RetryableProviderError(str(exc)) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableProviderError(f"{resp.status_code}: {resp.text[:300]}")
        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text[:300]}")

        try:
            payload = resp.json()
            text = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {resp.text[:300]}") from exc

        return ProviderResponse(
            text=text,
            model_returned=payload.get("model") or self.model,
            latency_ms=latency_ms,
            citation_urls=_extract_citation_urls(payload),
        )
