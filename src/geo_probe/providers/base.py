"""The entire abstraction layer: one ABC, two implementations."""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    model_returned: str
    latency_ms: int
    citation_urls: list[str] = field(default_factory=list)


class ProviderError(Exception):
    """Terminal failure -- do not retry."""


class RetryableProviderError(ProviderError):
    """429 / 5xx / transport error -- retry with backoff."""


class Provider(ABC):
    id: str

    @abstractmethod
    def complete(self, prompt: str, temperature: float | None) -> ProviderResponse: ...


T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    max_attempts: int = 5,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Exponential backoff with jitter on retryable errors, max 5 attempts."""
    r = rng or random.Random()
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except RetryableProviderError as exc:
            last = exc
            if attempt == max_attempts - 1:
                break
            sleep(base_delay * (2**attempt) + r.uniform(0.0, 0.25))
    assert last is not None
    raise last


DEFAULT_ENV_FILE = ".env"


def load_dotenv(path: str | Path = DEFAULT_ENV_FILE, override: bool = False) -> list[str]:
    """Read KEY=value lines from a .env file into os.environ.

    Returns the names of the keys applied (never the values -- a caller that
    logs the return value should not be able to leak a secret).

    An already-set environment variable wins unless override=True: a key you
    just exported should not be silently replaced by a stale file. Missing file
    is not an error; the keys may legitimately come from the environment.

    Deliberately small: blank lines and `#` comments are skipped, an optional
    `export ` prefix is allowed, and a value may be wrapped in matching single or
    double quotes. There is no interpolation, no escape processing, and no
    inline-comment stripping -- an unquoted `#` is part of the value, because an
    API key is more likely to contain one than a comment is.
    """
    p = Path(path)
    if not p.is_file():
        return []

    applied: list[str] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise ValueError(f"{p}:{lineno}: expected KEY=value, got {raw!r}")
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"{p}:{lineno}: empty key in {raw!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        applied.append(key)
    return applied


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ProviderError(
            f"{name} is not set. Put it in a .env file in the directory you are "
            f"running from (see .env.example), or export it in your shell. An "
            f"exported variable takes precedence over the .env file."
        )
    return value
