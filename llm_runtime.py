"""Shared chat-model runtime helpers.

The application uses DeepSeek through the OpenAI-compatible API.  V4 enables
thinking by default, so every call must choose the mode explicitly instead of
silently inheriting a provider default.
"""

from functools import lru_cache

from openai import OpenAI

from .config.settings import settings


@lru_cache(maxsize=4)
def _get_chat_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
        max_retries=1,
    )


def get_chat_client(config=settings) -> OpenAI:
    """Reuse an HTTP connection pool for each configured chat endpoint."""
    return _get_chat_client(
        config.CHAT_API_KEY,
        config.CHAT_BASE_URL,
    )


def thinking_extra_body(enabled: bool) -> dict:
    return {
        "thinking": {
            "type": "enabled" if enabled else "disabled",
        }
    }
