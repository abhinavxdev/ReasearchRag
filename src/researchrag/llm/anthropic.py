"""Anthropic Messages API through HTTP; no provider SDK dependency."""

import json
from collections.abc import Iterator

import httpx

from researchrag.core.exceptions import ProviderError


class AnthropicClient:
    """Claude streaming transport with explicit timeout and safe error messages."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key, self.model = api_key, model

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Read text_delta events from the Messages SSE protocol."""
        if not self.api_key:
            raise ProviderError("Set RESEARCHRAG_ANTHROPIC_API_KEY or select the ollama provider")
        try:
            with httpx.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json={
                    "model": self.model,
                    "max_tokens": 1200,
                    "system": system,
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=180,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event.get("type") == "error":
                        raise ProviderError("Anthropic reported a streaming error")
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield str(delta["text"])
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ProviderError(
                "Anthropic request failed; check credentials, model and connectivity"
            ) from exc
