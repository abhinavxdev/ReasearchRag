"""Local Ollama HTTP streaming transport."""

import json
from collections.abc import Iterator

import httpx

from researchrag.core.exceptions import ProviderError


class OllamaClient:
    """Use a pre-pulled local model; no automatic switch to a paid API."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Read Ollama's newline-delimited chat stream."""
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": True,
                    "options": {"temperature": 0, "num_predict": 900, "num_ctx": 4096},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=300,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise ProviderError("Ollama reported an error; check the model is pulled")
                    text = event.get("message", {}).get("content", "")
                    if text:
                        yield str(text)
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(
                "Ollama unavailable; start Ollama and pull the configured model"
            ) from exc
