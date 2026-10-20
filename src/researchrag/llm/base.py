"""Provider-neutral streaming language model interface."""

from collections.abc import Iterator
from typing import Protocol


class LLMClient(Protocol):
    """Yield raw provider text; downstream validation happens before public output."""

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Stream completion fragments or raise a sanitized ProviderError."""
        ...
