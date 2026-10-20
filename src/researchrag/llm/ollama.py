"""Local Ollama HTTP transport with grammar-constrained citation fields."""

import json
import re
from collections.abc import Iterator

import httpx

from researchrag.core.exceptions import ProviderError
from researchrag.llm.prompts import REFUSAL


class OllamaClient:
    """Use a pre-pulled model and normalize structured sentences to citation text."""

    def __init__(
        self, base_url: str, model: str, context_size: int = 8192, batch_size: int = 64
    ) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model
        self.context_size, self.batch_size = context_size, batch_size

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Constrain source IDs to the retrieved set; downstream validation still applies.

        Small local models often omit IDs in free prose. Grammar-constrained JSON
        makes citation structure reliable, but does not prove semantic entailment.
        """
        ids = list(dict.fromkeys(re.findall(r"SOURCE \[(S_[a-z0-9]+)\]", prompt)))
        if not ids:
            yield REFUSAL
            return
        schema = {
            "type": "object",
            "properties": {
                "sentences": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "source_id": {"type": "string", "enum": ids},
                        },
                        "required": ["text", "source_id"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["sentences"],
            "additionalProperties": False,
        }
        transport = (
            "\nTRANSPORT FORMAT OVERRIDE: Return JSON with a sentences array, not prose. "
            "Each item has text (one supported sentence WITHOUT brackets or citations) and "
            "source_id (the supporting source ID). If evidence is insufficient, sentences is []. "
            "Do not put the source ID in text. Example: "
            '{"sentences":[{"text":"The optimizer was Adam.","source_id":"' + ids[0] + '"}]}'
        )
        try:
            parts: list[str] = []
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": True,
                    "format": schema,
                    "options": {
                        "temperature": 0,
                        "num_predict": 900,
                        "num_ctx": self.context_size,
                        "num_batch": self.batch_size,
                    },
                    "messages": [
                        {"role": "system", "content": system + transport},
                        {"role": "user", "content": prompt + transport},
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
                    parts.append(str(event.get("message", {}).get("content", "")))
                    if sum(map(len, parts)) > 24000:
                        raise ProviderError("Ollama output limit exceeded")
            result = json.loads("".join(parts))
            sentences = result["sentences"]
            if not isinstance(sentences, list):
                raise ValueError("Invalid sentences")
            if not sentences:
                yield REFUSAL
            for sentence in sentences:
                if sentence["source_id"] not in ids or not isinstance(sentence["text"], str):
                    raise ValueError("Invalid source")
                yield f"{sentence['text'].strip().rstrip('.')} [{sentence['source_id']}]. "
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                "Ollama request failed; check model, memory, connectivity and structured-output support"
            ) from exc
