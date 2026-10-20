import json

import httpx
import pytest

from researchrag.core.exceptions import ProviderError
from researchrag.llm.anthropic import AnthropicClient
from researchrag.llm.ollama import OllamaClient


def test_anthropic_missing_key():
    with pytest.raises(ProviderError):
        list(AnthropicClient("", "model").stream("system", "prompt"))


def test_ollama_structured(monkeypatch):
    output = json.dumps({"sentences": [{"text": "Uses Adam.", "source_id": "S_abc"}]})
    event = json.dumps({"message": {"content": output}}) + "\n"

    class Response:
        def __enter__(self):
            return httpx.Response(200, text=event, request=httpx.Request("POST", "http://test"))

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(httpx, "stream", lambda *args, **kwargs: Response())
    text = "".join(
        OllamaClient("http://test", "test").stream(
            "system", "SOURCE [S_abc] | t | p.1 | s\nUses Adam.\nEND SOURCE"
        )
    )
    assert text.strip() == "Uses Adam [S_abc]."


def test_anthropic_sse(monkeypatch):
    class Response:
        def __enter__(self):
            return httpx.Response(
                200,
                text='data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\n',
                request=httpx.Request("POST", "http://test"),
            )

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(httpx, "stream", lambda *args, **kwargs: Response())
    assert "".join(AnthropicClient("test", "model").stream("system", "question")) == "Hello"


def test_ollama_rejects_invented_excerpt(monkeypatch):
    from researchrag.llm.prompts import REFUSAL

    output = json.dumps(
        {"sentences": [{"text": "Uses an invented optimizer.", "source_id": "S_abc"}]}
    )

    class Response:
        def __enter__(self):
            return httpx.Response(
                200,
                text=json.dumps({"message": {"content": output}}) + "\n",
                request=httpx.Request("POST", "http://test"),
            )

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(httpx, "stream", lambda *args, **kwargs: Response())
    assert (
        "".join(
            OllamaClient("http://test", "model").stream(
                "system", "SOURCE [S_abc] | title\nUses Adam.\nEND SOURCE"
            )
        )
        == REFUSAL
    )
