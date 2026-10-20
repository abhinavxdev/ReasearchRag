"""Restricted arXiv downloader, without arbitrary URL fetching."""

import re
from urllib.parse import urlparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from researchrag.core.exceptions import ResearchError

ID_PATTERN = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")


def parse_id(value: str) -> str:
    """Accept modern/legacy IDs and arxiv.org abs/pdf URLs only."""
    value = value.strip()
    if "://" in value:
        url = urlparse(value)
        if url.scheme != "https" or url.hostname not in {"arxiv.org", "www.arxiv.org"}:
            raise ResearchError("Use an https://arxiv.org/abs/ or /pdf/ URL")
        value = re.sub(r"^/(?:abs|pdf)/", "", url.path).removesuffix(".pdf")
    if not ID_PATTERN.fullmatch(value):
        raise ResearchError("Invalid arXiv ID")
    return value


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=8),
    reraise=True,
)
def download(arxiv_id: str, max_bytes: int) -> bytes:
    """Download only from the fixed arXiv host with bounded memory and retries."""
    try:
        with httpx.stream(
            "GET",
            f"https://arxiv.org/pdf/{parse_id(arxiv_id)}",
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": "ResearchRAG/0.1 (local research tool)"},
        ) as response:
            response.raise_for_status()
            data = bytearray()
            for block in response.iter_bytes():
                data.extend(block)
                if len(data) > max_bytes:
                    raise ResearchError("arXiv PDF exceeds upload size limit")
            if not data.startswith(b"%PDF-"):
                raise ResearchError("arXiv did not return a PDF")
            return bytes(data)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code < 500 and exc.response.status_code != 429:
            raise ResearchError(f"arXiv returned HTTP {exc.response.status_code}") from exc
        raise
