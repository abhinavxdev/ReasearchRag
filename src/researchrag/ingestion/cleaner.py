"""Conservative removal of repeated margin lines and line-wrap hyphenation."""

import re
from collections import Counter


def clean_pages(pages: list[str]) -> list[str]:
    """Remove recurring headers/footers without removing repeated body paragraphs."""
    margins: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        margins.update(set(lines[:2] + lines[-2:]))
    repeated = {
        line
        for line, count in margins.items()
        if len(pages) >= 3 and count >= max(3, len(pages) * 0.6)
    }
    result = []
    for page in pages:
        lines = page.splitlines()
        keep = [
            line
            for i, line in enumerate(lines)
            if not (
                (i < 3 or i >= len(lines) - 3)
                and (line.strip() in repeated or re.fullmatch(r"\s*\d+\s*", line))
            )
        ]
        text = "\n".join(keep).replace("\x00", "")
        text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)
        result.append(text)
    return result
