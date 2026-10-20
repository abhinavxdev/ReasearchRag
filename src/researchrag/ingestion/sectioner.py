"""Heading detection for common research-paper structures."""

import re
from dataclasses import dataclass

HEADING = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*\.?|[IVX]+\.?)\s+)?"
    r"(abstract|introduction|background|related work|methods?|methodology|"
    r"materials and methods|approach|experiments?|results?(?: and discussion)?|"
    r"discussion|conclusions?|limitations?|references|bibliography)\s*[:.]?\s*$",
    re.I,
)


NUMBERED = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s{1,8}([A-Z][A-Za-z ,:&()/-]{2,65})\s*$")


@dataclass(frozen=True)
class Section:
    """Page-bounded section fragment."""

    page: int
    name: str
    text: str


def sections(pages: list[str]) -> list[Section]:
    """Carry detected section names across page boundaries."""
    name = "Front matter"
    result: list[Section] = []
    for page, text in enumerate(pages, 1):
        lines: list[str] = []
        for line in text.splitlines():
            match = HEADING.fullmatch(line) or NUMBERED.fullmatch(line)
            if match:
                if "\n".join(lines).strip():
                    result.append(Section(page, name, "\n".join(lines).strip()))
                name = match[1].title()
                lines = []
            else:
                lines.append(line)
        if "\n".join(lines).strip():
            result.append(Section(page, name, "\n".join(lines).strip()))
    return result
