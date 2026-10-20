import pytest

from researchrag.core.exceptions import ResearchError
from researchrag.ingestion.arxiv import parse_id
from researchrag.ingestion.chunker import chunk_sections
from researchrag.ingestion.cleaner import clean_pages
from researchrag.ingestion.sectioner import Section, sections


def test_sections_carry_page():
    parts = sections(
        ["Abstract\nA summary\n1 Introduction\nA start", "continued\n2 Methods\nAn algorithm"]
    )
    assert [(p.page, p.name) for p in parts] == [
        (1, "Abstract"),
        (1, "Introduction"),
        (2, "Introduction"),
        (2, "Methods"),
    ]


def test_chunks_bounded_stable_overlap():
    parts = [Section(2, "Methods", " ".join(f"word{i}" for i in range(900)))]
    chunks = chunk_sections(parts, "paper", "Title", 100, 20)
    assert len(chunks) > 5
    assert chunks == chunk_sections(parts, "paper", "Title", 100, 20)
    assert all(c.page == 2 and c.section == "Methods" for c in chunks)
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())
    with pytest.raises(ValueError):
        chunk_sections(parts, "p", "t", 10, 10)


def test_cleaner():
    assert clean_pages(["A hy-\nphenated word"])[0] == "A hyphenated word"


def test_arxiv_validation():
    assert parse_id("https://arxiv.org/pdf/1706.03762.pdf") == "1706.03762"
    assert parse_id("hep-th/9901001") == "hep-th/9901001"
    with pytest.raises(ResearchError):
        parse_id("https://evil.test/1706.03762")
