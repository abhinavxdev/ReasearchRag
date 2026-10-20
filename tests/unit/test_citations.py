from researchrag.core.models import Citation
from researchrag.rag.citations import validate_answer

SOURCES = {
    "S_abc": Citation(
        source_id="S_abc",
        paper_id="p",
        title="Study",
        page=2,
        section="Methods",
        snippet="Exact evidence",
    )
}


def test_valid():
    result = validate_answer("The method uses attention [S_abc].", SOURCES)
    assert not result.refused
    assert "[Study, p.2, Methods]" in result.answer
    assert result.citations[0].snippet == "Exact evidence"


def test_unknown_and_missing():
    for text in ["A claim [S_fake].", "A claim.", "A claim [S_abc]. Another uncited claim."]:
        assert validate_answer(text, SOURCES).refused


def test_post_punctuation():
    assert not validate_answer("A claim. [S_abc]", SOURCES).refused
