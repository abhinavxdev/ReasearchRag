import pytest

from researchrag.retrieval.hybrid import rrf


def test_rrf():
    result = rrf([["a", "b"], ["b", "c"]])
    assert result[0][0] == "b"
    assert result[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert rrf([["a", "a"]])[0][1] == pytest.approx(1 / 61)
    assert rrf([]) == []
