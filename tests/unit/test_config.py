import pytest
from pydantic import ValidationError

from researchrag.config import Settings


def test_defaults(tmp_path):
    assert Settings(_env_file=None, data_dir=tmp_path).top_k == 6


def test_invalid():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, top_k=0)


def test_environment(monkeypatch):
    monkeypatch.setenv("RESEARCHRAG_TOP_K", "9")
    assert Settings(_env_file=None).top_k == 9
