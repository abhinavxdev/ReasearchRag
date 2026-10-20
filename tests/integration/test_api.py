from pathlib import Path

from fastapi.testclient import TestClient
from test_rag import FakeEmbedder, FakeLLM

from researchrag.api.app import create_app
from researchrag.api.dependencies import build_services
from researchrag.config import Settings
from researchrag.retrieval.chroma import ChromaStore


def test_library_stream_and_delete(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    services = build_services(
        settings, FakeEmbedder(), ChromaStore(tmp_path / "chroma", "api"), FakeLLM()
    )
    with TestClient(create_app(settings, services)) as client:
        assert client.get("/api/health").status_code == 200
        data = Path("tests/fixtures/research.pdf").read_bytes()
        response = client.post(
            "/api/papers", files={"file": ("../../paper.pdf", data, "application/pdf")}
        )
        assert response.status_code == 202
        paper_id = response.json()["id"]
        assert client.get(f"/api/papers/{paper_id}").json()["status"] == "done"
        duplicate = client.post(
            "/api/papers", files={"file": ("paper.pdf", data, "application/pdf")}
        )
        assert duplicate.json()["id"] == paper_id
        assert len(client.get("/api/papers").json()) == 1
        result = client.post(
            "/api/query", json={"question": "How does fusion work?", "paper_ids": [paper_id]}
        )
        assert not result.json()["refused"]
        assert result.json()["citations"][0]["paper_id"] == paper_id
        stream = client.post("/api/query/stream", json={"question": "fusion"})
        assert "event: token" in stream.text
        assert "event: done" in stream.text
        assert client.post("/api/query", json={"question": " "}).status_code == 422
        assert client.post("/api/query", json={"question": "fusion", "paper_ids": []}).json()[
            "refused"
        ]
        assert client.post(f"/api/papers/{paper_id}/reingest").status_code == 202
        assert client.get(f"/api/papers/{paper_id}").json()["status"] == "done"
        assert client.delete(f"/api/papers/{paper_id}").status_code == 204
        assert not services.library.repo.chunks()
        assert client.get(f"/api/papers/{paper_id}").status_code == 404


def test_bad_pdf_and_scan(tmp_path: Path) -> None:
    import pymupdf

    settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_mb=1)
    services = build_services(
        settings, FakeEmbedder(), ChromaStore(tmp_path / "chroma", "bad"), FakeLLM()
    )
    with TestClient(create_app(settings, services)) as client:
        assert (
            client.post(
                "/api/papers", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/papers",
                files={"file": ("x.pdf", b"%PDF-" + b"x" * 1048576, "application/pdf")},
            ).status_code
            == 400
        )
        assert (
            client.post("/api/papers/arxiv", json={"arxiv": "https://evil.test/abc"}).status_code
            == 400
        )
        doc = pymupdf.open()
        doc.new_page()
        data = doc.tobytes()
        doc.close()
        result = client.post("/api/papers", files={"file": ("scan.pdf", data, "application/pdf")})
        status = client.get("/api/papers/" + result.json()["id"]).json()
        assert status["status"] == "failed"
        assert "OCR" in status["error"]
