"""Two-paper live demo; staged subprocess mode fits memory-limited development hosts."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from researchrag.config import get_settings

QUESTIONS = [
    ("1706.03762", "Which optimizer did the Transformer use?"),
    ("1706.03762", "How many learning rate warmup steps were used?"),
    ("1810.04805", "What percentage of WordPiece tokens is masked during BERT pre-training?"),
]


def prepare_demo(path: Path) -> None:
    """Run real ingestion/retrieval in a process that exits before LLM loading."""
    from researchrag.api.dependencies import build_services
    from researchrag.rag.context import build_context

    settings = get_settings()
    services = build_services(settings)
    papers = {}
    for arxiv_id in ["1706.03762", "1810.04805"]:
        paper = services.library.import_arxiv(arxiv_id)
        if paper.status == "failed":
            services.library.reingest(paper.id)
        services.library.ingest(paper.id)
        paper = services.library.get(paper.id)
        if paper.status != "done":
            raise RuntimeError(paper.error)
        papers[arxiv_id] = paper
        print(f"INGESTED {paper.title}: {paper.chunk_count} passages", flush=True)
    prepared = []
    for arxiv_id, question in QUESTIONS:
        hits = services.answers.retriever.search(question, 2, [papers[arxiv_id].id])
        hits = [h for h in hits if h.dense_score >= settings.min_dense_score]
        context, sources = build_context(hits, settings.context_chars)
        prepared.append(
            {
                "question": question,
                "context": context,
                "sources": {key: value.model_dump() for key, value in sources.items()},
            }
        )
    path.write_text(json.dumps(prepared))


def save_results(results: list[dict[str, Any]]) -> None:
    """Write answers with complete snippets; record refusals without hiding failures."""
    lines = [
        "# Live two-paper demo",
        "",
        "Real arXiv PDFs, BGE embeddings, Chroma/BM25/RRF, and Ollama. See verification.md for model and execution details.",
        "",
    ]
    for result in results:
        lines.extend([f"## {result['question']}", "", result["answer"], ""])
        for source in result["citations"]:
            lines.extend(
                [
                    f"### Source: {source['title']}, p.{source['page']}, {source['section']}",
                    f"ID: `{source['source_id']}`",
                    "",
                    "```text",
                    source["snippet"],
                    "```",
                    "",
                ]
            )
        if result["warnings"]:
            lines.append("Warnings: " + "; ".join(result["warnings"]))
    Path("docs/demo.md").write_text("\n".join(lines))
    Path("eval/demo-results.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    """Run the normal full service, or stage identical retrieval to reduce peak RAM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--prepare", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare:
        prepare_demo(args.prepare)
        return
    settings = get_settings()
    results: list[dict[str, Any]] = []
    if args.low_memory:
        from researchrag.core.models import Citation
        from researchrag.llm.anthropic import AnthropicClient
        from researchrag.llm.ollama import OllamaClient
        from researchrag.llm.prompts import REFUSAL, SYSTEM
        from researchrag.rag.citations import validate_answer

        # Release any previously loaded Ollama model before the embedding stage.
        if settings.llm_provider == "ollama":
            import httpx

            tags = httpx.get(settings.ollama_url + "/api/ps").json()
            for model in tags.get("models", []):
                httpx.post(
                    settings.ollama_url + "/api/generate",
                    json={"model": model["name"], "keep_alive": 0},
                    timeout=30,
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.json"
            subprocess.run([sys.executable, __file__, "--prepare", str(path)], check=True)
            prepared = json.loads(path.read_text())
        llm = (
            AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
            if settings.llm_provider == "anthropic"
            else OllamaClient(
                settings.ollama_url,
                settings.ollama_model,
                settings.ollama_context_size,
                settings.ollama_batch_size,
            )
        )
        for case in prepared:
            sources = {
                key: Citation.model_validate(value) for key, value in case["sources"].items()
            }
            raw = (
                "".join(
                    llm.stream(SYSTEM, f"CONTEXT\n{case['context']}\nQUESTION\n{case['question']}")
                )
                if sources
                else REFUSAL
            )
            result = validate_answer(raw, sources)
            results.append({"question": case["question"], **result.model_dump()})
            print(case["question"], "\n", result.answer, flush=True)
    else:
        from researchrag.api.dependencies import build_services

        with tempfile.TemporaryDirectory() as directory:
            prepare_demo(Path(directory) / "prepared.json")
        services = build_services(settings)
        papers = {p.arxiv_id: p.id for p in services.library.list() if p.arxiv_id}
        for arxiv_id, question in QUESTIONS:
            result = services.answers.answer(question, [papers[arxiv_id]], 2)
            results.append({"question": question, **result.model_dump()})
            print(question, "\n", result.answer, flush=True)
    save_results(results)


if __name__ == "__main__":
    main()
