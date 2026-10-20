"""Small, transparent retrieval/citation evaluation; not a semantic correctness benchmark."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from researchrag.api.dependencies import build_services
from researchrag.config import get_settings


def normalized(text: str) -> str:
    """Normalize PDF line wraps and common Unicode spacing for phrase checks."""
    return re.sub(r"\s+", "", text).casefold()


def main() -> None:
    """Import optional benchmark papers and report retrieval hit-rate plus citation validity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ingest", action="store_true", help="Download and ingest benchmark papers"
    )
    parser.add_argument(
        "--answers", action="store_true", help="Also call configured LLM for citation metrics"
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("eval/results.json"))
    args = parser.parse_args()
    services = build_services(get_settings())
    cases: list[dict[str, Any]] = json.loads(
        Path(__file__).with_name("questions.json").read_text()
    )[: args.limit]
    library = services.library
    mapping = {p.arxiv_id: p.id for p in library.list() if p.status == "done" and p.arxiv_id}
    for arxiv_id in sorted({case["paper"] for case in cases}):
        if arxiv_id not in mapping:
            if not args.ingest:
                parser.error(f"Missing benchmark paper {arxiv_id}; run with --ingest")
            paper = library.import_arxiv(arxiv_id)
            if paper.status == "failed":
                library.reingest(paper.id)
            library.ingest(paper.id)
            if library.get(paper.id).status != "done":
                raise RuntimeError(f"Ingestion failed: {library.get(paper.id).error}")
            mapping[arxiv_id] = paper.id
    rows: list[dict[str, Any]] = []
    valid_citations = 0
    all_citations = 0
    answered = 0
    for case in cases:
        ids = [mapping[case["paper"]]]
        hits = services.answers.retriever.search(case["question"], args.k, ids)
        # A hit requires the expected paper AND at least one gold phrase in a passage.
        hit = any(
            any(normalized(phrase) in normalized(h.chunk.text) for phrase in case["expected"])
            for h in hits
        )
        row: dict[str, Any] = {
            "question": case["question"],
            "paper": case["paper"],
            "retrieval_hit": hit,
            "retrieved": [
                {"id": h.chunk.id, "page": h.chunk.page, "section": h.chunk.section} for h in hits
            ],
        }
        if args.answers:
            result = services.answers.answer(case["question"], ids, args.k)
            row["result"] = result.model_dump()
            answered += int(not result.refused)
            source_chunks = {c.id: c for c in library.repo.chunks(ids)}
            for citation in result.citations:
                all_citations += 1
                chunk = source_chunks.get(citation.source_id.removeprefix("S_"))
                valid_citations += int(
                    chunk is not None
                    and chunk.text == citation.snippet
                    and chunk.page == citation.page
                    and chunk.title == citation.title
                    and chunk.section == citation.section
                )
        rows.append(row)
        print(f"{'HIT ' if hit else 'MISS'} {case['question']}", flush=True)
    report = {
        "questions": len(rows),
        "k": args.k,
        "retrieval_hit_rate_at_k": sum(row["retrieval_hit"] for row in rows) / max(1, len(rows)),
        "citation_validity_rate": valid_citations / all_citations if all_citations else None,
        "citation_count": all_citations,
        "answered": answered if args.answers else None,
        "answer_coverage": answered / max(1, len(rows)) if args.answers else None,
        "metric_note": "Citation validity is provenance only, not entailment. Null means no citations evaluated. Retrieval uses gold phrase proxies.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
