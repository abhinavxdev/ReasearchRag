# Verification record

Date: 2026-10-20. Environment: Linux x86_64, Python 3.13.16, approximately 2 GB RAM.
All results below are observed runs, not a claim that every deployment/model is verified.

## Milestones

| Milestone | Verification performed |
|---|---|
| M1 | Installed project; config/environment tests; Ruff check/format; strict mypy; git commit |
| M2 | Downloaded real arXiv 1706.03762; extracted/printed page and section chunks; unit tests; Ruff/mypy; commit |
| M3 | Downloaded BGE-small weights; persisted Chroma vectors; actual attention/optimizer queries; Ruff/mypy/tests; commit |
| M4 | Real PDF + SQLite + Chroma + deterministic fake LLM integration; CLI citation validation; provider error/refusal checks; Ruff/mypy/tests; commit |
| M5 | Actual API process health; TestClient upload, duplicate, job status, filter, re-ingest, delete, invalid/scanned/oversize input, SSE; live Ollama answer; Ruff/mypy/tests; commit |
| M6 | Streamlit server started; Chromium browser rendered library, controls, chat, and submitted a query; screenshot saved. SSE/result protocol is tested independently. Browser session closed before that slow live response finished; not claimed as a full browser chat completion test. Ruff/mypy/tests; commit |
| M7 | Expanded regression tests; 12-query real retrieval eval; two-paper, three-question live-model demo; exact source provenance audit; Compose YAML parsing; packaging; final checks and commit |

## Final checks

- `ruff check .` and `ruff format --check .`: pass.
- `mypy src ui eval`: strict mode, 46 source files pass.
- `pytest -q`: **22 passed**, fully offline/deterministic.
- `pip check`: no broken requirements.
- Wheel build: `python -m build --wheel` succeeded.
- FastAPI and Streamlit processes were run on 0.0.0.0 and rendered/served locally.
- Docker CLI/daemon unavailable: YAML parsed; Docker build/runtime **not tested**.
- Anthropic: mock SSE protocol and missing-key tests pass; real Claude API **not tested**, no key supplied.
- Optional cross-encoder: adapter statically checked; model/runtime **not exercised**.

## Retrieval evaluation

`OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 python eval/run.py`

- Papers: Attention Is All You Need (1706.03762), BERT (1810.04805).
- Latest stored corpus after final parser re-ingestion: 39 Transformer passages; 55 BERT passages.
- 12 questions; k=6; **11 hits (91.67%)**.
- Miss: BERT's pre-training optimizer. This is a real miss, not removed from the set.
- Gold phrase matching is a proxy. It can count a passage containing a phrase without a sufficient answer.
- Full retrieval report: `eval/results.json`. No LLM was called during this 12-case retrieval run;
  citation validity is explicitly null in that report.

## Three-question live demo

Successful invocation (Ollama 0.40.6 installed temporarily for verification):

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 RESEARCHRAG_LLM_PROVIDER=ollama \
RESEARCHRAG_OLLAMA_MODEL=qwen2.5:1.5b RESEARCHRAG_OLLAMA_CONTEXT_SIZE=4096 \
RESEARCHRAG_OLLAMA_BATCH_SIZE=16 python eval/demo.py --low-memory
```

This uses real arXiv downloads/import dedup, real PDF parsing, local BGE,
Chroma + BM25/RRF, real Ollama generation, verbatim-excerpt validation,
and normal citation formatting. **No fake LLM or fake embeddings** in this demo.
A retrieval subprocess exits before loading the LLM to avoid holding both sets of weights
concurrently; this is not a claim that the normal single-process API fits a 1.5B model in 2 GB.

Observed answers:

1. Adam, β1=0.9, β2=0.98, ε=10⁻⁹ — Transformer p.7, Optimizer.
2. `warmup_steps = 4000` — Transformer p.7, Optimizer.
3. `15% of all WordPiece tokens` — BERT p.4, Pre-Training BERT.

All three were manually checked against their source passages and programmatically audited
against SQLite after final re-ingestion. **3/3 citations valid; 3/3 answered**. This is a
provenance metric over three examples, not evidence of general semantic accuracy.
See `demo.md`, `eval/demo-results.json`, and `eval/demo-metrics.json`.

## Failures and changes made because of them

- Default unconstrained PyTorch download failed mid-transfer. CPU-only PyTorch 2.6.0 worked.
- 1–3B Ollama models failed loading concurrently with the resident embedding model under 2 GB.
  No fake output was substituted; the staged demo is an explicit memory workaround.
- Tiny qwen2.5:0.5b omitted source IDs in free prose. A JSON grammar corrected citation structure.
- That same tiny model then invented a BERT tokenizer answer with a valid source ID. The initial
  unsafe result is retained in `initial-demo-failure.md` as a **known wrong answer**, not an example
  to trust. This motivated verbatim-excerpt checks for the local provider. A regression test now
  rejects invented excerpts. Tiny models still often refuse answerable questions or select an
  irrelevant but genuine excerpt. Use a stronger local model with adequate RAM.
- BERT's two-column text originally interleaved. Added column-aware extraction and ligature
  normalization, then re-ingested and reran retrieval evaluation.
- ArXiv's rotated margin stamp was initially mistaken for a paper title. Title detection now
  ignores rotated text.
- Source-internal `[20]` or `[MASK]` markers conflicted with output citation parsing. Local answer
  excerpts render these markers using parentheses after exact-text verification; source text stays intact.
- A final check initially stalled due to a resident local LLM consuming RAM. Unloaded it and reran
  checks successfully. Model binaries/caches are not packaged in the downloadable source.

## Recommended next verification

Run the full API and browser chat on an 8 GB+ host with llama3.2:3b or a stronger local model;
run the Docker image; smoke-test a permitted Claude model; calibrate confidence/refusal metrics
and add semantic entailment/relevance evaluation before trusting research conclusions.
