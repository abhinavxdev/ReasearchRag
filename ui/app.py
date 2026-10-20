"""Thin Streamlit client. All ingestion, retrieval and generation live behind HTTP."""

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

API = os.getenv("RESEARCHRAG_API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="ResearchRAG · Paper workspace", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.stApp {background:#f7f8fa;color:#172638}
[data-testid="stSidebar"] {background:#edf1f5;border-right:1px solid #dce2e9}
h1,h2,h3 {letter-spacing:-.035em}
[data-testid="stChatMessage"] {background:white;border:1px solid #e0e6ec;border-radius:14px}
[data-testid="stHeader"] {background:transparent}
.block-container {max-width:1160px;padding-top:2.8rem}
.eyebrow {font-size:12px;letter-spacing:2px;font-weight:700;color:#507a78;margin-bottom:12px}
.hero {font-size:44px;font-weight:650;letter-spacing:-2px;line-height:1.1;margin:8px 0 18px}
.sub {color:#667787;font-size:16px;line-height:1.6;max-width:720px}
.note {padding:16px 20px;border-left:3px solid #558d85;background:#edf4f1;color:#405e57;border-radius:4px}
</style>""",
    unsafe_allow_html=True,
)


def api(method: str, path: str, **kwargs: Any) -> Any:
    """Make a server-side API call and present safe HTTP errors."""
    response = httpx.request(method, API + path, timeout=180, **kwargs)
    if response.is_error:
        try:
            detail = response.json().get("detail", response.reason_phrase)
        except ValueError:
            detail = response.reason_phrase
        raise RuntimeError(str(detail))
    return response.json() if response.content else None


def render_sources(citations: list[dict[str, Any]]) -> None:
    """Render exact source text as inert text, never as PDF-supplied HTML."""
    for source in citations:
        label = f"{source['title']} · p.{source['page']} · {source['section']}"
        with st.expander(label):
            st.caption(f"Source {source['source_id']} · Paper {source['paper_id']}")
            st.text(source["snippet"])


def answer_stream(
    question: str, paper_ids: list[str] | None, top_k: int, result: dict[str, Any]
) -> Iterator[str]:
    """Consume SSE; save final citation mapping separately from streamed text."""
    with httpx.stream(
        "POST",
        API + "/api/query/stream",
        timeout=360,
        json={"question": question, "paper_ids": paper_ids, "top_k": top_k},
    ) as response:
        response.raise_for_status()
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "token":
                    yield str(data["text"])
                elif event == "done":
                    result.update(data)
                elif event == "error":
                    raise RuntimeError(data["detail"])
    if "answer" not in result:
        raise RuntimeError("Response interrupted before citation validation completed")


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### ◈ ResearchRAG")
    st.caption("YOUR PRIVATE PAPER WORKSPACE")
    st.divider()
    st.markdown("#### Add to your library")
    files = st.file_uploader("Research PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest PDFs", use_container_width=True, disabled=not files):
        for file in files or []:
            try:
                paper = api(
                    "POST",
                    "/api/papers",
                    files={"file": (file.name, file.getvalue(), "application/pdf")},
                )
                st.success(f"{file.name}: {paper['status']}")
            except (httpx.HTTPError, RuntimeError) as exc:
                st.error(str(exc))
    with st.form("arxiv", clear_on_submit=True):
        arxiv_id = st.text_input("arXiv ID or URL", placeholder="1706.03762")
        if st.form_submit_button("Import from arXiv", use_container_width=True):
            if arxiv_id.strip():
                try:
                    paper = api("POST", "/api/papers/arxiv", json={"arxiv": arxiv_id})
                    st.success(f"Paper {paper['status']}; refresh for updates")
                except (httpx.HTTPError, RuntimeError) as exc:
                    st.error(str(exc))
    st.divider()
    top_k = st.slider("Evidence passages", 1, 12, 6)
    st.caption(
        "Answers use only ingested papers. Citations link to exact extracted text, not external search."
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="eyebrow">LITERATURE REVIEW, WITH EVIDENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Read deeper. Trace every answer.</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">A focused workspace for your research papers. Ask across your library, compare findings, and inspect the passages behind each answer.</p>',
    unsafe_allow_html=True,
)

try:
    papers: list[dict[str, Any]] = api("GET", "/api/papers")
except (httpx.HTTPError, RuntimeError) as exc:
    st.error(f"Cannot reach the ResearchRAG API. Start it with make run-api. {exc}")
    st.stop()

ready = [paper for paper in papers if paper["status"] == "done"]
cols = st.columns(3)
cols[0].metric("Papers in library", len(papers))
cols[1].metric("Ready to explore", len(ready))
cols[2].metric("Indexed passages", sum(p["chunk_count"] for p in ready))
chat_tab, library_tab = st.tabs(["Ask your papers", "Paper library"])
with library_tab:
    heading, action = st.columns([4, 1])
    heading.subheader("Your source collection")
    if action.button("Refresh status", use_container_width=True):
        st.rerun()
    if not papers:
        st.info("Add a PDF or import an arXiv paper to begin.")
    for paper in papers:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**{paper['title']}**")
            left.caption(
                f"{paper['filename']} · {paper['status'].upper()} · {paper['chunk_count']} passages"
            )
            if paper["error"]:
                left.error(paper["error"])
            busy = paper["status"] in {"pending", "processing"}
            if right.button(
                "Re-ingest", key=f"re-{paper['id']}", disabled=busy, use_container_width=True
            ):
                try:
                    api("POST", f"/api/papers/{paper['id']}/reingest")
                    st.rerun()
                except (httpx.HTTPError, RuntimeError) as exc:
                    st.error(str(exc))
            confirm = right.checkbox("Confirm removal", key=f"confirm-{paper['id']}", disabled=busy)
            if right.button(
                "Delete",
                key=f"del-{paper['id']}",
                disabled=busy or not confirm,
                use_container_width=True,
            ):
                try:
                    api("DELETE", f"/api/papers/{paper['id']}")
                    st.rerun()
                except (httpx.HTTPError, RuntimeError) as exc:
                    st.error(str(exc))

with chat_tab:
    labels = {paper["id"]: paper["title"] for paper in ready}
    selection = st.multiselect(
        "Focus on specific papers",
        options=list(labels),
        format_func=lambda key: labels[key],
        placeholder="All completed papers",
    )
    st.caption(
        "No selection searches all completed papers. Each question is independent; restate context for follow-ups."
    )
    if not st.session_state.messages:
        st.markdown(
            '<div class="note">Start with a precise question, such as “What optimizer and training schedule did the authors use?”</div>',
            unsafe_allow_html=True,
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("citations", []))
    question = st.chat_input("Ask a question about your papers…", disabled=not ready)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            result: dict[str, Any] = {}
            try:
                with st.spinner("Finding evidence and checking citations…"):
                    st.write_stream(answer_stream(question, selection or None, top_k, result))
                render_sources(result.get("citations", []))
                for warning in result.get("warnings", []):
                    st.warning(warning)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": result.get("citations", []),
                    }
                )
            except (httpx.HTTPError, RuntimeError) as exc:
                st.error(str(exc))
    if st.session_state.messages:
        export = "\n\n".join(
            f"## {m['role'].title()}\n\n{m['content']}"
            + "".join(
                f"\n\n> {c['title']}, p.{c['page']}, {c['section']}\n> "
                + c["snippet"].replace("\n", "\n> ")
                for c in m.get("citations", [])
            )
            for m in st.session_state.messages
        )
        st.download_button(
            "Export conversation · Markdown", export, "researchrag-notes.md", "text/markdown"
        )
