"""
Streamlit frontend for adaptive RAG (LangGraph + ReAct self-reflection).

Run: streamlit run adaptive_rag/streamlit_app.py
Requires: Ollama with model gpt-oss:120b-cloude (or set in UI).
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

# Project root so ingest and adaptive_rag are importable
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Default: use BGE-small (384 dim) so retrieval matches docs_text collection from ingest
os.environ.setdefault("USE_LIGHT_TEXT_MODEL", "1")

import streamlit as st
from langchain_core.messages import HumanMessage

from adaptive_rag.graph import create_adaptive_rag_graph

# Supported for ingestion (PDF, DOCX, TXT, MD, PPTX)
UPLOAD_ACCEPT = [".pdf", ".docx", ".txt", ".md", ".pptx"]

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloude")
MAX_RETRIES = 2


def _get_or_create_graph(model: str, max_retries: int, use_long_term_memory: bool):
    """Reuse same graph (and checkpointer) per session so memory is preserved."""
    key = ("rag_graph", model, max_retries, use_long_term_memory)
    if key not in st.session_state:
        st.session_state[key] = create_adaptive_rag_graph(
            ollama_model=model,
            max_retries=max_retries,
            use_long_term_memory=use_long_term_memory,
        )
    return st.session_state[key]


def run_adaptive_rag(
    question: str,
    model: str,
    max_retries: int,
    thread_id: str,
    use_long_term_memory: bool = False,
) -> dict:
    """Run the adaptive RAG graph with memory (short- or long-term checkpointer)."""
    graph = _get_or_create_graph(model, max_retries, use_long_term_memory)
    initial = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "retry_count": 0,
        "max_retries": max_retries,
    }
    config = {"configurable": {"thread_id": thread_id}}
    final = graph.invoke(initial, config=config)
    return final


def ingest_uploaded_file(uploaded_file, use_light_text_model: bool = True) -> tuple[bool, str]:
    """Save uploaded file to temp dir and run ingestion (chunk → embed → Milvus via MCP). Returns (success, message)."""
    try:
        from ingest.milvus_indexer import run_ingestion
    except ImportError:
        from milvus_indexer import run_ingestion
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in UPLOAD_ACCEPT:
        return False, f"Unsupported type '{ext}'. Use: {', '.join(UPLOAD_ACCEPT)}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = Path(tmp.name)
    try:
        results = run_ingestion([path], use_custom_schema=False, use_light_text_model=use_light_text_model)
        path.unlink(missing_ok=True)
        n = len(results)
        return True, f"Indexed **{n}** chunks from `{uploaded_file.name}` into Milvus."
    except Exception as e:
        path.unlink(missing_ok=True)
        return False, str(e)


def main():
    st.set_page_config(page_title="Adaptive RAG", page_icon="🔀", layout="centered")
    st.title("🔀 Adaptive RAG (ReAct + self-reflection)")
    st.caption("Query analysis → Vectorstore / Web / Direct LLM → RAG + relevance/hallucination/answer checks")

    model = st.sidebar.text_input("Ollama model", value=OLLAMA_MODEL, help="e.g. gpt-oss:120b-cloude, llama3.2, qwen2.5")
    max_retries = st.sidebar.number_input("Max self-reflection retries", min_value=1, max_value=5, value=MAX_RETRIES)
    show_trace = st.sidebar.checkbox("Show route & trace", value=True)
    use_long_term_memory = st.sidebar.checkbox(
        "Use long-term memory (SQLite)",
        value=False,
        help="Persist conversation across restarts. Requires: pip install langgraph-checkpoint-sqlite",
    )
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    st.sidebar.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")
    if st.sidebar.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📤 Upload file**")
    use_light_text = st.sidebar.checkbox("Use light text model (BGE-small, ~130MB)", value=True, help="Faster download & less RAM; 384 dim, dense only. Uncheck for BGE-M3 (dense+sparse, large).")
    os.environ["USE_LIGHT_TEXT_MODEL"] = "true" if use_light_text else "false"  # keep retrieval in sync with upload
    st.sidebar.caption("PDF, DOCX, TXT, MD, PPTX → chunk & index into Milvus")
    if "last_ingested_file" not in st.session_state:
        st.session_state.last_ingested_file = None
    uploaded = st.sidebar.file_uploader(
        "Choose a document",
        type=[e.lstrip(".") for e in UPLOAD_ACCEPT],
        label_visibility="collapsed",
    )
    upload_key = (uploaded.name, uploaded.size) if uploaded else None
    if uploaded is not None and upload_key != getattr(st.session_state, "last_ingested_file", None):
        with st.sidebar.status("Chunking & embedding…", state="running") as status:
            ok, msg = ingest_uploaded_file(uploaded, use_light_text_model=use_light_text)
            status.update(label="Done" if ok else "Error", state="complete" if ok else "error")
            if ok:
                st.sidebar.success(msg)
                st.session_state.last_ingested_file = upload_key
            else:
                st.sidebar.error(msg)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("source"):
                st.caption(f"Source: **{msg['source']}**")

    if prompt := st.chat_input("Ask a question (internal docs → vectorstore, current events → web, else → direct LLM)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        final = None
        with st.chat_message("assistant"):
            with st.spinner("Query analysis → retrieve → generate → self-reflection…"):
                try:
                    final = run_adaptive_rag(
                        prompt,
                        model=model,
                        max_retries=max_retries,
                        thread_id=st.session_state.thread_id,
                        use_long_term_memory=use_long_term_memory,
                    )
                    answer = final.get("final_answer", "")
                    source = final.get("source", "rag")
                    route = final.get("route", "")
                    if not answer:
                        answer = "(No answer produced; check Ollama and Milvus MCP.)"
                    st.markdown(answer)
                    st.caption(f"Source: **{source}**")
                    if show_trace and final:
                        with st.expander("Route & trace"):
                            st.json({
                                "route": route,
                                "source": source,
                                "retry_count": final.get("retry_count", 0),
                                "relevant": final.get("relevant"),
                                "has_hallucinations": final.get("has_hallucinations"),
                                "answers_question": final.get("answers_question"),
                            })
                except Exception as e:
                    st.error(str(e))
                    st.exception(e)

        if final:
            st.session_state.messages.append({
                "role": "assistant",
                "content": final.get("final_answer", ""),
                "source": final.get("source", "rag"),
            })

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Flow:** Query analysis → Vectorstore / Web / Direct LLM → RAG + self-reflection (relevance → generate → hallucinations? → answers question?) → Answer with RAG or fallback to LLM.")


if __name__ == "__main__":
    main()
