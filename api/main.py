"""∆∆ FastAPI backend for Adaptive RAG with real-time streaming.

Provides:
- POST /api/chat - SSE streaming chat endpoint
- POST /api/ingest - File upload and ingestion
- GET /api/health - Health check
- Static files served from /frontend
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Project root for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Use light text model by default
os.environ.setdefault("USE_LIGHT_TEXT_MODEL", "1")

from langchain_core.messages import HumanMessage
from adaptive_rag.graph import create_adaptive_rag_graph

# Supported file types for ingestion (PDF, Word, Text, PowerPoint, Excel)
UPLOAD_ACCEPT = {".pdf", ".docx", ".txt", ".pptx", ".xlsx"}

# Default settings
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
MAX_RETRIES = 2

app = FastAPI(
    title="Adaptive RAG API",
    description="Real-time streaming RAG with LangGraph + ReAct self-reflection",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for graphs per model config; LRU eviction when full
MAX_GRAPH_CACHE = 10
_graphs: OrderedDict[tuple, Any] = OrderedDict()


def get_graph(model: str, max_retries: int):
    """Get or create a graph for the given config. Evicts least recently used when cache is full."""
    key = (model, max_retries)
    if key in _graphs:
        _graphs.move_to_end(key)
        return _graphs[key]
    if len(_graphs) >= MAX_GRAPH_CACHE:
        _graphs.popitem(last=False)
    _graphs[key] = create_adaptive_rag_graph(
        ollama_model=model,
        max_retries=max_retries,
        use_long_term_memory=False,
    )
    return _graphs[key]


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None
    model: str = OLLAMA_MODEL
    max_retries: int = MAX_RETRIES
    source_file: str | None = None  # Filter retrieval to this document
    document_role: str | None = None  # Filter retrieval to this role (e.g. resume, paper)


class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks: int = 0
    thread_id: str | None = None  # Set when ingest had no thread_id; client should use for chat


async def stream_rag_response(
    question: str,
    model: str,
    max_retries: int,
    thread_id: str,
    source_file: str | None = None,
    document_role: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream RAG response as Server-Sent Events with real-time step updates.
    """
    import queue
    import threading
    
    # Map node names to user-friendly descriptions
    STEP_NAMES = {
        "query_analysis": "🔀 Analyzing query...",
        "clarify": "❓ Asking for clarification...",
        "retrieve_vectorstore": "🔍 Searching knowledge base...",
        "check_info_sufficient": "📋 Checking memory & chunks...",
        "answer_from_memory_and_chunks": "✍️ Answering from context...",
        "retrieve_web": "🌐 Searching the web...",
        "retrieve_hybrid": "🔀 Hybrid search...",
        "check_relevant": "📋 Grading relevance...",
        "generate": "✍️ Generating response...",
        "check_hallucinations": "✅ Checking for hallucinations...",
        "check_answers_question": "🎯 Verifying answer...",
        "answer_rag": "📝 Finalizing RAG answer...",
        "answer_llm": "🤖 Using LLM directly...",
        "increment_retry": "🔄 Retrying...",
    }
    
    try:
        graph = get_graph(model, max_retries)
        
        yield f"data: {json.dumps({'type': 'step', 'content': '🧠 Analyzing query...'})}\n\n"
        await asyncio.sleep(0.01)
        
        initial = {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "thread_id": thread_id,
            "retry_count": 0,
            "max_retries": max_retries,
            "source_file": source_file,
            "document_role": document_role,
        }
        config = {"configurable": {"thread_id": thread_id}}
        
        # Use a queue to communicate between threads
        step_queue = queue.Queue()
        final_result = {}
        
        def run_graph_with_steps():
            """Run graph in a separate thread and push step updates to queue."""
            last_step = None
            try:
                for event in graph.stream(initial, config=config, stream_mode="updates"):
                    for node_name, node_state in event.items():
                        if node_name in STEP_NAMES and node_name != last_step:
                            last_step = node_name
                            step_queue.put(("step", STEP_NAMES[node_name]))
                        if "final_answer" in node_state:
                            final_result.update(node_state)
                step_queue.put(("done", None))
            except Exception as e:
                step_queue.put(("error", str(e)))
        
        # Start graph in background thread
        thread = threading.Thread(target=run_graph_with_steps)
        thread.start()
        
        # Yield step updates as they come
        while True:
            try:
                msg_type, content = step_queue.get(timeout=0.1)
                if msg_type == "step":
                    yield f"data: {json.dumps({'type': 'step', 'content': content})}\n\n"
                    await asyncio.sleep(0.01)
                elif msg_type == "error":
                    err_msg = content or "Unknown error"
                    payload = {'type':'error','content': err_msg, 'final_answer': f'Sorry, an error occurred: {err_msg}'}
                    yield f"data: {json.dumps(payload)}\n\n"
                    return
                elif msg_type == "done":
                    break
            except queue.Empty:
                if not thread.is_alive():
                    break
                await asyncio.sleep(0.05)
        
        thread.join()
        
        # Get final answer
        final = final_result if final_result else await asyncio.to_thread(graph.invoke, initial, config=config)
        
        answer = final.get("final_answer", "").strip()
        route = final.get("route", "unknown")
        source = final.get("source", "rag")
        
        yield f"data: {json.dumps({'type': 'step', 'content': '📝 Formatting response...'})}\n\n"
        await asyncio.sleep(0.01)
        
        if not answer:
            answer = "(No answer produced. Check Ollama and Milvus MCP.)"
        
        # Stream answer in chunks
        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.015)
        
        done_data = {
            "type": "done",
            "source": source,
            "route": route,
            "retry_count": final.get("retry_count", 0),
            "relevant": final.get("relevant"),
            "has_hallucinations": final.get("has_hallucinations"),
            "answers_question": final.get("answers_question"),
        }
        yield f"data: {json.dumps(done_data)}"
        
    except Exception as e:
        err_msg = str(e)
        payload = {'type': 'error', 'content': err_msg, 'final_answer': f'Sorry, an error occurred: {err_msg}'}
        yield f"data: {json.dumps(payload)}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Stream chat response using Server-Sent Events (SSE).
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    logger.info("Chat request - thread_id=%s, question=%r", thread_id, request.question)

    return StreamingResponse(
        stream_rag_response(
            question=request.question,
            model=request.model,
            max_retries=request.max_retries,
            thread_id=thread_id,
            source_file=request.source_file,
            document_role=request.document_role,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-ID": thread_id,
        },
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    thread_id: str | None = Form(None),
    document_role: str | None = Form(None),
):
    """
    Upload and ingest one or more documents into session-specific Milvus collection.
    Supports: PDF, DOCX, TXT, PPTX, XLSX
    Each chat session (thread_id) has its own document collection.
    Send either a single "file" (backward compatible) or multiple "files".
    If thread_id is omitted, a new one is generated and returned in the response message so the client can use it for chat.
    """
    if file is not None:
        uploads = [file]
    elif files:
        uploads = files
    else:
        raise HTTPException(status_code=400, detail="At least one file is required (use 'file' or 'files').")

    if not thread_id:
        thread_id = str(uuid.uuid4())

    paths: list[Path] = []
    for f in uploads:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in UPLOAD_ACCEPT:
            for p in paths:
                p.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Use: {', '.join(UPLOAD_ACCEPT)}",
            )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        content = await f.read()
        tmp.write(content)
        tmp.close()
        paths.append(Path(tmp.name))

    try:
        from ingest.milvus_indexer import run_ingestion
    except ImportError:
        for p in paths:
            p.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Ingestion module not found")

    try:
        results = run_ingestion(
            paths,
            use_custom_schema=True,
            use_light_text_model=True,
            thread_id=thread_id,
            document_role=document_role,
        )
        n = len(results)
        collection_name = f"docs_text_{thread_id}" if thread_id else "docs_text"
        num_files = len(uploads)
        msg = (
            f"Indexed {n} chunks from {num_files} file(s) into {collection_name}."
            if num_files > 1
            else f"Indexed {n} chunks from '{uploads[0].filename}' into {collection_name}."
        )
        return IngestResponse(success=True, message=msg, chunks=n, thread_id=thread_id)
    except Exception as e:
        return IngestResponse(success=False, message=str(e), chunks=0)
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


@app.api_route("/api/session/{thread_id}", methods=["DELETE", "POST"])
async def cleanup_session(thread_id: str):
    """
    Delete all collections and chat memory for a session when it ends.
    Clears Milvus document collections and LangGraph checkpointer state for this thread_id.
    Called when user refreshes the page or starts a new chat.
    Accepts both DELETE and POST (for sendBeacon compatibility).
    """
    results = {}
    try:
        from ingest.milvus_direct_client import MilvusDirectClient
        client = MilvusDirectClient()
        results["milvus"] = client.drop_session_collections(thread_id)
        logger.info("Session cleanup (Milvus) - thread_id=%s, results=%s", thread_id, results["milvus"])
    except Exception as e:
        logger.error("Session cleanup (Milvus) failed - thread_id=%s, error=%s", thread_id, e)
        results["milvus"] = {"error": str(e)}

    # Clear chat memory (checkpointer state) for this thread_id
    try:
        for graph in _graphs.values():
            checkpointer = getattr(graph, "checkpointer", None)
            if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
                checkpointer.delete_thread(thread_id)
                results["checkpointer"] = "deleted"
                logger.info("Session cleanup (checkpointer) - thread_id=%s", thread_id)
                break
        else:
            results["checkpointer"] = "no_checkpointer"
    except Exception as e:
        logger.error("Session cleanup (checkpointer) failed - thread_id=%s, error=%s", thread_id, e)
        results["checkpointer"] = {"error": str(e)}

    milvus_ok = "error" not in results.get("milvus", {}) if isinstance(results.get("milvus"), dict) else True
    checkpointer_ok = "error" not in results.get("checkpointer", {}) if isinstance(results.get("checkpointer"), dict) else True
    success = milvus_ok and checkpointer_ok
    return {"success": success, "thread_id": thread_id, "results": results}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": OLLAMA_MODEL}


# Serve frontend (prefer built dist for production if available)
FRONTEND_DIR = ROOT / "frontend" / "dist"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = ROOT / "frontend"

# Mount frontend at root to serve index.html and assets (API lives under /api/*)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
