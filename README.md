# Agentic RAG Backend (FastAPI)

FastAPI backend for an Agentic RAG system built with LangGraph/LangChain. It supports real‑time streaming responses, document ingestion to Milvus (Zilliz Cloud or local Milvus Lite), hybrid retrieval with reranking, and optional web search via Tavily MCP.

- API: `POST /api/chat` (SSE), `POST /api/ingest`, `GET /api/health`, session cleanup at `/api/session/{thread_id}`
- Ingestion: PDF, DOCX, TXT, PPTX, XLSX → chunk → embed (text + images) → Milvus
- Retrieval: Milvus vector search + rerank; optional image retrieval; optional web search tools
- Frontend: If `backend/frontend/index.html` exists, it’s served at `/`


## Prerequisites (Ollama first)

- Install Ollama and have it running locally:
  - macOS: `brew install ollama` (or download from ollama.com)
  - Login (if required by your registry): `ollama login`
  - Run the server: `ollama serve`
  - Pull a model you want to use, e.g.: `ollama pull llama3.1:8b`
  - Set `OLLAMA_MODEL` in `.env` to the pulled model name (e.g. `llama3.1:8b`)

- Python 3.10 or 3.11 (for local setup)
- Optional (for web search tools): Node.js 18+ and internet access
- Optional (for Zilliz Cloud): valid `MILVUS_URI` and `MILVUS_TOKEN`
- Local dev needs no external DB — defaults to Milvus Lite (file at `backend/milvus_data/milvus.db`).


## Directory Overview

- `api/main.py`: FastAPI app and endpoints
- `adaptive_rag/`: LangGraph graph, tools, and state definitions
- `ingest/`: chunking, embeddings, Milvus client/retriever, schema
- `requirements.txt`: Python dependencies
- `milvus_data/`: local Milvus Lite storage (created automatically)
- `venv/`: optional local virtual environment


## Environment Variables (.env)

Create `backend/.env` with your own values (do not commit secrets):

```
# Milvus (use ONE of the following)
# If both are set, uses Zilliz Cloud; otherwise falls back to local Milvus Lite (./milvus_data/milvus.db)
MILVUS_URI=
MILVUS_TOKEN=

# LLM model used by LangChain’s ChatOllama client
# Ensure the model is available on your Ollama-compatible endpoint or change to a locally available model
OLLAMA_MODEL=llama3.1:8b

# Optional: Tavily web search via MCP (enables hybrid retrieval from the web)
TAVILY_API_KEY=

# Optional: LangSmith traces (if you use it)
LANGSMITH_API_KEY=
```

Notes:
- If you don’t set `MILVUS_URI`/`MILVUS_TOKEN`, the app uses Milvus Lite stored at `backend/milvus_data/milvus.db`.
- If you don’t set `TAVILY_API_KEY`, web search tools are disabled (core RAG works fine).


## Setup Option 1: Normal (pip + your frontend)

1) Create and activate a virtualenv

```
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2) Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

3) Configure env

- Copy the template above into `backend/.env` with your own values.
- For local Milvus Lite you can keep `MILVUS_URI`/`MILVUS_TOKEN` empty.

4) Run the API

```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

5) Health check

- GET `http://localhost:8000/api/health`


If your frontend is a separate app, run it normally (e.g. `npm install && npm run dev`) and point it to `http://localhost:8000` for backend calls.