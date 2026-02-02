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


## Setup Option 2: Docker (simple)

1) Build the image (from repo root):

```
docker build -t adaptive-rag-backend -f backend/Dockerfile .
```

2) Run it. Make sure Ollama is running on your host (`ollama serve`). Pass the Ollama base URL into the container so it can reach the host. The container runs both the backend (8000) and the Vite dev server (5173):

- macOS/Windows:

```
docker run --rm -p 8000:8000 \
  --env-file backend/.env \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v $(pwd)/backend/milvus_data:/app/milvus_data \
  adaptive-rag-backend
```

- Linux (add host gateway mapping):

```
docker run --rm -p 8000:8000 \
  --env-file backend/.env \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v $(pwd)/backend/milvus_data:/app/milvus_data \
  adaptive-rag-backend
```

3) Open the frontend at `http://localhost:5173` (dev server) or call the backend at `http://localhost:8000`.


## Using the API

- Ingest one or multiple files into a session. If you omit `thread_id`, the server generates one.

```
# Single file
curl -F "file=@/path/to/doc.pdf" \
     -F "document_role=resume" \
     http://localhost:8000/api/ingest

# Multiple files
curl -F "files=@/path/to/doc1.pdf" -F "files=@/path/to/doc2.docx" \
     -F "document_role=paper" \
     http://localhost:8000/api/ingest
```

The response includes `thread_id` to use for chatting in the same session.

- Stream chat response (Server‑Sent Events). Replace `THREAD_ID` with your session id.

```
curl -N -H "Content-Type: application/json" \
     -d '{
           "question": "Summarize the key findings",
           "thread_id": "THREAD_ID",
           "model": "llama3.1:8b",
           "max_retries": 2
         }' \
     http://localhost:8000/api/chat
```

- End a session and clean up its collections/state:

```
curl -X DELETE http://localhost:8000/api/session/THREAD_ID
```


## Docker (optional)

Below is a working Dockerfile example for a CPU‑only build. It includes Node.js so `npx mcp-remote` can run Tavily MCP tools when `TAVILY_API_KEY` is set.

Save this as `backend/Dockerfile` if you want to build now:

```
# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps for building wheels + Node for MCP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
    nodejs npm \
  && rm -rf /var/lib/apt/lists/*

# Optional: pin npm to latest LTS
RUN npm install -g npm@latest

WORKDIR /app

# Install Python deps first for better layer caching
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/ .

# Expose API port
EXPOSE 8000

# Milvus Lite local storage (mount a volume here in compose)
VOLUME ["/app/milvus_data"]

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and run

```
# From repo root
docker build -t adaptive-rag-backend -f backend/Dockerfile .

# Run with local .env and a volume for Milvus Lite data
docker run --rm -p 8000:8000 \
  --env-file backend/.env \
  -v $(pwd)/backend/milvus_data:/app/milvus_data \
  adaptive-rag-backend
```

### docker-compose

Use the included `backend/docker-compose.yml` to run both backend and frontend:

```
cd backend
docker compose up --build
```

This will:
- Build the image from `backend/Dockerfile`
- Map ports `8000` (API) and `5173` (frontend)
- Mount `./milvus_data` for Milvus Lite persistence
- Set `OLLAMA_BASE_URL=http://host.docker.internal:11434` so the container can reach your host’s Ollama
- Add `extra_hosts` for Linux to resolve `host.docker.internal`

Open:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000


## Common Issues

- torch wheels on Apple Silicon: prefer Python 3.11 and the slim Debian base; CPU wheels install automatically. For GPU you’d need a different base and CUDA/cuDNN.
- Tavily MCP tools do not load: ensure `TAVILY_API_KEY` is in `.env` and internet access is available; Node.js must be present (Dockerfile above includes it).
- Milvus auth errors: if using Zilliz Cloud, verify `MILVUS_URI` and `MILVUS_TOKEN`. Otherwise remove them to use local Milvus Lite.
- Frontend not served: ensure `backend/frontend/index.html` exists; otherwise the root returns a JSON message.


## Development Tips

- The app caches LangGraph graphs per `(model, max_retries)` and a retriever instance to avoid reloading models; restart the server if you change model configs.
- Each chat `thread_id` creates isolated collections, cleaned up via `/api/session/{thread_id}`.
- Set `USE_LIGHT_TEXT_MODEL=1` (default via code) to use the lighter text embedding model during ingestion.
