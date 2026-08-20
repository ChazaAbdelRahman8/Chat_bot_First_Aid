# First-Aid RAG API

The API exposes the completed RAG as a service. It keeps the original seven
manuals frozen and manages newly uploaded PDFs in a separate corpus layer.

## Start locally

Qdrant and Ollama must already be running, with `qwen3:4b` available. From the
repository root in the `FirstAidFinal` Conda environment:

```powershell
$env:PYTHONPATH = "agent-system-a/src"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger interface.
Alternatively, start the API and Qdrant with:

```powershell
docker compose up --build agent-system-a
```

The Docker service reaches host Ollama through `host.docker.internal:11434`.

## Query System A

`POST /v1/chat` sends a text request through an explicit LangGraph `StateGraph`.
Its named input guard, structured supervisor, RAG/web/visual ReAct specialists,
answer agent, and output guard are visible in the exported Mermaid graph. The
response includes a `route` array and an additive `orchestration` diagnostic.

`POST /v1/chat/stream` accepts the same JSON body and returns Server-Sent Events
(`text/event-stream`). It emits `status`, periodic `heartbeat`, `result`, and
`done` events. The `result` data is the same validated JSON payload returned by
`POST /v1/chat`, and the completed turn is persisted to the same MongoDB
conversation. If processing fails, the stream emits a generic `error` event
without returning an internal exception or stack trace.

```powershell
$body = @{ query = "What should I do for severe bleeding?" } | ConvertTo-Json
Invoke-WebRequest -Method Post `
    -Uri http://127.0.0.1:8000/v1/chat/stream `
    -ContentType "application/json" `
    -Body $body
```

```json
{
  "query": "What should I do for a minor burn?",
  "doc_id": null
}
```

For first-aid questions, the RAG specialist preserves the existing retrieval
passages, validated citations, safety decision, and abstention information.
Questions explicitly requiring changing/current information are delegated to
the web-search specialist.

`POST /v1/chat/visual` accepts multipart form fields `query` and `image`. It
supports JPEG, PNG, and WebP images up to 15 MiB. The supervisor delegates to
the visual specialist and may also call RAG when first-aid action is requested.
Uploaded query images are held in a temporary file and removed after the run.

The specialists use LangChain's current `create_agent` API and remain compiled
ReAct graphs. The top-level router uses LangGraph `StateGraph` and fans selected
specialists out in parallel before joining at the answer node. With
`SUPERVISOR_PROVIDER=auto`, Groq is preferred when `GROQ_API_KEY` is present and
local Ollama is the fallback. The default Groq router is `openai/gpt-oss-120b`.

After specialist execution, a fourth ReAct Answer Agent performs final
synthesis using only specialist evidence. Its only tool validates citation
labels and web URLs; it has no retrieval, vision, or search access. RAG-only
answers use a lossless pass-through so the proven medical answer and validated
citations are not rewritten. Multi-specialist, web, and visual responses use
the Answer Agent and fail closed to abstention if reference validation fails.

Export the actual compiled orchestration graph with:

```powershell
python agent-system-a/scripts/show_supervisor_graph.py > supervisor_graph.mmd
```

Conversation turns retain their `conversation_id` in MongoDB. The complete
history is queryable through `/v1/conversations`; only the most recent 12
messages are injected into each graph invocation.

## Add a PDF

`POST /v1/documents` accepts `multipart/form-data`:

- `file`: PDF file (maximum 100 MiB by default)
- `doc_id`: stable source key using letters, digits, `_`, or `-`
- `ocr_lang`: `en` or `ar`
- `scope`: optional metadata, default `managed`
- `replace`: set to `true` to replace an existing API-managed document

The endpoint returns HTTP 202 and a `job_id`. Poll
`GET /v1/documents/jobs/{job_id}` until its status is `complete` or `failed`.
The job performs native extraction, quality gating, selective PaddleOCR,
optional VLM description, page merge, structure-aware chunking, BGE-M3
embedding, Qdrant upsert, and BM25 rebuild.

The original seven registered manuals cannot be overwritten through this API.
Uploads and their resumable metadata live in `data/managed_documents/`.

## List and delete managed documents

- `GET /v1/documents` lists API-managed documents.
- `DELETE /v1/documents/{doc_id}` removes that document's Qdrant points using
  its indexed `doc_id`, removes its BM25 chunks, and deletes its managed files.
- `GET /health` verifies the loaded models and Qdrant collection.

Add/delete operations hold a shared runtime lock while changing Qdrant and
BM25, so a chat request cannot observe only half of an index update.

## Important deployment limitation

The current background-job mechanism is intended for one local API worker. For
multi-worker or distributed production deployment, replace FastAPI background
tasks with a durable queue (for example Redis plus Celery/RQ) and add API
authentication before exposing document-management endpoints publicly.
