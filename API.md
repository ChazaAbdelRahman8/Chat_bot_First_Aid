# First-Aid RAG API

## Start locally

From the project root with the `FirstAidFinal` Conda environment active:

```powershell
$env:PYTHONPATH="agent-system-a\src"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Qdrant must be running on `http://localhost:6333`, and Ollama must expose `qwen3:4b`. PDF visual ingestion additionally uses `qwen2.5vl:7b` when `INGEST_VISION=true`.

Interactive documentation is available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Health: `http://127.0.0.1:8000/health`

## Query the RAG

```powershell
$body = @{
  query = "How should severe external bleeding be controlled?"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/v1/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

To restrict retrieval to one source, include `doc_id` in the request body.

The response contains the retrieval route, retrieved chunks, grounded answer, citation metadata, language, validation attempts, and abstention category.

## Stream a chat response with SSE

`POST /v1/chat/stream` accepts the same request body and responds as
`text/event-stream`. It emits `status`, optional `heartbeat`, `result`, and
`done` events. The `result` event contains the same validated response as the
normal chat endpoint and is persisted under the returned `conversation_id`.

```powershell
$body = @{ query = "How should severe bleeding be controlled?" } | ConvertTo-Json
Invoke-WebRequest `
  -Uri http://127.0.0.1:8000/v1/chat/stream `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Upload and ingest a PDF

Uploads are asynchronous because OCR, vision processing, and embedding may take several minutes.

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/documents `
  -F "file=@C:\path\new_manual.pdf;type=application/pdf" `
  -F "doc_id=NEW_MANUAL_2026" `
  -F "ocr_lang=en" `
  -F "scope=managed" `
  -F "replace=false"
```

The endpoint returns HTTP 202 and a `job_id`. Check it with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/documents/jobs/JOB_ID
```

Stages are:

1. SHA256 and registration
2. Native PyMuPDF extraction and quality gate
3. Selective PaddleOCR for failed pages
4. Heuristic visual-candidate detection
5. Optional Qwen2.5-VL description and deterministic validation
6. Text/eligible-vision merge
7. Structure-aware chunking
8. Normalized BGE-M3 embedding
9. Qdrant filtered replacement by `doc_id`
10. Managed chunk checkpoint update and in-memory BM25 rebuild

Ambiguous VLM output is not automatically merged. It is excluded until a future human-review workflow approves it.

Use `replace=true` to replace an existing API-managed document. The seven frozen source documents cannot be replaced or deleted through these endpoints.

## List managed documents

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/documents
```

## Delete a managed document

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/v1/documents/NEW_MANUAL_2026 `
  -Method Delete
```

Deletion removes the document's Qdrant points using its indexed `doc_id`, removes its managed chunks, rebuilds the active BM25 index, and removes its uploaded PDF and work files. Only API-managed documents can be deleted.

## Docker

```powershell
docker compose up --build qdrant agent-system-a
```

The first container startup may download BGE-M3 into the persistent `huggingface_cache` volume. Ollama remains on the host and is reached through `host.docker.internal` by default.

For a faster ingestion test without vision:

```powershell
$env:INGEST_VISION="false"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
