# RAG specialist implementation

This package will contain the complete first-aid RAG capability that is later
wrapped by Agent System A's `first_aid_specialist`.

Current modules:

- `extraction/`: verified native PDF extraction, quality routing, selective
  English/Arabic PaddleOCR, checkpointing, and final validation.
- `vision/`: frozen heuristic candidate selection, manually forced inclusions,
  page rendering, a strict evidence-only Qwen2.5-VL prompt, deterministic output
  validation/rejection, and resumable per-page JSONL checkpointing.
- `chunking/`: reviewed text/vision merging and structure-aware chunks.
- `embedding/`: resumable, SHA-bound, L2-normalized BGE-M3 checkpoints.
- `indexing/`: Qdrant collection management, `doc_id` replacement, and validation.

## Phase 3 commands

Run these from the project root with the `FirstAidFinal` interpreter selected:

```powershell
python agent-system-a/scripts/prepare_vision.py
python agent-system-a/scripts/run_vision.py --model qwen2.5vl:7b --timeout 600
python agent-system-a/scripts/validate_vision.py
```

Progress, validation failures, Ollama connection failures, and full exception
tracebacks are written to `data/vision/logs/vision_pipeline.log`. Logs rotate at
10 MB with five backups. Use `--log-level DEBUG` for more verbose diagnostics.

The full run resumes automatically: successful records already present in
`data/vision/vision_records.jsonl` are skipped, while failed records are retried.
For a smoke test, add `--limit 1` or `--limit 5`.

## Phase 3 human review

```powershell
python agent-system-a/scripts/build_vision_review.py
python agent-system-a/scripts/serve_vision_review.py
```

The server binds only to `127.0.0.1` and autosaves decisions to
`data/vision/review/vision_review_decisions.json`. Raw model records are never
modified. After every queued record has a decision:

```powershell
python agent-system-a/scripts/apply_vision_review.py
python agent-system-a/scripts/validate_reviewed_vision.py
```

## Phase 4 merge checkpoint

```powershell
python agent-system-a/scripts/merge_text_vision.py
python agent-system-a/scripts/validate_merged_pages.py
```

This produces `data/processed/chunk/unified_page_records.jsonl`. It keeps all
extracted pages, merges only reviewed useful visuals, preserves both `page` and
`pdf_page`, and places `doc_id` on every page before chunk creation.

## Phase 4 structure-aware chunks

```powershell
python agent-system-a/scripts/run_chunking.py
python agent-system-a/scripts/validate_chunks.py
```

The production-only structure-aware variant uses a 450-token target, a strict
650-token maximum, and up to 80 tokens of structural overlap. Every chunk is
created with the Qdrant payload fields `doc_id`, `page`, `pdf_page`, `section`,
`chunk_id`, `language`, and `content_type`.

## Phase 5 BGE-M3 and Qdrant

Start Qdrant from the project root, then create the embedding checkpoint and
load it into the collection:

```powershell
docker compose up -d qdrant
python agent-system-a/scripts/run_embeddings.py --batch-size 4
python agent-system-a/scripts/validate_embeddings.py
python agent-system-a/scripts/index_qdrant.py --url http://localhost:6333
python agent-system-a/scripts/validate_qdrant.py --url http://localhost:6333
```

`run_embeddings.py` writes `bge_m3_embeddings.npy`, `embedding_ids.jsonl`, and
`embedding_manifest.json` under `data/processed/embedding/`. It resumes after the
last completed batch and refuses to reuse a checkpoint if the chunk SHA256,
model, row count, or dimension changed. Use `--force` only for an intentional
checkpoint replacement.

The Qdrant loader creates a 1,024-dimensional cosine collection and creates the
keyword payload index on `doc_id` before upserting. By default it replaces only
the `doc_id` values present in the incoming artifact; unrelated documents are
not deleted or re-embedded. Point IDs are deterministic UUIDs derived from
`chunk_id`, while the readable `chunk_id` and source text remain in payload.

## Phase 6 retrieval

Build and validate the separate BM25 checkpoint, then run one or more live
queries against BM25 and Qdrant:

```powershell
python agent-system-a/scripts/build_bm25.py
python agent-system-a/scripts/validate_bm25.py
python agent-system-a/scripts/validate_retrieval.py --url http://localhost:6333
python agent-system-a/scripts/query_retrieval.py `
  --query "How do I control severe bleeding?" `
  --query "كيف أتعامل مع شخص فاقد للوعي؟"
```

The checkpoint is stored under `data/processed/retrieval/` and is bound to the
SHA256 and order of `data/processed/chunk/chunks_structure_aware.jsonl`. English
queries use dense top five plus up to three unique BM25 passages. Queries that
contain Arabic script use ten candidates from each retriever, reciprocal-rank
fusion with `k=60`, and return the fused top five. Both paths deduplicate by
`chunk_id` and return citation-ready `doc_id`, `page`, `pdf_page`, and `section`
metadata. `--doc-id` applies the same source filter to both retrievers.

For future uploads, replace or delete the document's chunks by stable `doc_id`,
then update both Qdrant and the BM25 checkpoint. Updating only one index is an
invalid ingestion state.

## Phase 7 grounded generation

Generation uses local Ollama `qwen3:4b`. Retrieved passages receive stable labels
`[S1]`, `[S2]`, and so on. The model must return structured JSON, answer only from
those passages, use the question's language, cite every factual statement, and
abstain when evidence is insufficient. Deterministic validation rejects unknown
or mismatched citation labels and retries once with the validation error.

Start Docker Desktop and Ollama, then run:

```powershell
python agent-system-a/scripts/validate_generation.py
python agent-system-a/scripts/query_rag.py `
  --query "How should severe external bleeding be controlled?"
```

The output distinguishes `retrieval_induced` abstention (no usable retrieved
text) from `model_generated` abstention (the model determines that available
evidence does not answer the question). Policy-gate abstention will be added in
Phase 8 before the request reaches retrieval or generation.

Planned module, added only when its phase is implemented and tested:

- `safety/`
