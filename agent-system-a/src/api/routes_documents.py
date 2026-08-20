"""Upload, inspect, and delete API-managed source documents."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from rag.embedding.bge_m3 import sha256_file

from .ingestion import DocumentIngestor, normalize_doc_id, utc_now
from .schemas import DocumentResponse, JobResponse


router = APIRouter(prefix="/v1", tags=["documents"])
MAX_UPLOAD_BYTES = int(os.getenv("MAX_PDF_UPLOAD_BYTES", str(100 * 1024 * 1024)))


def runtime_dependency():
    raise RuntimeError("RAG runtime is not configured")


def ingestor_dependency():
    raise RuntimeError("document ingestor is not configured")


def _base_doc_ids(project_root: Path) -> set[str]:
    registry = json.loads(
        (project_root / "data" / "source_registry.json").read_text(encoding="utf-8")
    )
    return {str(row["doc_id"]) for row in registry["documents"]}


async def _save_pdf(upload: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".uploading")
    total = 0
    try:
        with temporary.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"PDF exceeds {MAX_UPLOAD_BYTES} bytes")
                handle.write(chunk)
        with temporary.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise HTTPException(400, "uploaded file is not a PDF")
        os.replace(temporary, target)
    finally:
        await upload.close()
        temporary.unlink(missing_ok=True)


def _run_job(
    ingestor: DocumentIngestor, job_id: str, doc_id: str, filename: str,
    pdf_path: Path, ocr_lang: str, scope: str,
) -> None:
    def progress(stage: str, detail: dict) -> None:
        ingestor.store.set_job(job_id, {
            "job_id": job_id, "doc_id": doc_id, "status": "running",
            "detail": {"stage": stage, **detail}, "updated_at": utc_now(),
        })

    try:
        metadata = ingestor.ingest(
            doc_id=doc_id, filename=filename, pdf_path=pdf_path,
            ocr_lang=ocr_lang, scope=scope, progress=progress,
        )
        ingestor.store.set_job(job_id, {
            "job_id": job_id, "doc_id": doc_id, "status": "complete",
            "detail": metadata, "updated_at": utc_now(),
        })
    except Exception as exc:
        ingestor.store.set_job(job_id, {
            "job_id": job_id, "doc_id": doc_id, "status": "failed",
            "detail": {"error": f"{type(exc).__name__}: {exc}"},
            "updated_at": utc_now(),
        })


@router.post("/documents", response_model=JobResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    ocr_lang: str = Form("en"),
    scope: str = Form("managed"),
    replace: bool = Form(False),
    runtime=Depends(runtime_dependency),
    ingestor: DocumentIngestor = Depends(ingestor_dependency),
) -> dict:
    try:
        doc_id = normalize_doc_id(doc_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if ocr_lang not in {"en", "ar"}:
        raise HTTPException(422, "ocr_lang must be 'en' or 'ar'")
    if doc_id in _base_doc_ids(runtime.project_root):
        raise HTTPException(409, "frozen source documents cannot be replaced through this API")
    existing = runtime.store.registry().get(doc_id)
    if existing and not replace:
        raise HTTPException(409, "doc_id already exists; set replace=true to replace it")
    suffix = Path(file.filename or "upload.pdf").suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(400, "only .pdf files are accepted")
    target = runtime.store.upload_dir / f"{doc_id}.pdf"
    await _save_pdf(file, target)
    job_id = uuid.uuid4().hex
    queued = {
        "job_id": job_id, "doc_id": doc_id, "status": "queued",
        "detail": {"filename": file.filename, "sha256": sha256_file(target)},
        "updated_at": utc_now(),
    }
    runtime.store.set_job(job_id, queued)
    background_tasks.add_task(
        _run_job, ingestor, job_id, doc_id, file.filename or target.name,
        target, ocr_lang, scope,
    )
    return queued


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(runtime=Depends(runtime_dependency)) -> list[dict]:
    return list(runtime.store.registry().values())


@router.get("/documents/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str, runtime=Depends(runtime_dependency)) -> dict:
    job = runtime.store.jobs().get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    ingestor: DocumentIngestor = Depends(ingestor_dependency),
) -> dict:
    try:
        return await run_in_threadpool(ingestor.delete, doc_id)
    except KeyError as exc:
        raise HTTPException(404, "managed document not found") from exc
