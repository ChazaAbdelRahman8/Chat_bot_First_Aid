"""Selective English/Arabic PaddleOCR support for failed native pages."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from statistics import fmean

from .quality_gate import normalize_text


class _PaddleXCompatibilityTextSplitter:
   

    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 0, **_: object
    ) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))

    def split_text(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        return [text[index : index + self.chunk_size] for index in range(0, len(text), step)]

    def split_documents(self, documents: list[object]) -> list[object]:
        return documents


def _install_modelscope_compatibility_shim() -> None:
    """Bypass a broken optional ModelScope/Torch import when OCR uses its cache."""
    try:
        import modelscope  # noqa: F401

        return
    except Exception:
        # ModelScope is only one of PaddleX's optional model download backends.
        # PaddleOCR can use cached models or another configured backend without it.
        modelscope_module = types.ModuleType("modelscope")

        def unavailable_snapshot_download(*_: object, **__: object) -> None:
            raise RuntimeError(
                "ModelScope is unavailable in this environment. Use cached PaddleOCR "
                "models or set PADDLE_PDX_MODEL_SOURCE=huggingface."
            )

        setattr(modelscope_module, "snapshot_download", unavailable_snapshot_download)
        sys.modules["modelscope"] = modelscope_module


def _install_langchain_compatibility_shims() -> None:
    """Bridge PaddleX 3.3 imports when LangChain 1.x is installed."""
    _install_modelscope_compatibility_shim()
    try:
        import langchain.docstore.document  # type: ignore[import-not-found] # noqa: F401
        import langchain.text_splitter  # type: ignore[import-not-found] # noqa: F401
        return
    except ImportError:
        pass

    try:
        from langchain_core.documents import Document
    except ImportError:
        return

    docstore = types.ModuleType("langchain.docstore")
    document_module = types.ModuleType("langchain.docstore.document")
    setattr(document_module, "Document", Document)
    setattr(docstore, "document", document_module)
    splitter_module = types.ModuleType("langchain.text_splitter")
    setattr(
        splitter_module,
        "RecursiveCharacterTextSplitter",
        _PaddleXCompatibilityTextSplitter,
    )
    sys.modules.setdefault("langchain.docstore", docstore)
    sys.modules.setdefault("langchain.docstore.document", document_module)
    sys.modules.setdefault("langchain.text_splitter", splitter_module)


def make_ocr_engine(lang: str):
    if lang not in {"en", "ar"}:
        raise ValueError(f"Unsupported OCR language: {lang!r}")
    _install_langchain_compatibility_shims()
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is required for routed pages. Install agent-system-a/requirements.txt."
        ) from exc

    try:
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
        )
    except TypeError:
        return PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)


def flatten_ocr_result(result) -> tuple[str, list[float]]:
    """Normalize PaddleOCR 2.x and 3.x result objects."""
    lines: list[str] = []
    scores: list[float] = []
    for page_result in result or []:
        if hasattr(page_result, "json"):
            payload = page_result.json
            payload = payload() if callable(payload) else payload
            data = payload.get("res", payload) if isinstance(payload, dict) else {}
            lines.extend(str(text) for text in (data.get("rec_texts", []) or []))
            scores.extend(float(score) for score in (data.get("rec_scores", []) or []))
        elif isinstance(page_result, list):
            for item in page_result:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[1], (list, tuple))
                ):
                    lines.append(str(item[1][0]))
                    scores.append(float(item[1][1]))
    return normalize_text("\n".join(lines)), scores


def render_page(source_path: Path, page_index: int, target: Path, dpi: int) -> Path:
    if target.exists():
        return target
    from .native_pdf import _pymupdf

    target.parent.mkdir(parents=True, exist_ok=True)
    pymupdf = _pymupdf()
    with pymupdf.open(source_path) as pdf:
        pixmap = pdf.load_page(page_index).get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(target)
    return target


def run_ocr(engine, image_path: Path) -> tuple[str, float | None]:
    try:
        raw_result = engine.predict(str(image_path))
    except AttributeError:
        raw_result = engine.ocr(str(image_path), cls=True)
    text, scores = flatten_ocr_result(raw_result)
    confidence = round(fmean(scores), 4) if scores else None
    return text, confidence
